#!/usr/bin/env python3
"""
Audits and corrects placements of nodes in a noun taxonomy using LLM queries.
For each node, determines whether it is correctly placed under its parent via
an IS-A relationship, flags errors, and proposes better placements.

Features:
- Progress tracking with live updates
- Checkpoint saves at configurable intervals
- Error handling and retry logic
- Resume support (skips already-flagged paths)
- Configurable sampling modes

"""

import csv
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from dotenv import load_dotenv
from nltk.corpus import wordnet as wn
from openai import OpenAI
from tqdm import tqdm

from config import *
from utils_hierarchy import (
    load_json,
    parse_description,
    traverse_nodes,
    find_parent,
    get_siblings,
    safe_list,
)

load_dotenv()
wn.ensure_loaded()

# PROMPTS

SYSTEM_PROMPT = """
You are auditing AND correcting a noun taxonomy.

Each node represents a conceptual category of entities.

Your tasks:
1. Determine if the node is correctly placed under its parent via an IS-A relationship.
2. If correct → return OK.
3. If incorrect:
   - classify the error, with reasoning
   - propose the BEST new placement, with reasoning:
        - deepest valid node
        - reuse existing hierarchy where possible
        - only create a new category if absolutely necessary

Error types:
1. TYPE_MISMATCH
The node belongs to a fundamentally different type than its parent.

2. HIERARCHY_MISMATCH
The node is not a subtype of the parent.

3. GRANULARITY_ERROR
The node is too broad or too specific relative to its siblings or parent.

Error examples:

Input node: "policy" under "physical entity"
Output
- error_type: "TYPE_MISMATCH"
- reason: "policy is a conceptual noun, not necessarily describing a physical object"

Input node: "engine" under "vehicle"
Output
- error_type: "HIERARCHY_MISMATCH"
- reason: "an engine is not a type of vehicle, but is instead a type of equipment or motor"

Input node: "entity" under "document"
Example Output
- error_type: "GRANULARITY_ERROR"
- reason: "an entity describes a larger category of nouns than document does, and is neither a subtype nor a part of a document"


Be conservative: only flag clear errors, as placements are expected to be imperfect
Use conceptual meaning, not just surface words
Use the provided definition, supersenses, example sentences, and siblings to understand parent and node meanings

Return STRICT JSON only.
"""

USER_PROMPT = """
## Hierarchy Path
{path_tail}

## Parent Node
{parent_label}
Definition: {parent_definition}
Synonyms: {parent_synonyms}
Supersenses: {parent_supersenses}
Example: {parent_example}

## Siblings
{siblings}

## Node
{node_name}

## Node Info
Definition: {definition}
Synonyms: {synonyms}
Supersenses: {supersenses}
Example: {example}

## WordNet
Hypernyms: {hypernyms}

## Full Hierarchy
{hierarchy_json}

## Output JSON

{{
  "verdict": "OK" or "ERROR",
  "error_type": "TYPE_MISMATCH" | "HIERARCHY_MISMATCH" | "GRANULARITY_ERROR" | "N/A",
  "reason": "...",
  "new_path_isa": "",
  "placement_reason": "..."
}}
"""

OUTPUT_FIELDS = [
    "path", "node", "definition", "example",
    "verdict", "error_type", "reason", "new_path_isa", "placement_reason"
]


# AUDITOR CLASS

class NounHierarchyAuditor:
    """
    Audits noun taxonomy node placements using LLM queries.
    Flags IS-A errors and proposes corrected placements.
    """

    def __init__(self,
                 hierarchy_json: str = HIERARCHY_JSON,
                 output_csv: str = PLACEMENTS_CSV,
                 llm_model: str = MODEL,
                 max_retries: int = MAX_RETRIES,
                 temperature: float = TEMPERATURE,
                 max_tokens: int = MAX_TOKENS,
                 checkpoint_interval: int = CHECKPOINT_INTERVAL,
                 max_siblings: int = MAX_SIBLINGS):
        """
        Initialize the auditor.

        Args:
            hierarchy_json:      Path to the condensed hierarchy JSON
            output_csv:          Path to write flagged placements CSV
            llm_model:           OpenAI model to use
            max_retries:         Max retries for failed API calls
            temperature:         LLM sampling temperature
            max_tokens:          Max tokens per LLM response
            checkpoint_interval: Save output every N nodes processed
            max_siblings:        Max sibling nodes to include in prompt
        """
        self.hierarchy_json = hierarchy_json
        self.output_csv = output_csv
        self.llm_model = llm_model
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.checkpoint_interval = checkpoint_interval
        self.max_siblings = max_siblings

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)

        print("🔧 Noun Hierarchy Auditor initialized")
        print(f"   Hierarchy:   {self.hierarchy_json}")
        print(f"   Output CSV:  {self.output_csv}")
        print(f"   Model:       {self.llm_model}")

    # LLM

    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make an LLM API call with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"    🤖 LLM API call (attempt {attempt})")
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                )
                result = response.choices[0].message.content.strip()
                print(f"    ✅ LLM response: {result[:100]}{'...' if len(result) > 100 else ''}")
                return result
            except Exception as e:
                print(f"    ⚠️  Attempt {attempt} failed: {e}")
                time.sleep(2 * attempt)
        print("    ❌ All retries exhausted.")
        return ""

    def parse_json_safe(self, txt: str) -> Optional[dict]:
        """Strip markdown fences and parse JSON."""
        txt = txt.strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt)
        try:
            return json.loads(txt)
        except Exception:
            return None

    # Helpers

    def get_wordnet_hypernyms(self, word: str) -> list:
        """Return hypernym lemma names for the top synsets of a noun."""
        synsets = wn.synsets(word, pos=wn.NOUN)
        hypernyms = set()
        for syn in synsets[:3]:
            hypernyms.update([h.lemma_names()[0] for h in syn.hypernyms()])
        return list(hypernyms)

    def get_valid_path_prefix(self, full_path: list, flagged_paths: set) -> list:
        """
        If any ancestor of this node was already flagged as incorrect,
        trim the path display so the LLM knows the context is unreliable.
        """
        for i in range(len(full_path) - 1):
            ancestor_path = " > ".join(full_path[:i + 1])
            if ancestor_path in flagged_paths:
                return full_path[i:]
        return full_path

    def build_prompt(self, item: dict, root: dict,
                     flagged_paths: set, hierarchy_json_str: str) -> Optional[str]:
        """Assemble the USER_PROMPT string for one node. Returns None to skip."""
        full_path = item["path"]

        # Skip nodes too close to the root (human vetted)
        if len(full_path) <= 4:
            return None

        node      = item["node"]
        node_name = item["title"]
        parent    = find_parent(root, full_path)

        siblings = safe_list(get_siblings(parent, node_name), self.max_siblings)

        definition, synonyms, supersenses, example = parse_description(
            node.get("description", "")
        )
        parent_definition, parent_synonyms, parent_supersenses, parent_example = parse_description(
            (parent or {}).get("description", "")
        )

        hypernyms    = self.get_wordnet_hypernyms(node_name)
        valid_prefix = self.get_valid_path_prefix(full_path, flagged_paths)

        if len(valid_prefix) < len(full_path):
            path_context = (
                f"{' > '.join(valid_prefix)}\n"
                f"(Earlier ancestors were flagged as incorrect)"
            )
        else:
            path_context = " > ".join(full_path)

        return USER_PROMPT.format(
            path_tail=path_context,
            parent_label=(parent or {}).get("title", ""),
            parent_definition=parent_definition,
            parent_synonyms=parent_synonyms,
            parent_supersenses=parent_supersenses,
            parent_example=parent_example,
            siblings=siblings,
            node_name=node_name,
            definition=definition,
            synonyms=synonyms,
            supersenses=supersenses,
            example=example,
            hypernyms=hypernyms,
            hierarchy_json=hierarchy_json_str,
        )

    # Core processing

    def process_node(self, item: dict, root: dict,
                     flagged_paths: set, hierarchy_json_str: str) -> Optional[dict]:
        """Audit a single node. Returns a result dict, or None to skip."""
        full_path = item["path"]
        node_name = item["title"]
        node      = item["node"]

        user_prompt = self.build_prompt(item, root, flagged_paths, hierarchy_json_str)
        if user_prompt is None:
            return None

        definition, _, _, example = parse_description(node.get("description", ""))

        print(f"\n🔍 Auditing: {' > '.join(full_path[-3:])}")

        raw    = self.call_llm(SYSTEM_PROMPT, user_prompt)
        parsed = self.parse_json_safe(raw)

        if not parsed:
            print(f"      ⚠️  Could not parse JSON for: {node_name}")
            return None

        verdict = parsed.get("verdict", "")
        if "error_type" in parsed:
            print(f"      • Verdict:    {verdict}")
            print(f"      • Error type: {parsed.get('error_type')}")
            print(f"      • Reason:     {parsed.get('reason', '')[:80]}")

        return {
            "path":             " > ".join(full_path),
            "node":             node_name,
            "definition":       definition,
            "example":          example,
            "verdict":          verdict,
            "error_type":       parsed.get("error_type"),
            "reason":           parsed.get("reason"),
            "new_path_isa":     parsed.get("new_path_isa", ""),
            "placement_reason": parsed.get("placement_reason"),
        }

    # Main run

    def run(self) -> bool:
        """
        Load the hierarchy, sample nodes per config, audit each one,
        and write errors to the output CSV.
        """
        try:
            # Load hierarchy
            print(f"📊 Loading hierarchy: {self.hierarchy_json}")
            tree = load_json(self.hierarchy_json)
            root = tree.get("Entity", tree)
            hierarchy_json_str = json.dumps(tree, ensure_ascii=False)

            # Traverse all nodes
            nodes = traverse_nodes(root)
            print(f"   Found {len(nodes)} total nodes")

            # Apply sampling
            if SAMPLE_MODE == "first_n":
                nodes = nodes[:SAMPLE_SIZE]
            elif SAMPLE_MODE == "random":
                random.seed(RANDOM_SEED)
                nodes = random.sample(nodes, min(SAMPLE_SIZE, len(nodes)))
            elif SAMPLE_MODE == "range":
                nodes = nodes[RANGE_START:RANGE_END]

            print(f"🔄 Processing {len(nodes)} nodes (sample_mode={SAMPLE_MODE!r})")
            print("=" * 60)

            flagged_paths  = set()
            flagged_count  = 0
            skipped_count  = 0

            out    = open(self.output_csv, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(out, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = {
                    executor.submit(
                        self.process_node, n, root, flagged_paths, hierarchy_json_str
                    ): n
                    for n in nodes
                }

                with tqdm(total=len(futures), desc="Auditing nodes", unit="node") as pbar:
                    for i, future in enumerate(as_completed(futures)):
                        result = future.result()

                        if result is None:
                            skipped_count += 1
                        elif result["verdict"] == "ERROR":
                            flagged_paths.add(result["path"])
                            writer.writerow(result)
                            flagged_count += 1

                        if i % self.checkpoint_interval == 0:
                            out.flush()

                        pbar.update(1)
                        pbar.set_postfix({"flagged": flagged_count, "skipped": skipped_count})

            out.close()
            self.print_summary(len(nodes), flagged_count, skipped_count)
            return True

        except Exception as e:
            print(f"❌ Error during audit run: {e}")
            return False

    def print_summary(self, total: int, flagged: int, skipped: int):
        """Print a summary of the audit results."""
        print("\n" + "=" * 60)
        print("📊 AUDIT SUMMARY")
        print("=" * 60)
        audited = total - skipped
        print(f"Total nodes sampled: {total}")
        print(f"Nodes audited:       {audited}")
        print(f"Nodes skipped:       {skipped}  (too close to root)")
        print(f"Errors flagged:      {flagged}")
        if audited > 0:
            print(f"Error rate:          {flagged / audited * 100:.1f}%")
        print(f"\n✅ Results saved to: {self.output_csv}")


# MAIN

def main():
    print("🚀 Noun Hierarchy Auditor")
    print("=" * 50)

    if not os.path.exists(HIERARCHY_JSON):
        print(f"❌ Error: Hierarchy file '{HIERARCHY_JSON}' not found!")
        return 1

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("   Please set your OpenAI API key in a .env file or environment variable.")
        return 1

    try:
        auditor = NounHierarchyAuditor()
        success = auditor.run()

        if success:
            print("\n🎉 Audit completed successfully!")
            return 0
        else:
            print("\n💥 Audit failed!")
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️  Audit interrupted by user.")
        print("   Partial results have been saved to the output CSV.")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())