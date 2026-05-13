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
- Depth-level auditing (audit one level at a time for human review)
- Fix type classification (MERGE, PLACE_UNDER, ADD_NEW_AND_PLACE, TYPE_MISMATCH)
- Path validation with LLM retry on invalid paths

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
   - classify the fix type, with reasoning
   - propose the BEST new placement, with reasoning:
        - deepest valid node
        - reuse existing hierarchy where possible
        - only create a new category if absolutely necessary
4. If mostly correct but the title is ambiguous, suggest a clearer title that:
   - disambiguates meaning
   - preserves scope
   - avoids overlap

ERROR TYPES:
1. TYPE_MISMATCH
   The node does not belong in Physical Entity at all. It should be in a different
   sub-ontology. Our full noun hierarchy has four sub-ontologies: Physical Entity,
   Information, Activity, and Actor. For example, "policy" belongs in Information,
   "walking" in Activity, "doctor" in Actor. In new_path_isa, name the correct
   sub-ontology and a plausible path within it.

2. HIERARCHY_MISMATCH
   The node is a real physical entity but is not a subtype of its current parent.
   Example: "engine" under "vehicle" - an engine is not a type of vehicle.

3. GRANULARITY_ERROR
   The node is too broad or too specific relative to its siblings or parent.
   Example: "gem" as a direct child of "matter" - it is a type of matter, 
   but should be under more levels of parents like "solid" and "crystal"

FIX TYPES:
1. MERGE
    The node is so similar in meaning to an existing sibling or nearby node that
    it should be merged into that node (its synset added in parentheses to the existing
    entry). new_path_isa must point to the existing node to merge into.

2. PLACE_UNDER
    The node is under the wrong parent. It should be moved to an existing node
    elsewhere in the hierarchy. new_path_isa is the full path to the correct existing parent.

3. ADD_NEW_AND_PLACE
    In very rare cases, the best correct parent doesn't exist yet, and the 
    potential existing path is too broad or would not be MECE. One or two new [virtual]
    intermediate nodes must be created. new_path_isa includes those new [virtual] nodes.

4. TYPE_MISMATCH
    The node doesn't belong in Physical Entity at all. new_path_isa
    gives a hypothetical path in the correct sub-ontology.

NEW PATH GUIDELINES:
- Use the deepest valid existing node whenever possible
- new_path_isa is the path to the DESTINATION node — either the parent to place the
  node under, or (for MERGE) the existing node to merge into. It must NEVER include 
  the node being audited itself as the last segment.
- CRITICAL: Every segment must be copy-pasted EXACTLY from the hierarchy JSON keys
  including synsets and [virtual] tag. Do NOT construct paths from WordNet knowledge or memory.
- You may introduce up to 2 new [virtual] intermediate nodes if genuinely needed for
  MECE structure, but avoid unnecessary depth inflation.
- For MERGE: new_path_isa must point to the EXISTING node to merge into. No new nodes.
  Example: to merge "bit" into "fragment", new_path_isa ends at "fragment (fragment.n.01)".
- For PLACE_UNDER: new_path_isa ends at the existing parent to move the node under.
  Example: to place "excavation" under "structure", new_path_isa ends at "structure (structure.n.01)".
- For TYPE_MISMATCH: new_path_isa should name the other sub-ontology and a hypothetical
  path within it (e.g. "Information > Document > Policy"). No validation is done on
  this path since it is a different ontology.

PATH FORMAT — this is critical and will be validated:
- Separate levels with " > "
- Every node name must EXACTLY match a key in the "specializations" dict at that level
  of the provided hierarchy JSON, including the synset in parentheses and the [virtual]
  tag where present.
- Example of a valid path:
    [virtual] physical_entity (physical_entity.n.01) > artifact (artifact.n.01) > infrastructure > structure (structure.n.01)
- For ADD_NEW_AND_PLACE, all nodes except the final one (or final two if two new virtual
  nodes are needed) must exist exactly. New nodes must be tagged [virtual].
- If the path is invalid (node names don't match the JSON), the response will be
  rejected and you will be asked to try again. Use the full hierarchy JSON provided
  to copy-paste exact node names.

Title improvement example:
- Input node: window
- Possible Outputs: window (architectural) or window (UI)

GUIDELINES
Be conservative: only flag clear errors, as placements are expected to be imperfect. 
If the node could reasonably be interpreted as a subtype of the parent, return OK.

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

## Siblings (up to {max_siblings})
{siblings}

## Node Being Audited
{node_name}

## Node Info
Definition: {definition}
Synonyms: {synonyms}
Supersenses: {supersenses}
Example: {example}

## WordNet Hypernyms
{hypernyms}

## Full Hierarchy JSON
(Use exact key names from this when constructing new_path_isa)
{hierarchy_json}

## Output JSON

{{
  "verdict": "OK" or "ERROR",
  "error_type": "TYPE_MISMATCH" | "HIERARCHY_MISMATCH" | "GRANULARITY_ERROR" | "N/A",
  "reason": "...",
  "fix_type": "MERGE" | "PLACE_UNDER" | "ADD_NEW_AND_PLACE" | "TYPE_MISMATCH" | "N/A",
  "new_path_isa": "",
  "placement_reason": "...",
  "title_suggestion": "",
  "title_reason": ""
}}
"""

RETRY_PROMPT = """
Your previous response was rejected because new_path_isa was invalid.

Reason: {validation_error}

Your previous new_path_isa was: {bad_path}

Rules for new_path_isa:
- new_path_isa is the path to the DESTINATION — the parent to place under, or the
  node to merge into. It must NEVER end with the node being audited itself.
- Separate levels with " > "
- Every segment must be copy-pasted EXACTLY from the hierarchy JSON keys — including
  synsets in parentheses and [virtual] tags. Do NOT use WordNet IDs or memory.
  Open the JSON provided in the original prompt, find the correct node, copy its key.
- For MERGE: all nodes must already exist. The path ends at the merge target.
- For PLACE_UNDER: all nodes must already exist. The path ends at the new parent.
- For ADD_NEW_AND_PLACE: only the final 1-2 nodes may be new; they must be tagged
  [virtual] and include a synset, e.g. [virtual] bridge (bridge.n.01).

Please re-examine the hierarchy JSON and return a corrected response in the same
JSON format. Return STRICT JSON only.
"""

OUTPUT_FIELDS = [
    "path", "node", "definition", "example",
    "verdict", "error_type", "reason",
    "fix_type", "new_path_isa", "placement_reason",
    "title_suggestion", "title_reason"
]

PATH_VALIDATION_RETRIES = 2


# PATH VALIDATION

def _parse_path(path_str: str) -> list[str]:
    """Split a ' > '-delimited path string into segments."""
    return [seg.strip() for seg in path_str.split(" > ") if seg.strip()]


def validate_path(path_str: str, root: dict, root_key: str,
                  fix_type: str, node_name: str = "") -> tuple[bool, str]:
    """
    Validate new_path_isa against the actual hierarchy.

    Returns (is_valid, error_message).

    Rules:
    - MERGE / PLACE_UNDER: every segment must exist in the hierarchy.
    - ADD_NEW_AND_PLACE: all segments except the last 1-2 must exist; new ones
      must be tagged [virtual] and include a synset in parentheses.
    - TYPE_MISMATCH: no validation (path is in a different ontology).
    """
    if fix_type == "TYPE_MISMATCH":
        return True, ""

    if not path_str or not path_str.strip():
        return False, "new_path_isa is empty"

    segments = _parse_path(path_str)
    if not segments:
        return False, "new_path_isa has no segments after splitting on ' > '"

    # The path must never end with the node being audited itself
    if node_name:
        last_seg = segments[-1]
        # Compare by stripping synset from last segment for a loose match,
        # then also do an exact match against the full node key
        last_bare = re.sub(r"\s*\(.*?\)", "", last_seg).strip().lower()
        node_bare = re.sub(r"\s*\(.*?\)", "", node_name).strip().lower()
        if last_bare == node_bare or last_seg == node_name:
            return False, (
                f"new_path_isa ends with the node being audited ({last_seg!r}). "
                f"The path must end at the DESTINATION (parent or merge target), "
                f"not the node itself."
            )

    # Walk the hierarchy segment by segment
    # The first segment should be the root key
    if segments[0] != root_key:
        return False, (
            f"First segment {segments[0]!r} does not match root key {root_key!r}"
        )

    node = root
    new_node_count = 0

    for i, seg in enumerate(segments[1:], start=1):
        specs = node.get("specializations", {})

        if seg in specs:
            # Exists — continue walking
            node = specs[seg]
            if new_node_count > 0:
                # A real node after a new virtual one: that's fine only if the
                # new nodes came at the very end — this means we've gone past them
                # which is a structural error.
                return False, (
                    f"Segment {seg!r} at position {i} exists in the hierarchy, "
                    f"but follows a new [virtual] node. New nodes must be at the end."
                )
        else:
            # Does not exist
            if fix_type in ("MERGE", "PLACE_UNDER"):
                return False, (
                    f"Segment {seg!r} at position {i} does not exist in the hierarchy. "
                    f"For fix_type={fix_type!r}, every node must already exist. "
                    f"Available keys at this level: {list(specs.keys())[:10]}"
                )

            # ADD_NEW_AND_PLACE: new nodes allowed only at the end
            if not seg.startswith("[virtual]"):
                return False, (
                    f"New segment {seg!r} at position {i} must be tagged [virtual] "
                    f"since it does not exist in the hierarchy."
                )
            if not re.search(r"\(.+\)", seg):
                return False, (
                    f"New virtual segment {seg!r} must include a synset in "
                    f"parentheses, e.g. [virtual] bridge (bridge.n.01)."
                )
            new_node_count += 1
            if new_node_count > 2:
                return False, (
                    f"Too many new nodes: only up to 2 new [virtual] nodes are "
                    f"allowed. Found {new_node_count} so far."
                )
            # Stop walking — we're in new territory
            # (remaining segments, if any, should also be new virtual nodes)
            for remaining in segments[i + 1:]:
                if not remaining.startswith("[virtual]"):
                    return False, (
                        f"Segment {remaining!r} follows a new [virtual] node but "
                        f"is not itself tagged [virtual]. All trailing new nodes "
                        f"must be [virtual]."
                    )
                if not re.search(r"\(.+\)", remaining):
                    return False, (
                        f"New virtual segment {remaining!r} must include a synset "
                        f"in parentheses."
                    )
                new_node_count += 1
                if new_node_count > 2:
                    return False, (
                        f"Too many new nodes: only up to 2 new [virtual] nodes allowed."
                    )
            break

    return True, ""


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
                 max_siblings: int = MAX_SIBLINGS,
                 audit_depth: Optional[int] = AUDIT_DEPTH):
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
            audit_depth:         If set, only audit nodes at exactly this depth
                                 (root = depth 1). None = audit all beyond ROOT_DISTANCE.
        """
        self.hierarchy_json    = hierarchy_json
        self.output_csv        = output_csv
        self.llm_model         = llm_model
        self.max_retries       = max_retries
        self.temperature       = temperature
        self.max_tokens        = max_tokens
        self.checkpoint_interval = checkpoint_interval
        self.max_siblings      = max_siblings
        self.audit_depth       = audit_depth

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)

        print("🔧 Noun Hierarchy Auditor initialized")
        print(f"   Hierarchy:   {self.hierarchy_json}")
        print(f"   Output CSV:  {self.output_csv}")
        print(f"   Model:       {self.llm_model}")
        if self.audit_depth is not None:
            print(f"   Audit depth: {self.audit_depth} (only nodes at exactly this depth)")

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
            max_siblings=self.max_siblings,
            node_name=node_name,
            definition=definition,
            synonyms=synonyms,
            supersenses=supersenses,
            example=example,
            hypernyms=hypernyms,
            hierarchy_json=hierarchy_json_str,
        )

    def validate_and_retry(self, parsed: dict, root: dict, root_key: str,
                           system_prompt: str, base_user_prompt: str) -> tuple[dict, bool]:
        """
        Validate new_path_isa in parsed. If invalid, retry up to
        PATH_VALIDATION_RETRIES times with a correction prompt.

        Returns (final_parsed, is_valid).
        """
        fix_type = parsed.get("fix_type", "N/A")
        new_path = parsed.get("new_path_isa", "")

        # OK verdicts and N/A fix types don't need path validation
        if parsed.get("verdict") == "OK" or fix_type == "N/A":
            return parsed, True

        is_valid, error_msg = validate_path(new_path, root, root_key, fix_type, node_name=self._current_node_name)
        if is_valid:
            return parsed, True

        print(f"    ⚠️  Path validation failed: {error_msg}")

        for attempt in range(1, PATH_VALIDATION_RETRIES + 1):
            print(f"    🔄 Path retry {attempt}/{PATH_VALIDATION_RETRIES}")
            retry_user = RETRY_PROMPT.format(
                validation_error=error_msg,
                bad_path=new_path,
            ) + "\n\nOriginal prompt for context:\n" + base_user_prompt

            raw = self.call_llm(system_prompt, retry_user)
            retried = self.parse_json_safe(raw)
            if not retried:
                print(f"    ⚠️  Could not parse retry JSON")
                continue

            fix_type  = retried.get("fix_type", "N/A")
            new_path  = retried.get("new_path_isa", "")
            is_valid, error_msg = validate_path(new_path, root, root_key, fix_type, node_name=self._current_node_name)

            if is_valid:
                print(f"    ✅ Path valid after retry {attempt}")
                return retried, True

            print(f"    ⚠️  Still invalid after retry {attempt}: {error_msg}")

        print(f"    ❌ Path remained invalid after all retries. Keeping last response, marking path_valid=False.")
        return retried if retried else parsed, False

    # Core processing

    def process_node(self, item: dict, root: dict, root_key: str,
                     flagged_paths: set, hierarchy_json_str: str) -> Optional[dict]:
        """Audit a single node. Returns a result dict, or None to skip."""
        full_path = item["path"]
        node_name = item["title"]
        node      = item["node"]
        depth     = len(full_path)

        # Depth filter: if AUDIT_DEPTH is set, only process nodes at exactly that depth
        if self.audit_depth is not None:
            if depth != self.audit_depth:
                return None

        # Fallback: skip nodes too close to root (used when AUDIT_DEPTH is None)
        if self.audit_depth is None and depth <= ROOT_DISTANCE:
            return None

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

        # Validate path and retry if needed
        self._current_node_name = node_name
        parsed, path_valid = self.validate_and_retry(
            parsed, root, root_key, SYSTEM_PROMPT, user_prompt
        )

        verdict = parsed.get("verdict", "")
        if "error_type" in parsed:
            print(f"      • Verdict:    {verdict}")
            print(f"      • Error type: {parsed.get('error_type')}")
            print(f"      • Fix type:   {parsed.get('fix_type')}")
            print(f"      • Reason:     {parsed.get('reason', '')[:80]}")
            if not path_valid:
                print(f"      ⚠️  WARNING: new_path_isa could not be validated")

        return {
            "path":             " > ".join(full_path),
            "node":             node_name,
            "definition":       definition,
            "example":          example,
            "verdict":          verdict,
            "error_type":       parsed.get("error_type"),
            "reason":           parsed.get("reason"),
            "fix_type":         parsed.get("fix_type"),
            "new_path_isa":     parsed.get("new_path_isa", ""),
            "placement_reason": parsed.get("placement_reason"),
            "title_suggestion": parsed.get("title_suggestion"),
            "title_reason":     parsed.get("title_reason"),
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
            root_key = next(iter(tree))
            root = tree[root_key]
            hierarchy_json_str = json.dumps(tree, ensure_ascii=False)

            # Traverse all nodes
            nodes = traverse_nodes(root, path=[root_key])
            print(f"   Found {len(nodes)} total nodes")

            # Filter to audit depth before sampling, so sample sizes are meaningful
            if self.audit_depth is not None:
                nodes = [n for n in nodes if len(n["path"]) == self.audit_depth]
                print(f"   Filtered to {len(nodes)} nodes at depth {self.audit_depth}")

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
                        self.process_node, n, root, root_key, flagged_paths, hierarchy_json_str
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
        print(f"Nodes skipped:       {skipped}  (depth filter or too close to root)")
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
        print("❌ Error: OPENAI_API_KEY environment variable is not set!")
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