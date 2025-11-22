#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
import re
import time
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Any, Optional
from nltk.corpus import wordnet as wn

CSV_PATH = "missing verbs/finalmissingsynsets.csv"                 # your 'missingtasks.csv'
HIERARCHY_PATH = "missing verbs/1121_finalhierarchy.json"  # verbs-only hierarchy json
OUT_CSV_PATH = "missing verbs/1122nopathplacements.csv"             # final recommendations (authoritative)
LIMIT_SYNSETS: Optional[int] = None                      # set None to run all synsets
START_OFFSET: int = 0                                   # skip first N synsets

# Prompt size safety (truncate hierarchy path-lines if huge)
MAX_PROMPT_HIERARCHY_LINES = 6000

# LLM defaults
DEFAULT_OPENAI_MODEL = "gpt-4o"
MAX_RETRIES = 3
TEMPERATURE = 0.1
MAX_TOKENS = 500

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
    raise RuntimeError("OpenAI SDK not found. Install with `pip install openai python-dotenv`.")

# Prompts
SYSTEM_PROMPT = """
You are an expert verb taxonomy editor. Your task is to place ONE verb synset into an existing verb hierarchy.

Hierarchy facts:
- The hierarchy is provided as exact path lines formatted like: A > B > C
- Titles are case- and punctuation-sensitive; you must NOT invent new intermediate titles.
- Many titles include existing synsets in parentheses, e.g. Plan (Plan.v.02, Plan.v.03, Mastermind.v.01)
- Valid placements (choose ONLY ONE):
  1) ADD_CHILD_NEW_VERB: All parent segments exist, but the final segment is NEW. You must provide 'new_node_title' (e.g., Lay (Lay.v.03)), which will be created under the parent node.
  2) APPEND_TO_EXISTING: The final node in 'path' already exists. You will APPEND the given synset to that node’s parentheses (comma-separated) if absent.
  3) INCORRECT_SYNSET: If the synset is obviously wrong for the task contexts, flag this instead of placing it.

Context you will receive for the synset:
- WordNet hypernym ‘Definition’, and ‘Synonyms’ strings
- Representative O*NET task texts using this synset
- The entire hierarchy (compressed to path lines)

Return format:
- Respond with STRICT JSON ONLY (no markdown, no prose).
- Use this schema exactly:
{{
  "choice": "ADD_CHILD_NEW_VERB",
  "path": "Exact path using existing node titles separated by ' > '",
  "new_node_title": "Label (lemma.v.xx)"
}}
or
{{
  "choice": "APPEND_TO_EXISTING",
  "path": "Exact path using existing node titles separated by ' > '"
}}
or
{{
  "choice": "INCORRECT_SYNSET",
  "explanation": "Your explanation for why the synset is obviously wrong for the task contexts'"
}}


Guidance:
- Prefer the DEEPEST correct node on an appropriate branch over a higher-level bucket.
- Prefer ADD_CHILD_NEW_VERB if there is no appropriate existing title to append to. Prefer APPEND_TO_EXISTING if the synset is a synonym of the existing node. 
- Use INCORRECT_SYNSET when the synset’s core meaning does NOT match the action described in the tasks / verb–object pairs, even if there is a superficial lemma overlap.
- Use the evidence to locate the best-fitting part of the hierarchy.
- Titles must match exactly as provided; do not alter spelling, punctuation, or case.
- If you receive validator feedback about an invalid path, correct it and return a SINGLE valid JSON object.
"""

USER_PROMPT = """
You must place ONE synset.

SYNSET: {synset}
DEFINITION: {definition}
SYNONYMS: {synonyms}
REPRESENTATIVE O*NET TASK TEXTS: {onet_tasks}
VERB-OBJECT EXAMPLES: {verb_object_examples}

{correction_note}

HIERARCHY (exact path lines; do not invent titles): {hierarchy_lines}

EXAMPLE PLACEMENTS:
Synset: adhere.v.01
Definition: “be compatible or in accordance with”
Synonyms: “adhere”
Representative O*NET Task Texts: [“Adhere to local, state, and federal laws, regulations, and statutes.”, “Adhere to legal policies and procedures related to handling digital media.”, “Adhere to safety practices and procedures, such as checking equipment regularly and erecting barriers around work areas.”, “Adhere to safety practices and procedures, such as checking equipment regularly and erecting barriers around work areas.”, “Adhere to all applicable regulations, policies, and procedures for health, safety, and environmental compliance.”, “Adhere to schedules to keep events running on time.”]
Verb-Object Examples: [“Adhere laws”, “Adhere policies”, “Adhere practices”, “Adhere procedures”, “Adhere regulations”, “Adhere schedules”]
Output: {{
    "choice": "APPEND_TO_EXISTING", 
    "path": "Act > [Act on what?] > Act with other activities and actors (“Interact”) > Transfer between actors > [Transfer what?] > Provide service > Follow (Follow.v.01) > Comply (Comply.v.01, Obey.v.01)", 
}}

Synset: anesthetize.v.01
Definition: “administer an anesthetic drug to”
Synonyms: “anesthetize, anaesthetize, anesthetise, anaesthetise, put under, put out”
Representative O*NET Task Texts: [“Anesthetize and inoculate animals, according to instructions.”]
Verb-Object Examples: [“Anesthetize animals”]
Output: {{
    "choice": "ADD_CHILD_NEW_VERB",
    "path": "Act > [Act on what?] > Act with other activities and actors (“Interact”) > Transfer between actors > [Transfer what?] > Transfer service > Provide service > Assist (Support.v.02, Support.v.01, Help.v.01) > Treat (Treat.v.03) > Administer (treat) (Administer.v.04)",
    "new_node_title": "Anesthetize (anesthetize.v.01)"
}}
"""

# Helper: build the user prompt string
def format_user_prompt(
    synset: str,
    definition: str,
    synonyms: str,
    onet_tasks: List[str],
    verb_object_examples: List[str],
    hierarchy_lines: List[str],
    correction_note: Optional[str] = None
) -> str:
    def fmt_block(lines: List[str], bullet: bool = True, fallback: str = "- (none provided)"):
        if not lines:
            return fallback
        if bullet:
            return "\n".join([f"- {x}" for x in lines])
        return "\n".join(lines)

    return USER_PROMPT.format(
        synset=synset,
        definition=definition,
        synonyms=synonyms,
        onet_tasks=fmt_block(onet_tasks[:50]),  # keep prompt modest
        verb_object_examples=fmt_block(verb_object_examples),
        correction_note=(correction_note or "").strip(),
        hierarchy_lines=fmt_block(hierarchy_lines, bullet=False)
    )


# Traversing hierarchy helpers
def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_dict_node(node: Any) -> bool:
    return isinstance(node, dict)

def iter_paths(d: Dict[str, Any], prefix: Optional[List[str]] = None):
    prefix = prefix or []
    for title, child in d.items():
        cur = prefix + [title]
        yield cur
        if is_dict_node(child):
            yield from iter_paths(child, cur)

def compress_hierarchy_to_lines(h: Dict[str, Any]) -> List[str]:
    return [" > ".join(path) for path in iter_paths(h)]

def get_node(h: Dict[str, Any], path_segments: List[str]) -> Optional[Dict[str, Any]]:
    node = h
    for seg in path_segments:
        if not is_dict_node(node) or seg not in node:
            return None
        node = node[seg]
    return node if is_dict_node(node) else None

def get_parent_and_missing(h: Dict[str, Any], path_segments: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    node = h
    for seg in path_segments:
        if not is_dict_node(node):
            return None, seg
        if seg not in node:
            return node, seg
        node = node[seg]
    return node, None

def normalize_synset(s: str) -> str:
    s = s.strip()
    parts = s.split(".")
    if len(parts) >= 3:
        parts[0] = parts[0].lower()
        return ".".join(parts)
    return s.lower()

def unique_preserve_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


# Main
class SynsetPlacer:
    def __init__(self):
        # Client
        self.llm_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        self.max_retries = MAX_RETRIES
        self.client = OpenAI()

        # Load inputs
        self.rows = read_csv_rows(CSV_PATH)
        self.hier = load_json(HIERARCHY_PATH)
        self.hier_lines = compress_hierarchy_to_lines(self.hier)
        if len(self.hier_lines) > MAX_PROMPT_HIERARCHY_LINES:
            self.hier_lines = self.hier_lines[:MAX_PROMPT_HIERARCHY_LINES]

        # Index synset -> evidence
        self.synset_to_tasks: Dict[str, List[str]] = defaultdict(list)
        self.synset_to_verb_objects: Dict[str, List[str]] = defaultdict(list)
        self._index_csv()


        # Existing placements (for resume)
        self.existing_placements = self._load_existing_placements(OUT_CSV_PATH)

    def _index_csv(self):
        
        required = {
            "Task ID",
            "Task",
            "Verb",
            "Object",
            "Normalized",
            "Synset" 
        }

        if not self.rows:
            raise ValueError("CSV appears empty.")
        missing = required - set(self.rows[0].keys())
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for r in self.rows:
            syn = normalize_synset(r["Synset"])
            if not syn:
                continue

            t = (r.get("Task") or "").strip()

            verb = (r.get("Verb") or "").strip()
            obj = (r.get("Object") or "").strip()

            if t:
                self.synset_to_tasks[syn].append(t)

            if verb or obj:
                # e.g. "Adhere policies", "Review documentation", "File" (if object empty)
                pair = " ".join(x for x in [verb, obj] if x)
                self.synset_to_verb_objects[syn].append(pair)

        for syn in list(self.synset_to_tasks.keys()):
            self.synset_to_tasks[syn] = unique_preserve_order(self.synset_to_tasks[syn])
            self.synset_to_verb_objects[syn] = unique_preserve_order(self.synset_to_verb_objects[syn])

    def _load_existing_placements(self, path: str) -> Dict[str, Dict[str, str]]:
        """
        Load existing placements.csv (if present) and map synset -> row.
        Used only to know which synsets have been processed already.
        """
        if not os.path.exists(path):
            print(f"ℹ️ No existing placements file at {path}; starting fresh.")
            return {}

        print(f"🔁 Loading existing placements from {path}...")
        rows = read_csv_rows(path)
        by_synset: Dict[str, Dict[str, str]] = {}
        for r in rows:
            syn = (r.get("synset") or "").strip()
            if syn:
                by_synset[syn] = r
        print(f"   Loaded {len(by_synset)} existing synset placements.")
        return by_synset

    # LLM call (system/user prompts only)
    def call_llm(self, system_prompt: str, user_prompt: str, attempt: int = 1) -> str:
        try:
            print(f"    🤖 LLM API call (attempt {attempt})")
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                timeout=30
            )
            result = resp.choices[0].message.content.strip()
            print(f"    ✅ LLM response: {result[:120]}{'...' if len(result) > 120 else ''}")
            return result
        except Exception as e:
            print(f"    ❌ LLM API error (attempt {attempt}): {e}")
            if attempt < self.max_retries:
                delay = attempt * 2
                print(f"    ⏱️ Retrying in {delay} seconds...")
                time.sleep(delay)
                return self.call_llm(system_prompt, user_prompt, attempt + 1)
            else:
                print(f"    💥 Max retries ({self.max_retries}) reached")
                return f"API_ERROR: {str(e)}"
        
    def parse_llm_json(self, txt: str) -> Optional[Dict[str, Any]]:
        try:
            s = txt.strip()
            if s.startswith("```"):
                s = re.sub(r"^```(?:json)?\s*", "", s)
                s = re.sub(r"\s*```$", "", s)

            obj = json.loads(s)
            if not isinstance(obj, dict):
                return None

            choice = obj.get("choice")
            if not isinstance(choice, str):
                return None

            # For normal placements, we need a path
            if choice in {"APPEND_TO_EXISTING", "ADD_CHILD_NEW_VERB"}:
                if "path" not in obj or not isinstance(obj.get("path"), str):
                    return None

            # For INCORRECT_SYNSET we don't require path, but we do need an explanation
            if choice == "INCORRECT_SYNSET":
                if "explanation" not in obj or not isinstance(obj.get("explanation"), str):
                    return None

            return obj
        except Exception:
            return None
    
    def validate_choice(self, choice_obj: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        choice = choice_obj.get("choice", "")
        path = choice_obj.get("path", "")
        new_node_title = choice_obj.get("new_node_title")

        # INCORRECT_SYNSET stays as-is
        if choice == "INCORRECT_SYNSET":
            explanation = (choice_obj.get("explanation") or "").strip()
            if not explanation:
                return False, "INCORRECT_SYNSET requires non-empty 'explanation'.", {}
            return True, "OK: marked as INCORRECT_SYNSET.", {
                "action": "INCORRECT_SYNSET",
                "path": "",
                "new_node_title": None,
                "explanation": explanation,
            }

        if not isinstance(path, str) or " > " not in path:
            return False, "Path must be ' > '-separated exact titles.", {}

        segments = [seg.strip() for seg in path.split(" > ") if seg.strip()]
        if not segments:
            return False, "Empty path.", {}

        # handle ADD_CHILD_NEW_VERB *before* the generic 'target_node exists' logic
        if choice == "ADD_CHILD_NEW_VERB":
            parent_node = get_node(self.hier, segments)
            if parent_node is None:
                return False, "ADD_CHILD_NEW_VERB requires 'path' to be an existing parent node.", {}

            if not isinstance(new_node_title, str) or not new_node_title.strip():
                return False, "ADD_CHILD_NEW_VERB requires 'new_node_title'.", {}

            if not re.search(r"\([^)]+\)$", new_node_title.strip()):
                return False, "new_node_title should include '(lemma.v.xx)'.", {}

            # Prevent collision if a child with that exact title already exists
            if isinstance(parent_node, dict) and new_node_title.strip() in parent_node:
                return False, "Child with this new_node_title already exists under the parent.", {}

            return True, "OK: add new child under parent.", {
                "action": "ADD_CHILD_NEW_VERB",
                "path": path,  # parent path as given
                "new_node_title": new_node_title.strip(),
            }

        # For APPEND_TO_EXISTING: now do the existing checks
        target_node = get_node(self.hier, segments)
        if target_node is not None:
            return True, "OK: existing node; append synset.", {
                "action": "APPEND_TO_EXISTING",
                "path": path,
                "new_node_title": None,
            }

        parent, missing_seg = get_parent_and_missing(self.hier, segments)
        if parent is None:
            return False, f"Missing '{missing_seg}' with no valid parent chain.", {}

        return False, f"Path misses '{missing_seg}', and choice was not ADD_CHILD_NEW_VERB.", {}
        
    def _get_wordnet_metadata(self, syn: str) -> Tuple[str, str]:
        """
        Look up WordNet definition and synonyms for a synset string like 'run.v.01'.
        Returns (definition, synonyms_string).
        """
        try:
            wn_syn = wn.synset(syn)  # uses nltk.corpus.wordnet as wn
        except Exception:
            # If synset is missing or malformed, just return empty strings
            return "", ""

        definition = wn_syn.definition()
        # Unique synonyms (lemmas), human-readable (replace '_' with ' ')
        synonyms = sorted({lemma.name().replace("_", " ") for lemma in wn_syn.lemmas()})
        synonyms_str = ", ".join(synonyms) if synonyms else ""

        return definition, synonyms_str

    def run(self): 
        all_synsets = sorted(self.synset_to_tasks.keys()) # All from missingtasks.csv
        already_done = set(self.existing_placements.keys()) # Already in placements.csv
        synsets = [s for s in all_synsets if s not in already_done] # Remaining to do

        if START_OFFSET:
            synsets = synsets[START_OFFSET:]
        if LIMIT_SYNSETS:
            synsets = synsets[:LIMIT_SYNSETS]

        print(f"📦 Synsets to process this run: {len(synsets)} (skipping {len(already_done)} already in placements.csv)")

        placements = OrderedDict()
        for i, syn in enumerate(synsets, 1):
            print(f"\n=== [{i}/{len(synsets)}] {syn} ===")
            definition, synonyms = self._get_wordnet_metadata(syn)

            # First attempt
            system_prompt = SYSTEM_PROMPT
            user_prompt = format_user_prompt(
                synset=syn,
                definition=definition,
                synonyms=synonyms,
                onet_tasks=self.synset_to_tasks.get(syn, [])[:20],
                verb_object_examples=self.synset_to_verb_objects.get(syn, [])[:20],
                hierarchy_lines=self.hier_lines
            )
            response = self.call_llm(system_prompt, user_prompt, attempt=1)

            attempt = 1
            while attempt <= MAX_RETRIES:
                obj = self.parse_llm_json(response)
                if not obj:
                    reason = "LLM did not return valid JSON."
                    print(f"    ⚠️  {reason}")
                    if attempt >= MAX_RETRIES:
                        placements[syn] = {"status": "invalid", "attempts": attempt, "error": reason}
                        break
                    correction_note = (
                        "VALIDATOR FEEDBACK: Your previous response was not valid JSON per the required schema. "
                        "Return ONE corrected JSON object only."
                    )
                    user_prompt = format_user_prompt(
                        synset=syn,
                        definition=definition,
                        synonyms=synonyms,
                        onet_tasks=self.synset_to_tasks.get(syn, [])[:20],
                        verb_object_examples=self.synset_to_verb_objects.get(syn, [])[:20],
                        hierarchy_lines=self.hier_lines,
                        correction_note=correction_note
                    )
                    response = self.call_llm(system_prompt, user_prompt, attempt=attempt+1)
                    attempt += 1
                    continue

                ok, reason, normalized = self.validate_choice(obj)
                if ok:
                    print(f"    ✅ Valid: {reason}")
                    placements[syn] = {
                        "status": "ok",
                        "attempts": attempt,
                        "decision": normalized,
                        "evidence": {
                            "sample_tasks": self.synset_to_tasks.get(syn, [])[:20],
                            "verb_object_examples": self.synset_to_verb_objects.get(syn, [])[:20],
                            "definition": definition,
                            "synonyms": synonyms,
                        },
                    }
                    break
                else:
                    print(f"    ❌ Invalid: {reason}")
                    if attempt >= MAX_RETRIES:
                        placements[syn] = {"status": "invalid", "attempts": attempt, "error": reason, "last_obj": obj}
                        break

                    correction_note = (
                        "VALIDATOR FEEDBACK: " + reason + "\n"
                        "Please return ONE corrected JSON object using ONLY existing titles in 'path'. "
                        "If the final node exists, use APPEND_TO_EXISTING. "
                        "If only the parent exists, use ADD_CHILD_NEW_VERB and set 'new_node_title' accordingly."
                    )
                    user_prompt = format_user_prompt(
                        synset=syn,
                        definition=definition,
                        synonyms=synonyms,
                        onet_tasks=self.synset_to_tasks.get(syn, [])[:20],
                        verb_object_examples=self.synset_to_verb_objects.get(syn, [])[:20],
                        hierarchy_lines=self.hier_lines,
                        correction_note=correction_note
                    )
                    response = self.call_llm(system_prompt, user_prompt, attempt=attempt+1)
                    attempt += 1

        # Write ONLY placements.csv
        flat_rows = list(self.existing_placements.values()) # start with existing rows
        for syn, rec in placements.items():
            decision = rec.get("decision", {}) if rec.get("status") == "ok" else {}

            action = decision.get("action", "") if rec.get("status") == "ok" else ""
            base_path = decision.get("path", "") if rec.get("status") == "ok" else ""
            new_node_title = decision.get("new_node_title", "") if rec.get("status") == "ok" else ""

            # for ADD_CHILD_NEW_VERB, append the new node title to the path
            full_path = base_path
            if action == "ADD_CHILD_NEW_VERB" and base_path and new_node_title:
                full_path = f"{base_path} > {new_node_title}"

            row = {
                "synset": syn,
                "status": rec.get("status"),
                "attempts": rec.get("attempts"),
                "action": decision.get("action", "") if rec.get("status") == "ok" else "",
                "path_or_parent": full_path,
                "new_node_title": decision.get("new_node_title", "") if rec.get("status") == "ok" else "",
                "error": rec.get("error", ""),
                "incorrect_explanation": decision.get("explanation", "") if decision.get("action") == "INCORRECT_SYNSET" else "",
                "sample_onet_tasks": "|".join(rec.get("evidence", {}).get("sample_tasks", [])) if rec.get("status") == "ok" else "",
                "verb_object_examples": "|".join(rec.get("evidence", {}).get("verb_object_examples", [])) if rec.get("status") == "ok" else "",
                "definition": rec.get("evidence", {}).get("definition", "") if rec.get("status") == "ok" else "",
                "synonyms": rec.get("evidence", {}).get("synonyms", "") if rec.get("status") == "ok" else "",
            }
            flat_rows.append(row)

        csv_out = OUT_CSV_PATH
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            import csv as _csv
            writer = _csv.DictWriter(
                f,
                fieldnames=[
                    "synset", "status", "attempts", "action", "path_or_parent",
                    "new_node_title", "error", "incorrect_explanation", "sample_onet_tasks", "verb_object_examples", "definition", "synonyms"
                ]
            )
            writer.writeheader()
            writer.writerows(flat_rows)
        print(f"📄 Wrote placements to: {csv_out}")


if __name__ == "__main__":
    SynsetPlacer().run()