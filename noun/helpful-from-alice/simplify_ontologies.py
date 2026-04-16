"""
Generate simplified ontology JSON files from full verb, information, noun, and optionally supersenses ontologies.

Usage:
  python simplify_ontologies.py --syn y [--supersenses y]   # append noun synsets from Path (… > lemma.n.nn);
                                        # [virtual] nodes infer synset from descendant paths (entity trees) or
                                        # sibling item paths (supersenses flat lists). Verb keys keep (.v.nn, …).
  python simplify_ontologies.py --syn n [--supersenses y]   # no noun-synset parentheticals in entity/supersenses titles;
                                        # strip verb synset suffixes from keys.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
VERB_IN = ROOT / "verb-ontology.json"
VERB_OUT = ROOT / "verb-ontology-simplified.json"
ACT_IN = ROOT / "actor-ontology.json"
ACT_OUT = ROOT / "actor-ontology-simplified.json"
NOUN_IN = ROOT / "noun-ontology.json"
NOUN_OUT = ROOT / "noun-ontology-simplified.json"
SUPERSES_IN = ROOT / "supersenses-ontology.json"
SUPERSES_OUT = ROOT / "supersenses-ontology-simplified.json"

ATOMIC = "(Atomic Tasks)"
SPECS = "(Specializations)"

# Last segment of a Path line must look like a noun WordNet synset id.
_NOUN_SYNSET = re.compile(r"^[A-Za-z0-9_'%.\-]+\.n\.\d+$")

# Trailing parenthetical whose content is only comma-separated verb lemmas (… .v.nn).
_VERB_LEMMA = re.compile(r"^[A-Za-z0-9_'.\-]+\.v\.\d+$")


def simplify_verb_node(obj: Any) -> Any:
    if isinstance(obj, list):
        return []
    if not isinstance(obj, dict):
        return obj

    out: dict[str, Any] = {}

    for key, val in obj.items():
        if key == ATOMIC:
            continue
        if key == SPECS:
            merged = simplify_verb_node(val)
            if not isinstance(merged, dict):
                continue
            for sk, sv in merged.items():
                if sk in out:
                    existing = out[sk]
                    if isinstance(existing, dict) and isinstance(sv, dict):
                        out[sk] = _deep_merge_verb(existing, sv)
                    else:
                        out[sk] = sv
                else:
                    out[sk] = sv
            continue
        out[key] = simplify_verb_node(val)

    return out


def _deep_merge_verb(a: dict, b: dict) -> dict:
    r = dict(a)
    for k, v in b.items():
        if k in r and isinstance(r[k], dict) and isinstance(v, dict):
            r[k] = _deep_merge_verb(r[k], v)
        else:
            r[k] = v
    return r


def _is_only_verb_synset_list(inner: str) -> bool:
    inner = inner.strip()
    if not inner:
        return False
    for part in inner.split(","):
        p = part.strip()
        if not p or not _VERB_LEMMA.match(p):
            return False
    return True


def strip_verb_synset_suffixes(label: str) -> str:
    """Remove trailing `(lemma.v.nn, …)` groups from the right while they match verb synset lists."""
    s = label.rstrip()
    while s.endswith(")"):
        depth = 0
        open_idx = -1
        for i in range(len(s) - 1, -1, -1):
            ch = s[i]
            if ch == ")":
                depth += 1
            elif ch == "(":
                depth -= 1
                if depth == 0:
                    open_idx = i
                    break
        if open_idx < 0:
            break
        inner = s[open_idx + 1 : len(s) - 1].strip()
        if not _is_only_verb_synset_list(inner):
            break
        s = s[:open_idx].rstrip()
    return s


def remap_verb_tree_keys(obj: Any) -> Any:
    """Re-key verb tree with stripped labels; deep-merge on duplicate keys."""
    if not isinstance(obj, dict):
        return obj
    out: dict[str, Any] = {}
    for k, v in obj.items():
        nk = strip_verb_synset_suffixes(k)
        nv = remap_verb_tree_keys(v)
        if nk in out and isinstance(out[nk], dict) and isinstance(nv, dict):
            out[nk] = _deep_merge_verb(out[nk], nv)
        else:
            out[nk] = nv
    return out


def clean_bracket_title(title: str) -> str:
    """Remove [...] segments except [virtual] / [virtual] … (case-insensitive)."""
    if not title:
        return title
    result: list[str] = []
    i = 0
    n = len(title)
    while i < n:
        if title[i] != "[":
            result.append(title[i])
            i += 1
            continue
        j = title.find("]", i)
        if j == -1:
            result.append(title[i])
            i += 1
            continue
        segment = title[i : j + 1]
        inner = segment[1:-1].strip().lower()
        if inner == "virtual" or inner.startswith("virtual "):
            result.append(segment)
        i = j + 1
    s = "".join(result).strip()
    s = re.sub(r"  +", " ", s)
    return s


def _path_string_from_description(description: str) -> str | None:
    """Return the path chain text after '- Path:' / 'Path:', or None if missing/empty."""
    if not description or not isinstance(description, str):
        return None
    path_value: str | None = None
    for raw in description.splitlines():
        line = raw.strip()
        if line.startswith("- Path:"):
            path_value = line[len("- Path:") :].strip()
            break
        if line.startswith("Path:") and path_value is None:
            path_value = line[len("Path:") :].strip()
    if not path_value:
        return None
    return path_value


def _synset_segments_from_path(path_value: str) -> list[str]:
    return [p.strip() for p in path_value.split(">") if p.strip()]


def last_noun_synset_from_path_chain(path_value: str) -> str | None:
    """Last segment of a 'a > b > lemma.n.nn' chain if it is a noun synset id (supersenses `path` field or Path line)."""
    if not path_value or not isinstance(path_value, str):
        return None
    pv = path_value.strip()
    if not pv:
        return None
    if ">" not in pv:
        return pv if _NOUN_SYNSET.match(pv) else None
    segments = _synset_segments_from_path(pv)
    if not segments:
        return None
    last = segments[-1]
    if _NOUN_SYNSET.match(last):
        return last
    return None


def extract_noun_synset_from_description(description: str) -> str | None:
    """Return the last synset token from a '- Path: a > b > lemma.n.nn' line, if present."""
    path_value = _path_string_from_description(description)
    if not path_value:
        return None
    return last_noun_synset_from_path_chain(path_value)


_VIRTUAL_TITLE_LEMMA = re.compile(r"^\[[Vv]irtual\]\s*(.+)$")


def infer_virtual_synset_from_descendants(node: dict[str, Any], cleaned_title: str) -> str | None:
    """
    [virtual] nodes often omit Path; the intended noun synset appears as a path segment
    on descendants (e.g. scientific_research.n.01 under [virtual] scientific_research).
    """
    m = _VIRTUAL_TITLE_LEMMA.match(cleaned_title.strip())
    if not m:
        return None
    lemma = m.group(1).strip().replace(" ", "_")
    if not lemma:
        return None
    lemma_pat = re.compile(rf"^{re.escape(lemma)}\.n\.\d+$")

    def walk(n: dict[str, Any]) -> str | None:
        ps = _path_string_from_description(str(n.get("description") or ""))
        if ps:
            for seg in _synset_segments_from_path(ps):
                if lemma_pat.match(seg):
                    return seg
        specs_inner = n.get("specializations") or {}
        if not isinstance(specs_inner, dict):
            return None
        for c in specs_inner.values():
            if isinstance(c, dict):
                hit = walk(c)
                if hit:
                    return hit
        return None

    specs = node.get("specializations") or {}
    if not isinstance(specs, dict):
        return None
    for c in specs.values():
        if isinstance(c, dict):
            hit = walk(c)
            if hit:
                return hit
    return None


def infer_virtual_synset_from_flat_items(items: list[Any], cleaned_title: str) -> str | None:
    """
    Supersenses-ontology items use a flat `items` list; [virtual] rows often use path placeholders.
    Scan all items' `path` chains for the first segment matching `lemma.n.nn` for the [virtual] lemma.
    """
    m = _VIRTUAL_TITLE_LEMMA.match(cleaned_title.strip())
    if not m:
        return None
    lemma = m.group(1).strip().replace(" ", "_")
    if not lemma:
        return None
    lemma_pat = re.compile(rf"^{re.escape(lemma)}\.n\.\d+$")
    for it in items:
        if not isinstance(it, dict):
            continue
        pv = it.get("path")
        if not pv or not isinstance(pv, str) or ">" not in pv:
            continue
        for seg in pv.split(">"):
            seg = seg.strip()
            if lemma_pat.match(seg):
                return seg
    return None


def _unique_key(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    k = 2
    while True:
        cand = f"{base} ({k})"
        if cand not in used:
            used.add(cand)
            return cand
        k += 1


def _title_key_for_node(
    node: dict[str, Any],
    slug: str,
    *,
    include_syn: bool,
    used: set[str],
) -> str:
    raw_title = node.get("title") or slug
    ct = clean_bracket_title(str(raw_title))
    desc = str(node.get("description") or "")
    syn = extract_noun_synset_from_description(desc)
    if include_syn and syn is None:
        syn = infer_virtual_synset_from_descendants(node, ct)
    if include_syn and syn:
        base = f"{ct} ({syn})"
    else:
        base = ct
    return _unique_key(base, used)


def simplify_entity_tree_node(node: dict[str, Any], *, include_syn: bool) -> dict[str, Any]:
    """Nested dict from specializations (activity / noun style)."""
    specs = node.get("specializations") or {}
    if not isinstance(specs, dict):
        return {}

    used: set[str] = set()
    out: dict[str, Any] = {}
    for slug, child in specs.items():
        if not isinstance(child, dict):
            continue
        key = _title_key_for_node(child, str(slug), include_syn=include_syn, used=used)
        out[key] = simplify_entity_tree_node(child, include_syn=include_syn)
    return out


def simplify_root_entity_tree(data: dict[str, Any], *, include_syn: bool) -> dict[str, Any]:
    used: set[str] = set()
    out: dict[str, Any] = {}
    for slug, node in data.items():
        if not isinstance(node, dict):
            continue
        key = _title_key_for_node(node, str(slug), include_syn=include_syn, used=used)
        out[key] = simplify_entity_tree_node(node, include_syn=include_syn)
    return out


def _title_key_for_supersenses_item(
    item: dict[str, Any],
    *,
    items: list[Any],
    include_syn: bool,
    used: set[str],
) -> str:
    raw_title = item.get("title") or ""
    ct = clean_bracket_title(str(raw_title))
    path_val = item.get("path")
    syn = last_noun_synset_from_path_chain(str(path_val) if path_val else "")
    if include_syn and syn is None:
        syn = infer_virtual_synset_from_flat_items(items, ct)
    if include_syn and syn:
        base = f"{ct} ({syn})"
    else:
        base = ct
    return _unique_key(base, used)


def simplify_supersenses_ontology(data: dict[str, Any], *, include_syn: bool) -> dict[str, Any]:
    """
    categories[] -> subcategories[] -> items[] : nested dict
    { category_label: { subcategory_id: { item_title: {} } } }.
    """
    categories = data.get("categories")
    if not isinstance(categories, list):
        return {}
    out: dict[str, Any] = {}
    used_cat: set[str] = set()
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        cat_base = str(cat.get("label") or cat.get("id") or "category")
        cat_key = _unique_key(cat_base, used_cat)
        sub_out: dict[str, Any] = {}
        used_sub: set[str] = set()
        for sub in cat.get("subcategories") or []:
            if not isinstance(sub, dict):
                continue
            sub_base = str(sub.get("id") or sub.get("description") or "subcategory")
            sub_key = _unique_key(sub_base, used_sub)
            raw_items = sub.get("items")
            items_list: list[Any] = raw_items if isinstance(raw_items, list) else []
            used_item: set[str] = set()
            items_out: dict[str, Any] = {}
            for it in items_list:
                if not isinstance(it, dict):
                    continue
                k = _title_key_for_supersenses_item(
                    it, items=items_list, include_syn=include_syn, used=used_item
                )
                items_out[k] = {}
            sub_out[sub_key] = items_out
        out[cat_key] = sub_out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simplified ontology JSON files.")
    parser.add_argument(
        "--syn",
        choices=["y", "n"],
        default="n",
        help="y: append noun synsets to information & noun titles (and supersenses when --supersenses y); "
        "keep verb synsets in keys. n: strip those from entity/supersenses titles; strip verb synset suffixes.",
    )
    parser.add_argument(
        "--supersenses",
        choices=["y", "n"],
        default="n",
        help="y: also read supersenses-ontology.json and write supersenses-ontology-simplified.json.",
    )
    args = parser.parse_args()
    include_syn = args.syn == "y"
    include_supersenses = args.supersenses == "y"

    print("Loading verb ontology...")
    with VERB_IN.open(encoding="utf-8") as f:
        verb_data = json.load(f)
    print("Simplifying verb ontology...")
    verb_simple: Any = simplify_verb_node(verb_data)
    if not include_syn:
        print("Stripping verb synset suffixes from keys (--syn n)...")
        verb_simple = remap_verb_tree_keys(verb_simple)
    print(f"Writing {VERB_OUT.name}...")
    with VERB_OUT.open("w", encoding="utf-8") as f:
        json.dump(verb_simple, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Loading activity ontology...")
    with ACT_IN.open(encoding="utf-8") as f:
        act_data = json.load(f)
    if not isinstance(act_data, dict):
        raise SystemExit("activity-ontology root must be an object")
    print("Simplifying activity ontology...")
    act_simple = simplify_root_entity_tree(act_data, include_syn=include_syn)
    print(f"Writing {ACT_OUT.name}...")
    with ACT_OUT.open("w", encoding="utf-8") as f:
        json.dump(act_simple, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("Loading noun ontology...")
    with NOUN_IN.open(encoding="utf-8") as f:
        noun_data = json.load(f)
    if not isinstance(noun_data, dict):
        raise SystemExit("noun-ontology root must be an object")
    print("Simplifying noun ontology...")
    noun_simple = simplify_root_entity_tree(noun_data, include_syn=include_syn)
    print(f"Writing {NOUN_OUT.name}...")
    with NOUN_OUT.open("w", encoding="utf-8") as f:
        json.dump(noun_simple, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if include_supersenses:
        print("Loading supersenses ontology...")
        with SUPERSES_IN.open(encoding="utf-8") as f:
            super_data = json.load(f)
        if not isinstance(super_data, dict):
            raise SystemExit("supersenses-ontology root must be an object")
        print("Simplifying supersenses ontology...")
        super_simple = simplify_supersenses_ontology(super_data, include_syn=include_syn)
        print(f"Writing {SUPERSES_OUT.name}...")
        with SUPERSES_OUT.open("w", encoding="utf-8") as f:
            json.dump(super_simple, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print("Done.")


if __name__ == "__main__":
    main()
