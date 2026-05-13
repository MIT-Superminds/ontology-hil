"""
Rearranges an "original" ontology JSON to match the structure of a "target" condensed ontology JSON,
to add synsets and preserve 'description' from the original where available.

Matching strategy:
- Target node keys carry a synset ID in parentheses, e.g. "hole (hole.n.02)".
- Original node descriptions contain a Path: line whose last segment is the synset ID,
  e.g. "... > hole.n.02".
- Lookup is keyed by synset ID (preferred) with a bare-name fallback for virtual nodes
  (those with no Path / empty Path).

"""

import json
import re


# =========================
# CONFIGURATION (to edit)
# =========================
OG_INPUT_PATH = 'physical-condensed.json'
TS_INPUT_PATH = 'to-sample.json'
OUTPUT_PATH   = 'physical-rearranged-by-synset.json'

# Root node name (set to None to auto-detect first key)
ROOT_KEY = None

# =========================


def extract_synset_from_key(key: str) -> tuple[str, str]:
    """
    Split a target-ontology key into (bare_name, synset_id).

    Examples:
    'hole (hole.n.02)'                          -> ('hole', 'hole.n.02')
    '[virtual] physical_entity (physical_entity.n.01)' -> ('[virtual] physical_entity', 'physical_entity.n.01')
    'infrastructure'                            -> ('infrastructure', None)
    """
    m = re.match(r'^(.*?)\s*\(([^)]+)\)\s*$', key)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return key.strip(), None


def extract_synset_from_description(description: str) -> str | None:
    """
    Extract the synset ID from the Path: line in an original node's description.

    Looks for a line like:
        - Path: entity.n.01 > physical_entity.n.01 > ... > hole.n.02
    and returns the last segment ('hole.n.02').

    Returns None if the Path is absent or empty (virtual node).
    """
    if not description:
        return None
    # Match "- Path: <anything>" up to a newline
    m = re.search(r'-\s*Path:\s*([^\n]*)', description)
    if not m:
        return None
    path_str = m.group(1).strip()
    if not path_str:
        return None
    # Last segment after " > "
    last = path_str.split('>')[-1].strip()
    return last if last else None


def collect_og_nodes(d: dict) -> tuple[dict, dict]:
    """
    Flatten a nested original ontology into two lookup dicts:
      - by_synset  : { synset_id -> node_data }   (primary)
      - by_name    : { bare_name -> node_data }    (fallback for virtual nodes)

    Ignores structural keys: 'title', 'description', 'specializations'.
    """
    by_synset: dict = {}
    by_name: dict   = {}

    if not isinstance(d, dict):
        return by_synset, by_name

    for k, v in d.items():
        if k in ('title', 'description', 'specializations'):
            continue

        # Register this node
        description = v.get('description', '') if isinstance(v, dict) else ''
        synset = extract_synset_from_description(description)

        if synset:
            by_synset[synset] = v
        else:
            # Virtual / no-synset node — register by bare name only
            by_name[k] = v

        # Recurse into specializations
        if isinstance(v, dict) and 'specializations' in v:
            child_synset, child_name = collect_og_nodes(v['specializations'])
            by_synset.update(child_synset)
            by_name.update(child_name)

    return by_synset, by_name


def build_output(ts_node: dict,
                 og_by_synset: dict,
                 og_by_name: dict,
                 missing_log: list) -> dict:
    """
    Rebuild hierarchy using:
    - structure from target ontology (ts_node)
    - metadata from original ontology (og_by_synset / og_by_name)

    Matching order:
      1. synset ID extracted from target key  →  og_by_synset
      2. bare name                            →  og_by_name   (virtual-node fallback)
      3. neither found                        →  new node with defaults
    """
    result = {}

    for ts_key, ts_children in ts_node.items():
        bare_name, synset = extract_synset_from_key(ts_key)

        # Try synset lookup first, then bare-name fallback
        og_data = None
        if synset:
            og_data = og_by_synset.get(synset)
        if og_data is None:
            og_data = og_by_name.get(bare_name)

        new_node: dict = {}

        if og_data:
            # Existing node — preserve metadata
            new_node['title']       = og_data.get('title', bare_name) or bare_name
            new_node['description'] = og_data.get('description', '')
        else:
            # New node — create with defaults
            missing_log.append(ts_key)
            new_node['title']       = bare_name
            new_node['description'] = ''

        if isinstance(ts_children, dict) and ts_children:
            new_node['specializations'] = build_output(
                ts_children, og_by_synset, og_by_name, missing_log
            )

        result[ts_key] = new_node

    return result


def count_keys(d: dict) -> int:
    """Recursively count all keys in a nested dictionary."""
    if not isinstance(d, dict):
        return 0
    return len(d) + sum(count_keys(v) for v in d.values())


def get_root(data: dict) -> str:
    """
    Determine the root key of an ontology.

    Uses ROOT_KEY if explicitly set, otherwise the first key in the dict.
    """
    if ROOT_KEY:
        return ROOT_KEY
    return next(iter(data.keys()))


def main():
    with open(OG_INPUT_PATH, encoding='utf-8') as f:
        og = json.load(f)

    with open(TS_INPUT_PATH, encoding='utf-8') as f:
        ts = json.load(f)

    og_root = get_root(og)

    # Build lookups from original ontology
    og_by_synset, og_by_name = collect_og_nodes(og[og_root].get('specializations', {}))

    # Also register the root node itself
    root_data = og[og_root]
    root_synset = extract_synset_from_description(root_data.get('description', ''))
    if root_synset:
        og_by_synset[root_synset] = root_data
    else:
        og_by_name[og_root] = root_data

    # Build output
    missing_log: list = []
    output = build_output(ts, og_by_synset, og_by_name, missing_log)

    # Reporting
    print(f"Root node:                   {og_root}")
    print(f"Nodes in target structure:   {count_keys(ts)}")
    print(f"Nodes in OG synset lookup:   {len(og_by_synset)}")
    print(f"Nodes in OG name lookup:     {len(og_by_name)}")
    print(f"Nodes with no OG match:      {len(missing_log)}")

    if missing_log:
        print("\nMissing nodes:")
        for m in missing_log:
            print(f"  - {m}")

    # Write output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nOutput written to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()