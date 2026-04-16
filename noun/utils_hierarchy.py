# utils_hierarchy.py

import json
import re


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_description(desc: str):
    """
    Extract structured fields from a node description string.
    Returns (definition, synonyms, supersenses, example).
    """
    if not desc:
        return "", "", "", ""

    def extract(label):
        pattern = rf"{label}:\s*(.*?)(?=\n[A-Z][a-zA-Z]+:|$)"
        match = re.search(pattern, desc, re.DOTALL)
        return match.group(1).strip() if match else ""

    return (
        extract("Definition"),
        extract("Synonyms"),
        extract("Supersenses"),
        extract("Example"),
    )


def traverse_nodes(tree: dict, path: list = None) -> list:
    """
    Recursively collect all nodes in the hierarchy.
    Returns a list of dicts with keys: path, node, title.
    """
    if path is None:
        path = ["Entity"]

    nodes = []
    for title, child in tree.get("specializations", {}).items():
        current_path = path + [title]
        nodes.append({"path": current_path, "node": child, "title": title})
        nodes.extend(traverse_nodes(child, current_path))

    return nodes


def find_parent(root: dict, target_path: list) -> dict | None:
    """Walk the hierarchy to return the direct parent node of target_path."""
    node = root
    for seg in target_path[1:-1]:
        node = node.get("specializations", {}).get(seg)
        if node is None:
            return None
    return node


def get_siblings(parent_node: dict | None, current_title: str) -> list:
    """Return sibling titles under the same parent, excluding the current node."""
    if not parent_node:
        return []
    pool = parent_node.get("specializations", {})
    return [k for k in pool.keys() if k != current_title]

def safe_list(x: list, max_items: int) -> list:
    """Return up to max_items elements from x."""
    return x[:max_items] if x else []
