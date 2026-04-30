#!/usr/bin/env python3

"""
Condenses the raw noun hierarchy JSON for use as LLM context.

What it does:
- Strips emojis and metadata tags ([added], [edited], [moved]) from titles
- Collapses [virtual] intermediate nodes that have few children (≤ VIRTUAL_COLLAPSE_THRESHOLD)
  by promoting their children directly to the "grandparent" level

"""

import json
import re

INPUT_FILE  = "nodes-data.json"
OUTPUT_FILE = "nodes-data-condensed.json"

VIRTUAL_COLLAPSE_THRESHOLD = 3


# Helpers

def remove_emojis(text: str) -> str:
    if not text:
        return text
    return re.sub(r'[^\x00-\x7F]+', '', text)


def clean_title(title: str) -> str:
    """Strip [added], [edited], [moved] tags from a title."""
    if not title:
        return title
    return re.sub(r"\[(added|edited|moved)\]\s*", "", title).strip()


def clean_description(desc: str) -> str:
    """Strip emojis."""
    if not desc:
        return ""
    return remove_emojis(desc).strip()


def is_virtual(node: dict) -> bool:
    return "[virtual]" in node.get("title", "").lower()


def safe_insert(d: dict, key: str, value) -> None:
    """Insert key into d, appending a counter suffix to avoid overwrites."""
    i = 1
    new_key = key
    while new_key in d:
        new_key = f"{key}_{i}"
        i += 1
    d[new_key] = value

def condense_node(node: dict, is_root: bool = False):
    """
    Recursively condense a node.

    Returns:
        dict  — a cleaned node to keep as-is
        list  — a list of (key, child) pairs to be promoted to the grandparent
                (only returned when this node is a collapsible [virtual] node)
    """
    new_children: dict = {}

    for key, child in node.get("specializations", {}).items():
        processed = condense_node(child, is_root=False)

        if isinstance(processed, dict):
            safe_insert(new_children, clean_title(key), processed)
        elif isinstance(processed, list):
            # Flatten promoted grandchildren directly into this level
            for promoted_key, promoted_val in processed:
                safe_insert(new_children, promoted_key, promoted_val)

    # Collapse this node if it is a [virtual] intermediate with few children
    if not is_root and is_virtual(node) and len(new_children) <= VIRTUAL_COLLAPSE_THRESHOLD:
        return list(new_children.items())

    # Otherwise, build and return the cleaned node
    new_node: dict = {}

    title = clean_title(node.get("title", ""))
    if title:
        new_node["title"] = title

    desc = clean_description(node.get("description", ""))
    if desc:
        new_node["description"] = desc

    if new_children:
        new_node["specializations"] = new_children

    return new_node


# Main

def main():
    print(f"📂 Loading: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    root_key      = list(data.keys())[0]
    condensed     = condense_node(data[root_key], is_root=True)
    output        = {clean_title(root_key): condensed}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Condensed hierarchy saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
