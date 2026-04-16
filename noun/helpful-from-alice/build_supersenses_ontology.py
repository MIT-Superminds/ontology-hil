"""
Build supersenses-ontology.json from noun-ontology.json + supersenses-highlevel.json,
and Supersenses_Ontology_Report.html (top-level mismatches, parent supersense diffs).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")

DIR = Path(__file__).resolve().parent
NOUN_ONTO = DIR / "noun-ontology.json"
HIGHLEVEL = DIR / "supersenses-highlevel.json"
OUT_ONTO = DIR / "supersenses-ontology.json"
# Alternate filename matching the project naming typo:
OUT_ONTO_ALT = DIR / "superesenses-ontology.json"
OUT_REPORT = DIR / "Supersenses_Ontology_Report.html"

SUPERLINE = re.compile(r"-\s*Supersenses:\s*([^\n]*)", re.IGNORECASE)
NOUN_SS = re.compile(r"\bnoun\.([A-Za-z0-9_]+)\b")
PATHLINE = re.compile(r"-\s*Path:\s*([^\n]*)", re.IGNORECASE)
VIRTLINE = re.compile(r"-\s*Virtual:\s*(True|False)", re.IGNORECASE)

# JSON object key under Entity -> quadrant id used in supersenses-highlevel
ENTITY_BRANCH_TO_QUADRANT: dict[str, str] = {
    "[edited] Information": "information",
    "[edited] Actor": "actor",
    "process": "activity",
    "[virtual] physical_entity": "physical_entity",
    "[added] Unclassified": "unclassified",
}

QUADRANT_LABEL: dict[str, str] = {
    "information": "Information",
    "actor": "Actor",
    "activity": "Activity",
    "physical_entity": "Physical Entity",
    "unclassified": "Unclassified",
    "entity": "Entity",
}

QUADRANT_COLOR: dict[str, str] = {
    "information": "#2563eb",
    "actor": "#7c3aed",
    "activity": "#ea580c",
    "physical_entity": "#059669",
    "unclassified": "#64748b",
    "entity": "#0f172a",
}

REPORT_EXAMPLES_MISMATCH = 14
REPORT_EXAMPLES_PARENT = 14


def stratified_examples(
    items: list[_T],
    key: Callable[[_T], Any],
    per_key: int,
    max_total: int,
) -> list[_T]:
    buckets: dict[Any, list[_T]] = defaultdict(list)
    for x in items:
        buckets[key(x)].append(x)
    out: list[_T] = []
    for _k in sorted(buckets, key=lambda x: (str(type(x)), str(x))):
        out.extend(buckets[_k][:per_key])
        if len(out) >= max_total:
            break
    return out[:max_total]


def svg_horizontal_bars(
    title: str,
    pairs: list[tuple[str, int, str]],
    width: int = 640,
    row_h: int = 26,
    margin_l: int = 160,
) -> str:
    """pairs: (label, value, bar_hex_color). Simple SVG horizontal bar chart."""
    if not pairs:
        return f'<p class="empty">No data for: {title}</p>'
    maxv = max(v for _l, v, _c in pairs) or 1
    inner_w = width - margin_l - 24
    h = 36 + len(pairs) * row_h
    lines: list[str] = [
        f'<figure class="chart"><figcaption>{title}</figcaption>',
        f'<svg class="bar-svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}">',
        '<rect width="100%" height="100%" fill="#f8fafc" rx="6"/>',
    ]
    y = 28
    for lab, val, col in pairs:
        bw = int(inner_w * val / maxv)
        lines.append(
            f'<text x="8" y="{y + 14}" font-size="12" fill="#334155" font-family="system-ui,sans-serif">{lab}</text>'
        )
        lines.append(
            f'<rect x="{margin_l}" y="{y}" width="{bw}" height="{18}" fill="{col}" rx="3"/>'
        )
        lines.append(
            f'<text x="{margin_l + bw + 6}" y="{y + 14}" font-size="12" fill="#475569" font-family="ui-monospace,monospace">{val}</text>'
        )
        y += row_h
    lines.append("</svg></figure>")
    return "\n".join(lines)


def esc_html(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def quadrant_pill(q: str) -> str:
    col = QUADRANT_COLOR.get(q, "#64748b")
    lab = QUADRANT_LABEL.get(q, q)
    return f'<span class="pill" style="--pill-bg:{col}">{esc_html(lab)}</span>'


def extract_supersenses(description: str | None) -> frozenset[str]:
    if not description:
        return frozenset()
    m = SUPERLINE.search(description)
    if not m:
        return frozenset()
    return frozenset(f"noun.{x}" for x in NOUN_SS.findall(m.group(1)))


def extract_path(description: str | None) -> str:
    if not description:
        return ""
    m = PATHLINE.search(description)
    return (m.group(1) or "").strip() if m else ""


def extract_virtual(description: str | None) -> bool | None:
    if not description:
        return None
    m = VIRTLINE.search(description)
    if not m:
        return None
    return m.group(1) == "True"


def item_record(title: str, description: str | None, ss: frozenset[str]) -> dict[str, Any]:
    return {
        "title": title,
        "path": extract_path(description),
        "virtual": extract_virtual(description),
        "supersenses": sorted(ss),
    }


def load_highlevel_maps() -> tuple[list[dict[str, Any]], dict[str, str], set[str]]:
    """Returns (categories template), supersense -> quadrant id, all known subcategory ids."""
    raw = json.loads(HIGHLEVEL.read_text(encoding="utf-8"))
    ss_to_quadrant: dict[str, str] = {}
    known_sub: set[str] = set()
    for cat in raw:
        cid = cat["id"]
        for sub in cat.get("subcategories") or []:
            sid = sub["id"]
            known_sub.add(sid)
            ss_to_quadrant[sid] = cid
    return raw, ss_to_quadrant, known_sub


def build_empty_ontology(categories_template: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cat in categories_template:
        c = {
            "id": cat["id"],
            "label": cat.get("label", cat["id"]),
            "description": cat.get("description", ""),
            "subcategories": [],
        }
        for sub in cat.get("subcategories") or []:
            c["subcategories"].append(
                {
                    "id": sub["id"],
                    "description": sub.get("description", ""),
                    "items": [],
                }
            )
        out.append(c)
    return out


def subcat_index(ontology: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """supersense id -> (category_idx, subcategory_idx)"""
    idx: dict[str, tuple[int, int]] = {}
    for ci, cat in enumerate(ontology):
        for si, sub in enumerate(cat["subcategories"]):
            idx[sub["id"]] = (ci, si)
    return idx


def walk_noun_tree(
    node: dict[str, Any],
    depth: int,
    branch_key: str | None,
    parent_ss: frozenset[str],
    rows: list[dict[str, Any]],
) -> None:
    title = node.get("title") or ""
    desc = node.get("description")
    ss = extract_supersenses(desc)
    quadrant = ENTITY_BRANCH_TO_QUADRANT.get(branch_key) if branch_key else None

    rows.append(
        {
            "title": title,
            "description": desc,
            "supersenses": ss,
            "tree_branch_key": branch_key,
            "tree_quadrant": quadrant,
            "parent_supersenses": parent_ss,
            "depth": depth,
        }
    )

    specs = node.get("specializations") or {}
    if not isinstance(specs, dict):
        return
    for sk, child in specs.items():
        if not isinstance(child, dict):
            continue
        next_branch = branch_key
        if depth == 0:
            next_branch = sk
        walk_noun_tree(child, depth + 1, next_branch, ss, rows)


def supersense_highlevel_set(
    ss: frozenset[str], ss_to_q: dict[str, str]
) -> frozenset[str]:
    qs: set[str] = set()
    for s in ss:
        q = ss_to_q.get(s)
        if q:
            qs.add(q)
    return frozenset(qs)


def main() -> None:
    cats_template, ss_to_quadrant, known_sub = load_highlevel_maps()
    ontology = build_empty_ontology(cats_template)
    idx = subcat_index(ontology)

    data = json.loads(NOUN_ONTO.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for _k, root in data.items():
        if isinstance(root, dict):
            walk_noun_tree(root, 0, None, frozenset(), rows)

    # Place items into supersenses-ontology (duplicate across subcats / multisense)
    other_items: dict[str, list[dict[str, Any]]] = {}

    for r in rows:
        rec = item_record(r["title"], r["description"], r["supersenses"])
        placed = False
        for s in r["supersenses"]:
            if s in idx:
                ci, si = idx[s]
                ontology[ci]["subcategories"][si]["items"].append(deepcopy(rec))
                placed = True
            else:
                other_items.setdefault(s, []).append(deepcopy(rec))
        if not r["supersenses"] and r["title"]:
            other_items.setdefault("_no_supersense_parsed", []).append(deepcopy(rec))

    other_block: dict[str, Any] | None = None
    if other_items:
        subs = []
        for sid in sorted(other_items.keys(), key=lambda x: (x.startswith("_"), str(x))):
            subs.append({"id": sid, "description": "", "items": other_items[sid]})
        other_block = {
            "id": "_other",
            "label": "Other / unmapped supersenses",
            "description": "Supersenses not listed in supersenses-highlevel.json, or items with no parsed supersenses.",
            "subcategories": subs,
        }

    out_doc: dict[str, Any] = {
        "meta": {
            "source_noun_ontology": "noun-ontology.json",
            "source_highlevel": "supersenses-highlevel.json",
            "item_count_unique_titles": len({r["title"] for r in rows if r.get("title")}),
            "node_count": len(rows),
        },
        "tree_branch_mapping": {
            "noun_ontology_json_key": "quadrant_id (matches supersenses-highlevel category id)",
            **{k: v for k, v in ENTITY_BRANCH_TO_QUADRANT.items()},
        },
        "categories": ontology,
    }
    if other_block:
        out_doc["categories"].append(other_block)

    text = json.dumps(out_doc, ensure_ascii=False, indent=2)
    OUT_ONTO.write_text(text, encoding="utf-8")
    OUT_ONTO_ALT.write_text(text, encoding="utf-8")

    # --- Mismatch: tree quadrant vs supersense-derived quadrants ---
    big4 = frozenset({"information", "actor", "activity", "physical_entity"})
    mismatches: list[dict[str, Any]] = []
    for r in rows:
        tq = r["tree_quadrant"]
        if tq is None:
            continue
        ss_set = r["supersenses"]
        sq = supersense_highlevel_set(ss_set, ss_to_quadrant)
        if tq == "unclassified":
            continue
        if tq in big4:
            if not sq:
                mismatches.append(
                    {
                        "title": r["title"],
                        "path": extract_path(r["description"]),
                        "reason": "no_mapped_supersense_for_quadrant_compare",
                        "tree_quadrant": tq,
                        "tree_branch_key": r["tree_branch_key"],
                        "supersenses": sorted(ss_set),
                        "supersense_derived_quadrants": [],
                    }
                )
            elif tq not in sq:
                mismatches.append(
                    {
                        "title": r["title"],
                        "path": extract_path(r["description"]),
                        "reason": "tree_quadrant_not_in_supersense_derived_set",
                        "tree_quadrant": tq,
                        "tree_branch_key": r["tree_branch_key"],
                        "supersenses": sorted(ss_set),
                        "supersense_derived_quadrants": sorted(sq),
                    }
                )

    # Parent supersense set differences (exclude root)
    parent_diffs: list[dict[str, Any]] = []
    for r in rows:
        if r["depth"] == 0:
            continue
        c, p = r["supersenses"], r["parent_supersenses"]
        if c == p:
            continue
        parent_diffs.append(
            {
                "title": r["title"],
                "path": extract_path(r["description"]),
                "child_supersenses": sorted(c),
                "parent_supersenses": sorted(p),
                "only_on_child": sorted(c - p),
                "only_on_parent": sorted(p - c),
            }
        )

    # --- Visual HTML report (summary + charts + example rows only) ---
    n_ok_tree = sum(
        1
        for r in rows
        if r["tree_quadrant"] in big4
        and r["tree_quadrant"] != "unclassified"
        and (
            supersense_highlevel_set(r["supersenses"], ss_to_quadrant)
            and r["tree_quadrant"] in supersense_highlevel_set(r["supersenses"], ss_to_quadrant)
        )
    )
    n_big4_nodes = sum(1 for r in rows if r["tree_quadrant"] in big4)

    by_reason = Counter(m["reason"] for m in mismatches)
    by_tree_q = Counter(m["tree_quadrant"] for m in mismatches)
    reason_labels = {
        "no_mapped_supersense_for_quadrant_compare": "No mappable supersense (e.g. only noun.Tops)",
        "tree_quadrant_not_in_supersense_derived_set": "Supersenses map to other quadrant(s), not the tree branch",
    }
    reason_pairs = [
        (reason_labels.get(k, k), v, "#94a3b8") for k, v in sorted(by_reason.items(), key=lambda x: -x[1])
    ]
    tree_pairs = [
        (QUADRANT_LABEL.get(q, q), c, QUADRANT_COLOR.get(q, "#64748b"))
        for q, c in sorted(by_tree_q.items(), key=lambda x: -x[1])
    ]

    chart_reason = svg_horizontal_bars("Flagged items by reason", reason_pairs)
    chart_tree = svg_horizontal_bars("Flagged items by tree quadrant", tree_pairs)

    mm_examples = stratified_examples(
        mismatches,
        key=lambda m: (m["reason"], m["tree_quadrant"]),
        per_key=2,
        max_total=REPORT_EXAMPLES_MISMATCH,
    )
    pd_examples = stratified_examples(
        parent_diffs,
        key=lambda d: (tuple(d["only_on_child"]), tuple(d["only_on_parent"])),
        per_key=1,
        max_total=REPORT_EXAMPLES_PARENT,
    )

    def mismatch_card(m: dict[str, Any]) -> str:
        derived = " ".join(quadrant_pill(x) for x in m["supersense_derived_quadrants"]) or (
            '<span class="pill pill-muted">(none from high-level map)</span>'
        )
        reason_short = (
            "Supersense labels do not map to any of the four high-level quadrants."
            if m["reason"] == "no_mapped_supersense_for_quadrant_compare"
            else "Mapped quadrants from supersenses do not include this item's tree branch."
        )
        ss = ", ".join(esc_html(s) for s in m["supersenses"][:10])
        if len(m["supersenses"]) > 10:
            ss += "…"
        return f"""<article class="card card-mm">
<div class="card-head"><strong class="item-title">{esc_html(m['title'])}</strong>
<span class="branch-tag">{esc_html(m['tree_branch_key'] or '')}</span></div>
<div class="flow">
  <div class="flow-col"><span class="flow-label">Where it lives in the tree</span>{quadrant_pill(m['tree_quadrant'])}</div>
  <span class="flow-arrow" aria-hidden="true">&#8596;</span>
  <div class="flow-col"><span class="flow-label">What supersenses imply</span><div class="pill-row">{derived}</div></div>
</div>
<p class="card-reason">{esc_html(reason_short)}</p>
<p class="mono path-line">{esc_html(m['path'][:140])}</p>
<p class="mono ss-line"><span class="ss-label">Supersenses:</span> {ss or '—'}</p>
</article>"""

    def parent_card(d: dict[str, Any]) -> str:
        oc = ", ".join(esc_html(x) for x in d["only_on_child"]) or "—"
        op = ", ".join(esc_html(x) for x in d["only_on_parent"]) or "—"
        return f"""<article class="card card-pd">
<strong class="item-title">{esc_html(d['title'])}</strong>
<div class="diff-grid">
  <div><span class="flow-label">Child</span><p class="mono">{esc_html(', '.join(d['child_supersenses']))}</p></div>
  <div><span class="flow-label">Parent</span><p class="mono">{esc_html(', '.join(d['parent_supersenses']))}</p></div>
</div>
<p class="diff-detail"><span class="tag tag-add">only on child</span> <span class="mono">{oc}</span></p>
<p class="diff-detail"><span class="tag tag-drop">only on parent</span> <span class="mono">{op}</span></p>
<p class="mono path-line">{esc_html(d['path'][:120])}</p>
</article>"""

    cards_mm = "\n".join(mismatch_card(m) for m in mm_examples) or '<p class="empty">No mismatches.</p>'
    cards_pd = "\n".join(parent_card(d) for d in pd_examples) or '<p class="empty">None.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supersenses ontology report</title>
<style>
:root {{
  --bg: #0f172a;
  --surface: #ffffff;
  --muted: #64748b;
  --border: #e2e8f0;
  --accent: #38bdf8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  background: linear-gradient(165deg, #0f172a 0%, #1e293b 42%, #f1f5f9 42%);
  color: #1e293b;
  min-height: 100vh;
}}
.wrap {{ max-width: 920px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }}
header.hero {{
  color: #f8fafc;
  margin-bottom: 2rem;
}}
header.hero h1 {{
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin: 0 0 0.5rem;
}}
header.hero p {{
  margin: 0;
  color: #94a3b8;
  font-size: 0.95rem;
  max-width: 52ch;
}}
.stats {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.75rem;
  margin: 1.5rem 0 2rem;
}}
.stat {{
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(148,163,184,0.25);
  border-radius: 12px;
  padding: 1rem 1.1rem;
  color: #f8fafc;
}}
.stat b {{
  display: block;
  font-size: 1.65rem;
  font-weight: 700;
  color: #fff;
  line-height: 1.1;
}}
.stat span {{
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #94a3b8;
}}
.panel {{
  background: var(--surface);
  border-radius: 16px;
  box-shadow: 0 4px 24px rgba(15,23,42,0.08);
  padding: 1.5rem 1.35rem;
  margin-bottom: 1.5rem;
}}
.panel h2 {{
  margin: 0 0 0.35rem;
  font-size: 1.15rem;
}}
.panel > .lead {{
  margin: 0 0 1rem;
  color: var(--muted);
  font-size: 0.9rem;
  line-height: 1.5;
}}
.charts {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.25rem;
}}
@media (max-width: 700px) {{ .charts {{ grid-template-columns: 1fr; }} }}
figure.chart {{
  margin: 0;
  padding: 0.75rem;
  background: #f8fafc;
  border-radius: 12px;
  border: 1px solid var(--border);
}}
figure.chart figcaption {{
  font-size: 0.8rem;
  font-weight: 600;
  color: #475569;
  margin-bottom: 0.35rem;
}}
.bar-svg {{ display: block; width: 100%; height: auto; max-width: 100%; }}
.legend {{
  font-size: 0.8rem;
  color: var(--muted);
  margin: 0 0 1rem;
  padding: 0.65rem 0.85rem;
  background: #f8fafc;
  border-radius: 8px;
  border-left: 4px solid var(--accent);
}}
.examples-title {{
  font-size: 0.85rem;
  font-weight: 600;
  color: #475569;
  margin: 0 0 0.75rem;
}}
.card {{
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1rem 0.85rem;
  margin-bottom: 0.75rem;
  background: #fff;
}}
.card-head {{
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.5rem 0.75rem;
  margin-bottom: 0.65rem;
}}
.item-title {{ font-size: 1rem; }}
.branch-tag {{
  font-size: 0.72rem;
  background: #f1f5f9;
  color: #475569;
  padding: 0.2rem 0.5rem;
  border-radius: 6px;
  font-family: ui-monospace, monospace;
}}
.flow {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem 0.75rem;
  margin: 0.5rem 0;
}}
.flow-col {{ flex: 1; min-width: 140px; }}
.flow-label {{
  display: block;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
  margin-bottom: 0.25rem;
}}
.flow-arrow {{
  font-size: 1.25rem;
  color: #cbd5e1;
  flex-shrink: 0;
}}
.pill-row {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
.pill {{
  display: inline-block;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  color: #fff;
  background: var(--pill-bg, #64748b);
}}
.pill-muted {{ background: #e2e8f0; color: #475569; }}
.card-reason {{ font-size: 0.85rem; margin: 0.35rem 0 0.25rem; color: #475569; }}
.path-line, .ss-line {{ font-size: 0.72rem; color: #64748b; margin: 0.25rem 0; word-break: break-all; }}
.ss-label {{ color: #94a3b8; }}
.mono {{ font-family: ui-monospace, Consolas, monospace; font-size: 0.78rem; }}
.diff-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  margin: 0.5rem 0;
}}
@media (max-width: 560px) {{ .diff-grid {{ grid-template-columns: 1fr; }} }}
.diff-detail {{ margin: 0.35rem 0; font-size: 0.85rem; }}
.tag {{
  display: inline-block;
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  margin-right: 0.35rem;
}}
.tag-add {{ background: #dcfce7; color: #166534; }}
.tag-drop {{ background: #fee2e2; color: #991b1b; }}
.empty {{ color: var(--muted); font-style: italic; }}
footer {{
  text-align: center;
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: 1rem;
}}
code {{ font-size: 0.88em; background: #f1f5f9; padding: 0.1em 0.35em; border-radius: 4px; }}
</style></head><body>
<div class="wrap">
<header class="hero">
  <h1>Supersenses ontology report</h1>
  <p>High-level view of how the noun tree lines up with supersense-based grouping. Full data lives in <code>supersenses-ontology.json</code>.</p>
</header>

<div class="stats">
  <div class="stat"><b>{len(rows)}</b><span>Nodes in noun tree</span></div>
  <div class="stat"><b>{n_big4_nodes}</b><span>Under main four branches</span></div>
  <div class="stat"><b>{len(mismatches)}</b><span>Tree vs supersense flags</span></div>
  <div class="stat"><b>{len(parent_diffs)}</b><span>Parent supersense diffs</span></div>
</div>

<section class="panel">
  <h2>1. Tree branch vs supersense quadrants</h2>
  <p class="lead">Each item sits under one of four children of <code>Entity</code> (Information, Actor, Activity via <code>process</code>, Physical Entity). Independently, its <code>noun.*</code> tags map into the same four quadrants using <code>supersenses-highlevel.json</code>. We flag items where the tree quadrant is not supported by any mapped supersense, or only unmapped tags (e.g. <code>noun.Tops</code>) appear. Items under <code>[added] Unclassified</code> are not scored here. <strong>{n_ok_tree}</strong> nodes under the main four branches align (tree quadrant appears in the supersense-derived set).</p>
  <div class="charts">
    {chart_reason}
    {chart_tree}
  </div>
  <p class="legend"><strong>Examples below</strong> (sample of {len(mm_examples)}; total flagged: {len(mismatches)}). Scan cards to see typical mismatch shapes.</p>
  <p class="examples-title">Sample mismatches</p>
  {cards_mm}
</section>

<section class="panel">
  <h2>2. Supersenses differ from immediate tree parent</h2>
  <p class="lead">Compared to the parent node in the <code>specializations</code> tree. Any difference in the supersense set is listed; most of the ontology repeats parent tags, so the total count is a minority of nodes.</p>
  <p class="legend"><strong>Examples</strong> (sample of {len(pd_examples)} of {len(parent_diffs)}). Each card shows what changed between parent and child.</p>
  <p class="examples-title">Sample parent/child differences</p>
  {cards_pd}
</section>

<footer>Regenerate with <code>python build_supersenses_ontology.py</code></footer>
</div>
</body></html>"""

    OUT_REPORT.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
