"""
Analyze noun-ontology.json: supersense counts, per-supersense frequencies,
depth vs supersense count, parent/ancestor overlap.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ONTOLOGY_PATH = Path(__file__).resolve().parent / "noun-ontology.json"
REPORT_DIR = Path(__file__).resolve().parent
SUPERLINE = re.compile(r"-\s*Supersenses:\s*([^\n]*)", re.IGNORECASE)
NOUN_SS = re.compile(r"\bnoun\.([A-Za-z0-9_]+)\b")


def extract_supersenses(description: str | None) -> frozenset[str]:
    if not description:
        return frozenset()
    m = SUPERLINE.search(description)
    if not m:
        return frozenset()
    line = m.group(1)
    # Normalize to canonical id e.g. noun.Tops
    found = NOUN_SS.findall(line)
    return frozenset(f"noun.{x}" for x in found)


def walk_node(
    node: dict[str, Any],
    depth: int,
    ancestor_ss: list[frozenset[str]],
    rows: list[dict[str, Any]],
) -> None:
    """ancestor_ss[0]=parent supersenses, [1]=grandparent, ... from immediate parent outward."""
    title = node.get("title") or ""
    desc = node.get("description")
    ss = extract_supersenses(desc)

    gens = node.get("generalizations") or []
    immediate_parent = gens[0] if gens else None

    parent_ss = ancestor_ss[0] if ancestor_ss else frozenset()
    grandparent_ss = ancestor_ss[1] if len(ancestor_ss) > 1 else frozenset()

    rows.append(
        {
            "title": title,
            "depth": depth,
            "n_supersenses": len(ss),
            "supersenses": ss,
            "generalizations": list(gens),
            "immediate_parent": immediate_parent,
            "parent_supersenses": parent_ss,
            "grandparent_supersenses": grandparent_ss,
        }
    )

    specs = node.get("specializations") or {}
    if isinstance(specs, dict):
        chain = [ss] + list(ancestor_ss)
        for child in specs.values():
            if isinstance(child, dict):
                walk_node(child, depth + 1, chain, rows)


def load_rows() -> list[dict[str, Any]]:
    with ONTOLOGY_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    rows: list[dict[str, Any]] = []
    for _root_name, root_node in data.items():
        if isinstance(root_node, dict):
            walk_node(root_node, 0, [], rows)
    return rows


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, statistics.stdev(xs)


def overlap_stats(child_ss: frozenset[str], ancestor_ss: frozenset[str]) -> dict[str, float]:
    if not child_ss:
        return {
            "jaccard": float("nan"),
            "frac_child_in_ancestor": float("nan"),
            "frac_ancestor_in_child": float("nan"),
            "intersection": 0.0,
        }
    inter = child_ss & ancestor_ss
    union = child_ss | ancestor_ss
    j = len(inter) / len(union) if union else float("nan")
    return {
        "jaccard": j,
        "frac_child_in_ancestor": len(inter) / len(child_ss),
        "frac_ancestor_in_child": (len(inter) / len(ancestor_ss)) if ancestor_ss else float("nan"),
        "intersection": float(len(inter)),
    }


def main() -> None:
    rows = load_rows()
    n_items = len(rows)
    counts = [r["n_supersenses"] for r in rows]
    n_zero_ss = sum(1 for c in counts if c == 0)

    # Per-supersense item counts
    ss_item_counter: Counter[str] = Counter()
    for r in rows:
        for s in r["supersenses"]:
            ss_item_counter[s] += 1

    # Histogram: # supersenses per item
    max_k = max(counts) if counts else 0
    hist_bins = range(0, max_k + 2)
    hist_vals = [sum(1 for c in counts if c == k) for k in range(max_k + 1)]

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(list(range(max_k + 1)), hist_vals, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Number of supersenses (noun.*) per ontology item")
    ax1.set_ylabel("Number of items")
    ax1.set_title("Histogram: supersenses per item")
    fig1.tight_layout()
    fig1.savefig(REPORT_DIR / "noun_supersenses_hist.png", dpi=150)
    plt.close(fig1)

    # Bar chart: items per supersense (sorted by count desc)
    labels, vals = zip(*ss_item_counter.most_common(), strict=False) if ss_item_counter else ([], [])
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.barh(labels[::-1], vals[::-1], color="teal", edgecolor="white")
    ax2.set_xlabel("Number of items tagged with this supersense")
    ax2.set_ylabel("Supersense")
    ax2.set_title("Items per supersense")
    fig2.tight_layout()
    fig2.savefig(REPORT_DIR / "noun_supersenses_per_label.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Depth vs supersense count
    by_depth: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_depth[r["depth"]].append(r["n_supersenses"])

    depth_stats: list[tuple[int, int, float, float, float, float]] = []
    for d in sorted(by_depth):
        xs = by_depth[d]
        med = statistics.median(xs)
        mu, sd = mean_std([float(x) for x in xs])
        depth_stats.append((d, len(xs), mu, med, sd, max(xs)))

    # Ancestor overlap from actual tree walk (not title string resolution)
    parent_rows: list[dict[str, float]] = []
    gp_rows: list[dict[str, float]] = []
    for r in rows:
        child_ss = r["supersenses"]
        if not child_ss:
            continue
        ps = r["parent_supersenses"]
        if r["depth"] > 0:
            parent_rows.append(overlap_stats(child_ss, ps))
        if r["depth"] > 1:
            gp_rows.append(overlap_stats(child_ss, r["grandparent_supersenses"]))

    def summarize_overlap(name: str, lst: list[dict[str, float]]) -> str:
        if not lst:
            return f"{name}: no pairs.\n"
        j = [x["jaccard"] for x in lst if not math.isnan(x["jaccard"])]
        fc = [x["frac_child_in_ancestor"] for x in lst]
        lines = [
            f"{name} (n={len(lst)} child-ancestor pairs, items with no supersenses excluded):",
            f"  Mean Jaccard(child, ancestor): {statistics.mean(j) if j else float('nan'):.4f}",
            f"  Median Jaccard: {statistics.median(j) if j else float('nan'):.4f}",
            f"  Mean fraction of child's supersenses present on ancestor: {statistics.mean(fc):.4f}",
            f"  Median fraction: {statistics.median(fc):.4f}",
            f"  Share with child's supersenses all on ancestor: {sum(1 for x in lst if x['frac_child_in_ancestor']>=0.999)/len(lst)*100:.2f}%",
            f"  Share with no overlap: {sum(1 for x in lst if x['frac_child_in_ancestor']<1e-9)/len(lst)*100:.2f}%",
            "",
        ]
        return "\n".join(lines)

    # Correlation depth vs n_supersenses
    depths = [r["depth"] for r in rows]
    corr = statistics.correlation(depths, counts) if len(rows) > 2 else float("nan")

    # Build HTML report
    hist_img = (REPORT_DIR / "noun_supersenses_hist.png").read_bytes()
    bar_img = (REPORT_DIR / "noun_supersenses_per_label.png").read_bytes()
    import base64

    b64_hist = base64.b64encode(hist_img).decode("ascii")
    b64_bar = base64.b64encode(bar_img).decode("ascii")

    lines_out: list[str] = []
    lines_out.append("<!DOCTYPE html>")
    lines_out.append('<html lang="en"><head><meta charset="utf-8">')
    lines_out.append("<title>Noun Supersenses</title>")
    lines_out.append(
        "<style>body{font-family:system-ui,Segoe UI,sans-serif;max-width:960px;margin:2rem auto;line-height:1.45;color:#222}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #ccc;padding:6px 10px;text-align:right}"
        "th{text-align:center;background:#f4f4f4}td:first-child{text-align:center}h1,h2{border-bottom:1px solid #ddd;padding-bottom:.2em}"
        "img{max-width:100%;height:auto}.note{color:#555;font-size:.95rem}</style>"
    )
    lines_out.append("</head><body>")
    lines_out.append("<h1>Noun Supersenses</h1>")
    lines_out.append(
        '<p class="note">Report generated from <code>noun-ontology.json</code>. '
        "Supersenses are parsed from the <code>- Supersenses:</code> line in each item's description "
        r"(patterns <code>noun.*</code>). Tree depth is nesting depth from the root node <code>Entity</code>.</p>"
    )

    lines_out.append("<h2>1. Supersenses per item (histogram)</h2>")
    lines_out.append(
        f"<p>Total ontology items (nodes in tree): <strong>{n_items}</strong>. "
        f"Distinct supersense labels: <strong>{len(ss_item_counter)}</strong>. "
        f"Overall mean supersenses per item: <strong>{statistics.mean(counts):.3f}</strong>, "
        f"median: <strong>{statistics.median(counts):.1f}</strong>, "
        f"stdev: <strong>{statistics.stdev(counts) if len(counts)>1 else 0:.3f}</strong>.</p>"
    )
    lines_out.append(f'<p><img src="data:image/png;base64,{b64_hist}" alt="Histogram"></p>')

    lines_out.append("<h3>Frequency table (# items with exactly k supersenses)</h3>")
    lines_out.append("<table><tr><th>k</th><th># items</th><th>% of items</th></tr>")
    for k in range(max_k + 1):
        v = hist_vals[k]
        pct = 100.0 * v / n_items if n_items else 0
        lines_out.append(f"<tr><td>{k}</td><td>{v}</td><td>{pct:.2f}</td></tr>")
    lines_out.append("</table>")

    lines_out.append("<h2>2. Items per supersense (bar chart)</h2>")
    lines_out.append(f'<p><img src="data:image/png;base64,{b64_bar}" alt="Bar chart"></p>')

    lines_out.append("<h2>3. Supersense count vs hierarchy depth</h2>")
    lines_out.append(
        f"<p>Pearson correlation between depth and number of supersenses: <strong>{corr:.4f}</strong> "
        "(positive means deeper nodes tend to have more supersenses; negative means fewer).</p>"
    )
    lines_out.append("<table><tr><th>Depth</th><th># items</th><th>Mean</th><th>Median</th><th>Stdev</th><th>Max</th></tr>")
    for d, n, mu, med, sd, mx in depth_stats:
        sd_s = f"{sd:.3f}" if not math.isnan(sd) else "-"
        lines_out.append(
            f"<tr><td>{d}</td><td>{n}</td><td>{mu:.3f}</td><td>{med:.1f}</td><td>{sd_s}</td><td>{mx}</td></tr>"
        )
    lines_out.append("</table>")

    lines_out.append("<h2>4. Inheritance: overlap with ancestors</h2>")
    lines_out.append(
        "<p>Using the <strong>tree structure</strong> (nested <code>specializations</code>), each node's "
        "supersenses are compared to its direct parent and to its grandparent (two levels up). "
        "Items with <strong>no parsed supersenses</strong> are excluded from these overlap rows. "
        f"There are <strong>{n_zero_ss}</strong> such items ({100 * n_zero_ss / n_items:.2f}% of all items).</p>"
    )
    lines_out.append("<pre>")
    lines_out.append(summarize_overlap("Immediate parent", parent_rows))
    lines_out.append(summarize_overlap("Grandparent (2nd generalization)", gp_rows))
    lines_out.append("</pre>")

    lines_out.append("</body></html>")
    out_path = REPORT_DIR / "Noun_Supersenses.html"
    out_path.write_text("\n".join(lines_out), encoding="utf-8")


if __name__ == "__main__":
    main()
