#!/usr/bin/env python3
"""
JSON Hierarchy Error Checking with LLM Queries
============================================
- Goes through the hierarchy JSON, finds each "(Atomic Tasks)" group
- For every label in those groups, asks the model if it is misplaced.
- Provides path + local context (ancestor labels with synset hints, sibling atomics & sample O*NET lines)
- (Adjustable) Flags only two error types: VERB_ERROR, OBJECT_ERROR (or BOTH). Otherwise returns OK.
- Writes errors to a CSV as it goes with a progress bar showing % done, sec/check, and ETA (of completion).

Usage:
1. Configure the context parameters
2. Customize the system and user prompts for a specific use case
3. Run the script to process the hierarchy JSON and generate the error report CSV
"""

import os, sys, re, json, csv, time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime
from dotenv import load_dotenv; load_dotenv()

# Config

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_RETRIES = 3
DELAY_BETWEEN_CALLS = 1.0
MAX_TOKENS = 500
TEMPERATURE = 0.1

MAX_SIBLING_ATOMICS = 5          # number of neighbor atomics under the same "(Atomic Tasks)" bucket
MAX_ONET_EXAMPLES_PER_ATOMIC = 2 # per atomic task, how many O*NET lines to include
MAX_ANCESTOR_LABELS = 3          # deepest N labels from the path tail to include as context

OUT_CSV = "hierarchy_audit_errors.csv"
FIELDNAMES = ["ts","path","atomic_task","error_type","reason","confidence"]

PROGRESS_BAR_WIDTH = 30
SMOOTHING_ALPHA = 0.2

# LLM call
def call_llm(system_prompt: str, user_prompt: str, attempt: int = 1) -> str:
    try:
        print(f"\n    🤖 LLM API call (attempt {attempt})")
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=30
        )
        result = response.choices[0].message.content.strip()
        print(f"    ✅ LLM response: {result[:100]}{'...' if len(result) > 100 else ''}")
        return result

    except Exception as e:
        print(f"    ❌ LLM API error (attempt {attempt}): {e}")
        if attempt < MAX_RETRIES:
            delay = attempt * 2
            print(f"    ⏱️ Retrying in {delay} seconds...")
            time.sleep(delay)
            return call_llm(system_prompt, user_prompt, attempt + 1)
        else:
            print(f"    💥 Max retries ({MAX_RETRIES}) reached")
            return f"API_ERROR: {str(e)}"

# Walk hierarchy
def walk_hierarchy(node: Any, path: List[str]) -> List[Tuple[List[str], Dict[str, List[str]]]]:
    """
    Returns a list of (path_to_atomic_bucket, atomics_dict).
    Each atomics_dict is { atomic_task_name: [onet_example_lines...] } for a single "(Atomic Tasks)" bucket.
    """
    buckets = []
    if isinstance(node, dict):
        if "(Atomic Tasks)" in node and isinstance(node["(Atomic Tasks)"], dict):
            buckets.append((path + ["(Atomic Tasks)"], node["(Atomic Tasks)"]))
        for k, v in node.items():
            if k == "(Atomic Tasks)":
                continue
            buckets.extend(walk_hierarchy(v, path + [k]))
    return buckets

def extract_synset_hints(label: str) -> List[str]:
    m = re.search(r"\(([^)]*\.v\.\d+[^)]*)\)", label, flags=re.IGNORECASE)
    if not m:
        return []
    blob = m.group(1)
    hints = [p.strip() for p in re.split(r"[;,/]", blob) if ".v." in p]
    return hints[:8]

def build_context_for_bucket(path_to_bucket: List[str], atomics_dict: Dict[str, List[str]]) -> Dict[str, Any]:
    trimmed_path = path_to_bucket[-(MAX_ANCESTOR_LABELS + 1):]
    parent_label = trimmed_path[-2] if len(trimmed_path) >= 2 else ""
    synset_hints = extract_synset_hints(parent_label)

    all_atoms = list(atomics_dict.items())
    all_atoms.sort(key=lambda kv: kv[0].lower())

    siblings_sample = []
    for name, examples in all_atoms[:MAX_SIBLING_ATOMICS]:
        siblings_sample.append({
            "name": name,
            "onet_examples": (examples or [])[:MAX_ONET_EXAMPLES_PER_ATOMIC]
        })

    return {
        "path_tail": trimmed_path,
        "parent_label": parent_label,
        "synset_hints": synset_hints,
        "siblings_sample": siblings_sample
    }

# Prompts
SYSTEM_PROMPT = """You are auditing a hierarchical verb–object taxonomy.
Your job: decide if a given Atomic Task is misplaced under a specific hierarchy path.

Rules:
- Focus ONLY on two misclassification types: verb misfit or direct-object irrelevance (or both).
- Do NOT propose new labels or rewrites; this is a judgment-only check.
- Use the provided local context (path words, synset hints, sibling examples) to infer intended meaning.
- Respond STRICTLY in JSON format as specified below.
"""

USER_PROMPT = """You will judge ONE Atomic Task in context.

## Hierarchy Path
{path_tail}

## Parent Label
{parent_label}

## Synset Hints (if any)
{synset_hints}

## Sibling/Neighbor Atomic Tasks (for context)
{siblings_sample}

## Atomic Task to Judge
name: {atomic_name}
onet_examples: {atomic_examples}

## Decision you must return (strict JSON with keys below):
- verdict: "OK" or "ERROR"
- error_type: "VERB_ERROR" or "OBJECT_ERROR" or "BOTH" (or "N/A" if OK)
- reason: short explanation grounded in path vs. task (cite the conflicting verb/object)
- confidence: 1 (least) – 5 (most)
"""

def format_user_prompt(context: Dict[str,Any], atomic_name: str, atomic_examples: List[str]) -> str:
    def fmt_list(xs):
        return json.dumps(xs, ensure_ascii=False)

    return USER_PROMPT.format(
        path_tail=fmt_list(context["path_tail"]),
        parent_label=context["parent_label"],
        synset_hints=fmt_list(context["synset_hints"]),
        siblings_sample=json.dumps(context["siblings_sample"], ensure_ascii=False),
        atomic_name=atomic_name,
        atomic_examples=json.dumps((atomic_examples or [])[:MAX_ONET_EXAMPLES_PER_ATOMIC], ensure_ascii=False)
    )

# Progress bar and timing
def count_total_atomics(buckets: List[Tuple[List[str], Dict[str, List[str]]]]) -> int:
    return sum(len(atomics or {}) for _, atomics in buckets)

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"

def draw_progress(current: int, total: int, secs_per_check: float, start_time: float):
    pct = (current / total) if total else 0.0
    filled = int(PROGRESS_BAR_WIDTH * pct)
    bar = "█" * filled + "░" * (PROGRESS_BAR_WIDTH - filled)
    elapsed = time.time() - start_time
    remaining = max(0.0, (total - current) * secs_per_check) if total else 0.0
    sys.stdout.write(
        f"\r[{bar}] {pct*100:6.2f}%  {current}/{total}  "
        f"{secs_per_check:4.1f}s/check  "
        f"ETA {format_duration(remaining)}  "
        f"Elapsed {format_duration(elapsed)}"
    )
    sys.stdout.flush()

# Main
def main():
    src = "1014_hierarchycollapsedmerged.json"
    print(f"📁 Loading hierarchy JSON: {src}")
    with open(src, "r", encoding="utf-8") as f:
        tree = json.load(f)

    # CSV setup
    base_dir = Path(__file__).parent.resolve()
    out_csv_path = base_dir / OUT_CSV
    print(f"🗂️  Streaming CSV to: {out_csv_path}")

    need_header = (not out_csv_path.exists()) or (out_csv_path.stat().st_size == 0)
    f_csv = open(out_csv_path, "a", newline="", encoding="utf-8", buffering=1)
    w = csv.DictWriter(f_csv, fieldnames=FIELDNAMES)
    if need_header:
        w.writeheader()
        f_csv.flush()

    def write_row(row: dict):
        row.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        for k in FIELDNAMES:
            row.setdefault(k, "")
        w.writerow(row)
        f_csv.flush()

    # Compute totals
    buckets = walk_hierarchy(tree, [])
    total_atomics = count_total_atomics(buckets)
    print(f"🔎 Found {len(buckets)} atomic buckets containing {total_atomics} atomic tasks")

    streamed_error_count = 0
    processed = 0
    start_time = time.time()
    smoothed_secs = None

    try:
        for path_to_bucket, atomics in buckets:
            context = build_context_for_bucket(path_to_bucket, atomics)

            for atomic_name, onet_lines in (atomics or {}).items():
                iter_start = time.time()

                user_prompt = format_user_prompt(context, atomic_name, onet_lines)
                # Debug: Print relevant information
                print(f"\n---\nProcessing Atomic Task {atomic_name}")

                raw = call_llm(SYSTEM_PROMPT, user_prompt)

                if raw.startswith("API_ERROR"):
                    write_row({
                        "path": " → ".join(path_to_bucket),
                        "atomic_task": atomic_name,
                        "error_type": "API_ERROR",
                        "reason": raw,
                        "confidence": 1
                    })
                    streamed_error_count += 1
                else:
                    m = re.search(r"\{.*\}", raw, flags=re.S)
                    if not m:
                        write_row({
                            "path": " → ".join(path_to_bucket),
                            "atomic_task": atomic_name,
                            "error_type": "PARSE_ERROR",
                            "reason": raw[:300],
                            "confidence": 1
                        })
                        streamed_error_count += 1
                    else:
                        try:
                            result = json.loads(m.group(0))
                        except Exception as e:
                            write_row({
                                "path": " → ".join(path_to_bucket),
                                "atomic_task": atomic_name,
                                "error_type": "PARSE_ERROR",
                                "reason": f"{e}: {raw[:300]}",
                                "confidence": 1
                            })
                            streamed_error_count += 1
                        else:
                            verdict = (result.get("verdict") or "").upper()
                            if verdict == "ERROR":
                                err_type = (result.get("error_type") or "").upper() or "N/A"
                                reason = result.get("reason") or ""
                                conf = result.get("confidence") or 3
                                write_row({
                                    "path": " → ".join(path_to_bucket),
                                    "atomic_task": atomic_name,
                                    "error_type": err_type,
                                    "reason": reason,
                                    "confidence": conf
                                })
                                streamed_error_count += 1

                processed += 1
                # per-check timing (includes delay and all processing)
                iter_elapsed = time.time() - iter_start
                smoothed_secs = (
                    iter_elapsed if smoothed_secs is None
                    else (SMOOTHING_ALPHA * iter_elapsed + (1 - SMOOTHING_ALPHA) * smoothed_secs)
                )

                # Update progress bar
                draw_progress(processed, total_atomics, smoothed_secs or 0.0, start_time)

                # respect pacing between calls
                time.sleep(DELAY_BETWEEN_CALLS)

        sys.stdout.write("\n")
        print(f"✅ Completed. Streamed {streamed_error_count} error rows out of {total_atomics} atomics to {out_csv_path}.")

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print(f"Interrupted. Partial results are at: {out_csv_path}")
    finally:
        try:
            f_csv.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()