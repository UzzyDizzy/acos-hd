"""
human_review.py - Interactive CLI for reviewing failed/queued samples.
Loads review_queue.csv, lets reviewer accept/edit/reject samples, saves to reviewed.csv.
"""
import csv, json, os, sys
from typing import Dict, List
import pandas as pd
from configs import ACOSHDConfig, get_config, SchemaConfig
from validate_stage1 import validate_stage1, span_grounded
from validate_stage2 import validate_stage2

CSV_FIELDS = [
    "sample_id", "source_dataset", "source_id", "original_text", "cleaned_text",
    "aspect_target", "aspect_category", "opinion_span",
    "stance", "explanation",
    "mapped_stance", "cost_usd", "s1_attempts", "s2_attempts",
    "status", "errors", "timestamp",
]


def _init_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def _display_sample(row: Dict, idx: int, total: int, schema: SchemaConfig):
    """Display a sample for review."""
    print(f"\n{'='*70}")
    print(f"  Review [{idx+1}/{total}]  |  Sample ID: {row.get('sample_id', 'N/A')}")
    print(f"  Source: {row.get('source_dataset', '')} | ID: {row.get('source_id', '')}")
    print(f"{'='*70}")
    print(f"\n  POST (cleaned):")
    text = row.get("cleaned_text", row.get("original_text", ""))
    # Wrap long text
    for i in range(0, len(text), 80):
        print(f"    {text[i:i+80]}")

    print(f"\n  ACOS-HD Annotation:")
    print(f"    Aspect Target:   {row.get('aspect_target', '—')}")
    print(f"    Aspect Category: {row.get('aspect_category', '—')}")
    print(f"    Opinion Span:    {row.get('opinion_span', '—')}")
    print(f"    Stance:          {row.get('stance', '—')}")
    print(f"    Explanation:     {row.get('explanation', '—')}")

    # Show errors
    errors = row.get("errors", "")
    if errors:
        print(f"\n  ⚠ ERRORS: {errors}")

    # Validation status
    post = text
    at = row.get("aspect_target", "")
    op = row.get("opinion_span", "")
    ex = row.get("explanation", "")
    cat = row.get("aspect_category", "")
    stance = row.get("stance", "")

    checks = []
    if at and span_grounded(at, post, 0.6):
        checks.append("✓ aspect_target grounded")
    elif at:
        checks.append("✗ aspect_target NOT grounded")
    if op and span_grounded(op, post, 0.6):
        checks.append("✓ opinion_span grounded")
    elif op:
        checks.append("✗ opinion_span NOT grounded")
    if ex and span_grounded(ex, post, 0.5):
        checks.append("✓ explanation grounded")
    elif ex:
        checks.append("✗ explanation NOT grounded")
    if cat in schema.aspect_categories:
        checks.append("✓ category valid")
    elif cat:
        checks.append("✗ category INVALID")
    if stance in schema.stance_labels:
        checks.append("✓ stance valid")
    elif stance:
        checks.append("✗ stance INVALID")

    if checks:
        print(f"\n  Validation: {' | '.join(checks)}")

    print(f"\n  Actions:")
    print(f"    [a] Accept as-is")
    print(f"    [e] Edit fields manually")
    print(f"    [r] Reject (discard)")
    print(f"    [s] Skip for later")
    print(f"    [q] Quit review session")


def _edit_fields(row: Dict, schema: SchemaConfig) -> Dict:
    """Interactive field editing."""
    editable = ["aspect_target", "aspect_category", "opinion_span", "stance", "explanation"]
    print(f"\n  Edit fields (press Enter to keep current value):")
    for field in editable:
        current = row.get(field, "")
        if field == "aspect_category":
            print(f"    Valid categories: {', '.join(schema.aspect_categories)}")
        elif field == "stance":
            print(f"    Valid stances: {', '.join(schema.stance_labels)}")
        new_val = input(f"    {field} [{current}]: ").strip()
        if new_val:
            row[field] = new_val
    return row


def run_review(cfg: ACOSHDConfig = None):
    """Main review loop."""
    if cfg is None:
        cfg = get_config()

    queue_path = cfg.data.review_queue_path()
    reviewed_path = cfg.data.reviewed_path()

    if not os.path.exists(queue_path):
        print(f"No review queue found at: {queue_path}")
        return

    df = pd.read_csv(queue_path, dtype=str).fillna("")
    if len(df) == 0:
        print("Review queue is empty.")
        return

    _init_csv(reviewed_path)

    print(f"\n{'#'*70}")
    print(f"  ACOS-HD Human Review")
    print(f"  Queue: {len(df)} samples pending")
    print(f"{'#'*70}")

    accepted = 0
    rejected = 0
    skipped = 0

    for idx, row_series in df.iterrows():
        row = row_series.to_dict()
        _display_sample(row, idx, len(df), cfg.schema)

        while True:
            action = input("\n  > ").strip().lower()
            if action == "a":
                row["status"] = "reviewed_accepted"
                row["errors"] = ""
                with open(reviewed_path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(
                        {k: row.get(k, "") for k in CSV_FIELDS}
                    )
                accepted += 1
                print("  ✓ Accepted")
                break
            elif action == "e":
                row = _edit_fields(row, cfg.schema)
                row["status"] = "reviewed_edited"
                row["errors"] = ""
                with open(reviewed_path, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(
                        {k: row.get(k, "") for k in CSV_FIELDS}
                    )
                accepted += 1
                print("  ✓ Edited and accepted")
                break
            elif action == "r":
                rejected += 1
                print("  ✗ Rejected")
                break
            elif action == "s":
                skipped += 1
                print("  → Skipped")
                break
            elif action == "q":
                print(f"\n  Session summary: {accepted} accepted, {rejected} rejected, {skipped} skipped")
                return
            else:
                print("  Invalid action. Use [a]ccept, [e]dit, [r]eject, [s]kip, or [q]uit.")

    print(f"\n{'='*70}")
    print(f"  Review complete: {accepted} accepted, {rejected} rejected, {skipped} skipped")
    print(f"  Reviewed samples saved to: {reviewed_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_review()
