"""
merge.py - Merge accepted.csv and reviewed.csv into final dataset.
Performs final deduplication and validation before producing the merged output.
"""
import argparse, csv, logging, os, sys
from collections import Counter
from typing import Dict, List

import pandas as pd
from rapidfuzz import fuzz
from configs import ACOSHDConfig, get_config
from validate_stage1 import span_grounded

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "sample_id", "source_dataset", "source_id", "original_text", "cleaned_text",
    "aspect_target", "aspect_category", "opinion_span",
    "stance", "explanation",
    "mapped_stance", "cost_usd", "s1_attempts", "s2_attempts",
    "status", "errors", "timestamp",
]

FINAL_FIELDS = [
    "sample_id", "text", "aspect_target", "aspect_category",
    "opinion_span", "stance", "explanation", "source_dataset",
]


def load_and_validate(path: str, label: str) -> pd.DataFrame:
    """Load a CSV and perform basic validation."""
    if not os.path.exists(path):
        logger.warning("%s not found: %s", label, path)
        return pd.DataFrame(columns=CSV_FIELDS)
    df = pd.read_csv(path, dtype=str).fillna("")
    logger.info("Loaded %s: %d rows from %s", label, len(df), path)
    return df


def dedup_final(df: pd.DataFrame, threshold: int = 85) -> pd.DataFrame:
    """Remove near-duplicate rows based on cleaned_text fuzzy matching."""
    if len(df) <= 1:
        return df
    keep = [True] * len(df)
    texts = df["cleaned_text"].tolist() if "cleaned_text" in df.columns else df["text"].tolist()
    for i in range(len(texts)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(texts)):
            if not keep[j]:
                continue
            if fuzz.ratio(texts[i].lower(), texts[j].lower()) >= threshold:
                keep[j] = False
                logger.debug("Duplicate pair: %d ↔ %d", i, j)
    before = len(df)
    df = df[keep].reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        logger.info("Removed %d near-duplicates", removed)
    return df


def quality_filter(df: pd.DataFrame, schema) -> pd.DataFrame:
    """Final quality gate: remove rows with empty required fields."""
    required = ["aspect_target", "aspect_category", "opinion_span", "stance", "explanation"]
    mask = pd.Series([True] * len(df))
    for field in required:
        if field in df.columns:
            mask &= df[field].str.strip().astype(bool)
    before = len(df)
    df = df[mask].reset_index(drop=True)
    if before - len(df) > 0:
        logger.info("Quality filter removed %d rows with empty fields", before - len(df))

    # Filter invalid categories
    if hasattr(schema, "aspect_categories"):
        valid_cats = set(schema.aspect_categories)
        cat_mask = df["aspect_category"].isin(valid_cats) if "aspect_category" in df.columns else True
        cat_removed = len(df) - cat_mask.sum()
        if cat_removed > 0:
            logger.info("Removed %d rows with invalid categories", cat_removed)
        df = df[cat_mask].reset_index(drop=True)

    # Filter invalid stances
    if hasattr(schema, "stance_labels"):
        valid_stances = set(schema.stance_labels)
        st_mask = df["stance"].isin(valid_stances) if "stance" in df.columns else True
        st_removed = len(df) - st_mask.sum()
        if st_removed > 0:
            logger.info("Removed %d rows with invalid stances", st_removed)
        df = df[st_mask].reset_index(drop=True)

    return df


def merge_datasets(cfg: ACOSHDConfig = None):
    """Merge accepted.csv and reviewed.csv into final output."""
    if cfg is None:
        cfg = get_config()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    accepted_path = cfg.data.accepted_path()
    repaired_path = cfg.data.repaired_path()
    reviewed_path = cfg.data.reviewed_path()
    output_path = os.path.join(cfg.data.output_dir, "generated_dataset.csv")

    # Load all sources
    accepted_df = load_and_validate(accepted_path, "accepted")
    repaired_df = load_and_validate(repaired_path, "repaired")
    reviewed_df = load_and_validate(reviewed_path, "reviewed")

    # Merge
    merged = pd.concat([accepted_df, repaired_df, reviewed_df], ignore_index=True)
    logger.info("Total merged rows: %d (accepted=%d, repaired=%d, reviewed=%d)",
                len(merged), len(accepted_df), len(repaired_df), len(reviewed_df))

    if len(merged) == 0:
        logger.warning("No data to merge!")
        return

    # Deduplicate
    merged = dedup_final(merged, cfg.filtering.dedup_fuzzy_threshold)

    # Quality filter
    merged = quality_filter(merged, cfg.schema)

    # Re-assign sample IDs
    merged = merged.reset_index(drop=True)
    merged["sample_id"] = range(1, len(merged) + 1)

    # Create final format
    final = pd.DataFrame()
    final["sample_id"] = merged["sample_id"]
    final["text"] = merged.get("cleaned_text", merged.get("original_text", ""))
    final["aspect_target"] = merged["aspect_target"]
    final["aspect_category"] = merged["aspect_category"]
    final["opinion_span"] = merged["opinion_span"]
    final["stance"] = merged["stance"]
    final["explanation"] = merged["explanation"]
    final["source_dataset"] = merged.get("source_dataset", "")

    # Save
    final.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Final dataset saved: %s (%d samples)", output_path, len(final))

    # ── Summary statistics ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ACOS-HD Generated Dataset Summary")
    print(f"{'='*60}")
    print(f"  Total samples: {len(final)}")

    if "stance" in final.columns:
        stance_counts = final["stance"].value_counts()
        print(f"\n  Stance distribution:")
        for label, count in stance_counts.items():
            pct = 100 * count / len(final)
            print(f"    {label}: {count} ({pct:.1f}%)")

    if "aspect_category" in final.columns:
        cat_counts = final["aspect_category"].value_counts()
        print(f"\n  Category distribution:")
        for cat, count in cat_counts.items():
            pct = 100 * count / len(final)
            print(f"    {cat}: {count} ({pct:.1f}%)")

    if "source_dataset" in final.columns:
        src_counts = final["source_dataset"].value_counts()
        print(f"\n  Source distribution:")
        for src, count in src_counts.items():
            pct = 100 * count / len(final)
            print(f"    {src}: {count} ({pct:.1f}%)")

    print(f"\n  Output: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ACOS-HD results")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output file path")
    args = parser.parse_args()

    cfg = get_config()
    merge_datasets(cfg)
