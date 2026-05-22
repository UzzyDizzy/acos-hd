"""
run_pipeline.py - Main ACOS-HD generation pipeline orchestrator.
Flow: Load → Filter → Clean → For each stance class: Generate → Validate → Repair → Accept/Queue
Implements the full pipeline from paper Figure 2.
"""
import csv, json, logging, os, pickle, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from configs import ACOSHDConfig, get_config, OPENAI_API_KEY
from data_filtering import (
    run_filtering, load_datasets, load_cached_candidates,
    is_duplicate_of_accepted, register_accepted, is_duplicate_of_gold,
)
from preprocessing import clean_text
from preprocessing import clean_texts_batch
from prompt_stage1 import load_few_shot_from_gold
from prompt_stage2 import load_few_shot_stage2_from_gold
from stage1 import generate_and_validate_stage1, get_cost_summary as s1_costs
from stage2 import generate_and_validate_stage2, get_cost_summary as s2_costs
from repair_stage1 import repair_stage1, load_llama_model
from repair_stage2 import repair_stage2
from validate_stage1 import validate_stage1
from validate_stage2 import validate_stage2

logger = logging.getLogger("acos_hd")

# ── CSV helpers ───────────────────────────────────────────────────────────
CSV_FIELDS = [
    "sample_id", "source_dataset", "source_id", "original_text", "cleaned_text",
    "aspect_target", "aspect_category", "opinion_span",
    "stance", "explanation",
    "mapped_stance", "cost_usd", "s1_attempts", "s2_attempts",
    "status", "errors", "timestamp",
]


def _init_csv(path: str):
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def _append_csv(path: str, row: Dict):
    """Append a single row to a CSV file."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def _save_checkpoint(state: Dict, path: str):
    """Save pipeline state for resume capability."""
    with open(path, "wb") as f:
        pickle.dump(state, f)


def _load_checkpoint(path: str) -> Optional[Dict]:
    """Load pipeline checkpoint if exists."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ── Logging setup ─────────────────────────────────────────────────────────
def setup_logging(cfg: ACOSHDConfig):
    os.makedirs(cfg.data.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.data.output_dir, cfg.pipeline.log_file)
    logging.basicConfig(
        level=getattr(logging, cfg.pipeline.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ── Process single sample ────────────────────────────────────────────────
def process_sample(
    row: pd.Series, sample_id: int, cfg: ACOSHDConfig,
    few_shot_s1, few_shot_s2, client,
) -> Dict:
    """Process a single candidate through Stage 1 → Validate → Repair → Stage 2 → Validate → Repair.
    Returns a result dict for CSV output.
    """
    # original_text = str(row.get("text", ""))
    # cleaned = clean_text(original_text, cfg.cleaning)
    # if cleaned is None:
    #     return {"status": "filtered", "errors": "Text filtered during cleaning"}

    # post_text = cleaned

    original_text = str(
        row.get("text","")
    )

    source_stance = row.get(
        "stance",
        "NONE"
    )

    mapped_stance = row.get(
        "mapped_stance",
        source_stance
    )

    # fallback if missing
    if mapped_stance not in [
        "HOPEFUL",
        "HATE",
        "NEUTRAL"
    ]:
        from data_filtering import map_stance

        mapped_stance = map_stance(
            source_stance
        )

    batch_result = clean_texts_batch(
        texts=[original_text],
        stances=[mapped_stance],
        cfg=cfg.cleaning
    )

    if not batch_result:
        return {
            "status":"filtered",
            "errors":"Filtered by GPT relevance"
        }

    post_text = batch_result[0][1]

    total_cost = 0.0
    s1_attempts = 0
    s2_attempts = 0

    # ── Stage 1: Structured Extraction (GPT-4.1-mini) ────────────────
    s1_out, s1_vr, s1_cost, s1_usage = generate_and_validate_stage1(
        post_text, cfg, few_shot_s1, client
    )
    total_cost += s1_cost
    s1_attempts = 1

    if s1_out is None:
        return _build_result(
            sample_id, row, original_text, post_text, None, None,
            total_cost, s1_attempts, 0, "failed", "Stage 1 generation returned None"
        )

    # Stage 1 repair loop if invalid
    if not s1_vr.valid:
        s1_out, s1_vr, repair_attempts = repair_stage1(
            post_text, s1_out, s1_vr, cfg
        )
        s1_attempts += repair_attempts

        if not s1_vr.valid:
            errs = "; ".join(f"{e.error_type}:{e.field}" for e in s1_vr.errors)
            return _build_result(
                sample_id, row, original_text, post_text, s1_out, None,
                total_cost, s1_attempts, 0, "review", f"Stage 1 repair failed: {errs}"
            )

    # ── Stage 2: Stance + Rationale (GPT-4.1-mini) ───────────────────
    s2_out, s2_vr, s2_cost, s2_usage = generate_and_validate_stage2(
        post_text, s1_out, cfg, few_shot_s2, client
    )
    total_cost += s2_cost
    s2_attempts = 1

    if s2_out is None:
        return _build_result(
            sample_id, row, original_text, post_text, s1_out, None,
            total_cost, s1_attempts, s2_attempts, "failed",
            "Stage 2 generation returned None"
        )

    # Stage 2 repair loop if invalid
    if not s2_vr.valid:
        s2_out, s2_vr, repair_attempts = repair_stage2(
            post_text, s1_out, s2_out, s2_vr, cfg
        )
        s2_attempts += repair_attempts

        if not s2_vr.valid:
            errs = "; ".join(f"{e.error_type}:{e.field}" for e in s2_vr.errors)
            return _build_result(
                sample_id, row, original_text, post_text, s1_out, s2_out,
                total_cost, s1_attempts, s2_attempts, "review",
                f"Stage 2 repair failed: {errs}"
            )

    # ── Both stages passed ───────────────────────────────────────────
    status = "repaired" if (s1_attempts > 1 or s2_attempts > 1 or
                           s1_vr.auto_fixed or s2_vr.auto_fixed) else "accepted"

    return _build_result(
        sample_id, row, original_text, post_text, s1_out, s2_out,
        total_cost, s1_attempts, s2_attempts, status, ""
    )


def _build_result(
    sample_id, row, original_text, post_text,
    s1_out, s2_out, cost, s1_att, s2_att, status, errors
) -> Dict:
    return {
        "sample_id": sample_id,
        "source_dataset": row.get("dataset", ""),
        "source_id": row.get("id", ""),
        "original_text": original_text,
        "cleaned_text": post_text or "",
        "aspect_target": (s1_out or {}).get("aspect_target", ""),
        "aspect_category": (s1_out or {}).get("aspect_category", ""),
        "opinion_span": (s1_out or {}).get("opinion_span", ""),
        "stance": (s2_out or {}).get("stance", ""),
        "explanation": (s2_out or {}).get("explanation", ""),
        "mapped_stance": row.get("mapped_stance", ""),
        "cost_usd": f"{cost:.6f}",
        "s1_attempts": s1_att,
        "s2_attempts": s2_att,
        "status": status,
        "errors": errors,
        "timestamp": datetime.now().isoformat(),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────
def run_pipeline(cfg: Optional[ACOSHDConfig] = None):
    """Execute the full ACOS-HD generation pipeline."""
    if cfg is None:
        cfg = get_config()

    setup_logging(cfg)
    logger.info("=" * 70)
    logger.info("ACOS-HD Generation Pipeline")
    logger.info("=" * 70)
    logger.info("Samples per class: %d", cfg.pipeline.samples_per_class)
    logger.info("Stance classes: %s", cfg.pipeline.stance_classes)
    logger.info("Annotation model: %s", cfg.model.annotation_model)
    logger.info("Validator model: %s", cfg.model.validator_model_name)
    logger.info("Budget limit: $%.2f", cfg.pipeline.total_budget_limit_usd)

    # ── 1. Setup output dirs and CSVs ────────────────────────────────
    os.makedirs(cfg.data.output_dir, exist_ok=True)
    os.makedirs(cfg.data.cache_dir, exist_ok=True)

    accepted_path = cfg.data.accepted_path()
    repaired_path = cfg.data.repaired_path()
    review_path = cfg.data.review_queue_path()
    checkpoint_path = os.path.join(cfg.data.cache_dir, "pipeline_checkpoint.pkl")

    _init_csv(accepted_path)
    _init_csv(repaired_path)
    _init_csv(review_path)

    # ── 2. Load data ONCE ────────────────────────────────────────────
    stance_df, gold_df = load_datasets(cfg.data)

    # ── 3. Load few-shot examples from gold ──────────────────────────
    few_shot_s1 = load_few_shot_from_gold(gold_df, cfg.pipeline.num_few_shot_examples)
    few_shot_s2 = load_few_shot_stage2_from_gold(gold_df, cfg.pipeline.num_few_shot_examples)

    # ── 4. Filter candidates ─────────────────────────────────────────
    candidates = load_cached_candidates(cfg)
    if candidates is None:
        candidates = run_filtering(cfg)
    logger.info("Total candidates: %d", len(candidates))

    # ── 5. Create OpenAI client ──────────────────────────────────────
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # ── 6. Resume from checkpoint if available ───────────────────────
    state = _load_checkpoint(checkpoint_path)
    if state:
        counts = state.get("counts", {})
        processed_ids = state.get("processed_ids", set())
        total_cost = state.get("total_cost", 0.0)
        logger.info("Resuming from checkpoint: %s, cost=$%.4f", counts, total_cost)
    else:
        counts = {s: 0 for s in cfg.pipeline.stance_classes}
        processed_ids = set()
        total_cost = 0.0

    # ── 7. Process candidates per stance class ───────────────────────
    sample_id = sum(counts.values())
    target_total = cfg.pipeline.samples_per_class * len(cfg.pipeline.stance_classes)

    logger.info("Target: %d samples (%d per class)", target_total, cfg.pipeline.samples_per_class)

    pbar = tqdm(total=target_total, initial=sample_id, desc="Generating ACOS-HD")

    # Iterate through candidates
    for idx, row in candidates.iterrows():
        # Check if all classes are satisfied
        if all(counts[s] >= cfg.pipeline.samples_per_class for s in cfg.pipeline.stance_classes):
            logger.info("All stance classes satisfied!")
            break

        # Skip already processed
        src_id = f"{row.get('dataset', '')}_{row.get('id', '')}_{idx}"
        if src_id in processed_ids:
            continue

        # Budget check
        if total_cost >= cfg.pipeline.total_budget_limit_usd:
            logger.warning("Budget limit reached: $%.2f >= $%.2f",
                           total_cost, cfg.pipeline.total_budget_limit_usd)
            break

        # Process sample
        result = process_sample(row, sample_id, cfg, few_shot_s1, few_shot_s2, client)
        sample_cost = float(result.get("cost_usd", 0))
        total_cost += sample_cost

        # Route based on status
        if result["status"] == "accepted":
            stance = result["stance"]
            if stance in counts and counts[stance] < cfg.pipeline.samples_per_class:
                # Dedup check
                if not is_duplicate_of_accepted(result["cleaned_text"]):
                    _append_csv(accepted_path, result)
                    register_accepted(result["cleaned_text"])
                    counts[stance] += 1
                    sample_id += 1
                    pbar.update(1)
                    logger.info(
                        "ACCEPTED #%d [%s] cost=$%.4f | Totals: %s | Total cost=$%.4f",
                        sample_id, stance, sample_cost, counts, total_cost
                    )
                else:
                    logger.debug("Duplicate detected, skipping")

        elif result["status"] == "repaired":
            stance = result["stance"]
            if stance in counts and counts[stance] < cfg.pipeline.samples_per_class:
                if not is_duplicate_of_accepted(result["cleaned_text"]):
                    _append_csv(repaired_path, result)
                    register_accepted(result["cleaned_text"])
                    counts[stance] += 1
                    sample_id += 1
                    pbar.update(1)
                    logger.info(
                        "REPAIRED #%d [%s] cost=$%.4f | Totals: %s",
                        sample_id, stance, sample_cost, counts
                    )

        elif result["status"] == "review":
            _append_csv(review_path, result)
            logger.info("QUEUED FOR REVIEW: %s", result.get("errors", ""))

        elif result["status"] == "filtered":
            pass  # Text was filtered during cleaning

        elif result["status"] == "failed":
            logger.warning("FAILED: %s", result.get("errors", ""))

        processed_ids.add(src_id)

        # Checkpoint
        if sample_id % cfg.pipeline.checkpoint_every_n == 0 and sample_id > 0:
            _save_checkpoint({
                "counts": counts,
                "processed_ids": processed_ids,
                "total_cost": total_cost,
            }, checkpoint_path)

    pbar.close()

    # ── 8. Final summary ─────────────────────────────────────────────
    s1_cost = s1_costs()
    s2_cost = s2_costs()
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Samples per class: %s", counts)
    logger.info("Total samples: %d", sum(counts.values()))
    logger.info("Stage 1 cost: $%.4f (%d tokens)", s1_cost["total_cost_usd"], s1_cost["total_input_tokens"] + s1_cost["total_output_tokens"])
    logger.info("Stage 2 cost: $%.4f (%d tokens)", s2_cost["total_cost_usd"], s2_cost["total_input_tokens"] + s2_cost["total_output_tokens"])
    logger.info("Total cost: $%.4f", total_cost)
    logger.info("Outputs: %s", cfg.data.output_dir)
    logger.info("  accepted.csv: %s", accepted_path)
    logger.info("  repaired.csv: %s", repaired_path)
    logger.info("  review_queue.csv: %s", review_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ACOS-HD Generation Pipeline")
    parser.add_argument("--samples-per-class", type=int, default=None,
                        help="Override samples_per_class config")
    parser.add_argument("--budget", type=float, default=None,
                        help="Override budget limit in USD")
    parser.add_argument("--log-level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    cfg = get_config()
    if args.samples_per_class is not None:
        cfg.pipeline.samples_per_class = args.samples_per_class
    if args.budget is not None:
        cfg.pipeline.total_budget_limit_usd = args.budget
    if args.log_level is not None:
        cfg.pipeline.log_level = args.log_level

    run_pipeline(cfg)
