"""
repair_stage2.py - Repair/regeneration for Stage 2 using LLaMA 3.1 8B Instruct.
Handles: stance label normalization, rationale grounding fixes, LLM regeneration.
Implements the repair/regenerate path from paper §3.6.
"""
import json, logging
from typing import Dict, List, Optional, Tuple

from configs import ACOSHDConfig
from prompt_stage2 import build_stage2_repair_prompt
from validate_stage2 import (
    ValidationResult, ValidationError, parse_stage2_output,
    validate_stage2, normalize_stance,
)
from validate_stage1 import find_closest_span
from repair_stage1 import load_llama_model, _generate_with_llama

logger = logging.getLogger(__name__)


def _format_errors(errors: List[ValidationError]) -> List[str]:
    """Format validation errors for repair prompt."""
    msgs = []
    for e in errors:
        if e.error_type == "missing_field":
            msgs.append(f"Field '{e.field}' is missing or empty.")
        elif e.error_type == "invalid_label":
            msgs.append(f"Stance '{e.message}' is invalid. Must be HATE, NEUTRAL, or HOPEFUL.")
        elif e.error_type == "ungrounded_rationale":
            msgs.append(
                f"The explanation is NOT grounded in the post text. "
                f"Copy the explanation EXACTLY from the post. {e.message}"
            )
        elif e.error_type == "label_inconsistency":
            msgs.append(f"Label inconsistency: {e.message}")
        elif e.error_type == "parse_error":
            msgs.append("Output could not be parsed as JSON.")
        else:
            msgs.append(f"{e.error_type}: {e.message}")
    return msgs


def auto_repair_stage2(
    output: Dict, post_text: str, schema
) -> Tuple[Dict, bool]:
    """Attempt auto-repair without LLM. Returns (fixed_output, was_fixed)."""
    fixed = False

    # Fix stance label
    stance = output.get("stance", "")
    if stance:
        norm = normalize_stance(stance, schema)
        if norm and norm != stance:
            output["stance"] = norm
            fixed = True

    # Fix explanation span
    expl = output.get("explanation", "")
    if expl and expl.lower() not in post_text.lower():
        closest = find_closest_span(expl, post_text)
        if closest:
            output["explanation"] = closest
            fixed = True

    return output, fixed


def repair_stage2(
    post_text: str,
    stage1_output: Dict,
    previous_output: Dict,
    validation_result: ValidationResult,
    cfg: ACOSHDConfig,
) -> Tuple[Optional[Dict], ValidationResult, int]:
    """Repair Stage 2 output using auto-fix then LLaMA regeneration.

    Returns: (repaired_output, validation_result, num_attempts)
    """
    # Step 1: Try auto-repair
    repaired, was_auto_fixed = auto_repair_stage2(
        previous_output.copy(), post_text, cfg.schema
    )
    if was_auto_fixed:
        vr = validate_stage2(stage1_output, repaired, post_text, cfg.schema)
        if vr.valid:
            logger.info("Stage 2 auto-repair succeeded")
            return vr.output, vr, 0

    # Step 2: LLM-assisted regeneration loop
    current_output = previous_output.copy()
    current_errors = validation_result.errors

    for attempt in range(cfg.pipeline.max_repair_retries):
        error_msgs = _format_errors(current_errors)
        messages = build_stage2_repair_prompt(
            post_text, stage1_output, current_output, error_msgs, cfg.schema
        )

        try:
            raw = _generate_with_llama(messages, cfg)
            parsed, parse_err = parse_stage2_output(raw)

            if parsed is None:
                logger.warning("Stage 2 repair attempt %d: parse failed", attempt+1)
                from validate_stage1 import ValidationError as VE
                current_errors = [VE("parse_error", "output", parse_err or "")]
                continue

            # Auto-fix minor issues
            parsed, _ = auto_repair_stage2(parsed, post_text, cfg.schema)
            vr = validate_stage2(stage1_output, parsed, post_text, cfg.schema)

            if vr.valid:
                logger.info("Stage 2 repair succeeded on attempt %d", attempt+1)
                return vr.output, vr, attempt + 1

            current_output = parsed
            current_errors = vr.errors
            logger.info("Stage 2 repair attempt %d: %d errors remain",
                        attempt+1, len(vr.errors))

        except Exception as e:
            logger.error("Stage 2 repair attempt %d failed: %s", attempt+1, e)
            from validate_stage1 import ValidationError as VE
            current_errors = [VE("repair_error", "output", str(e))]

    # All retries exhausted
    logger.warning("Stage 2 repair exhausted after %d attempts",
                   cfg.pipeline.max_repair_retries)
    final_vr = validate_stage2(stage1_output, current_output, post_text, cfg.schema)
    return current_output, final_vr, cfg.pipeline.max_repair_retries
