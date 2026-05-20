"""
repair_stage1.py - Repair/regeneration for Stage 1 using LLaMA 3.1 8B Instruct.
Handles: auto-fix for minor issues, LLM-assisted regeneration for major errors.
Implements the repair/regenerate path from paper §3.6 Figure 2.
"""
import json, logging
from typing import Dict, List, Optional, Tuple

from configs import ACOSHDConfig, HF_TOKEN
from prompt_stage1 import build_stage1_repair_prompt
from validate_stage1 import (
    ValidationResult, ValidationError, parse_stage1_output,
    validate_stage1, normalize_category, find_closest_span,
)

logger = logging.getLogger(__name__)

# ── LLaMA model singleton ────────────────────────────────────────────────
_llama_model = None
_llama_tokenizer = None


def load_llama_model(cfg: ACOSHDConfig):
    """Load LLaMA 3.1 8B Instruct with 4-bit quantization (once)."""
    global _llama_model, _llama_tokenizer
    if _llama_model is not None:
        return _llama_model, _llama_tokenizer

    logger.info("Loading LLaMA 3.1 8B Instruct (4-bit quantized)...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    _llama_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.validator_model_name,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    _llama_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.validator_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    logger.info("LLaMA model loaded successfully.")
    return _llama_model, _llama_tokenizer


def _generate_with_llama(
    messages: List[Dict[str, str]], cfg: ACOSHDConfig
) -> str:
    """Generate text using local LLaMA model."""
    model, tokenizer = load_llama_model(cfg)

    # Format as chat template
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
    else:
        # Fallback: manual formatting
        text = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        input_ids = tokenizer.encode(text, return_tensors="pt")

    import torch
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=cfg.model.validator_max_tokens,
            temperature=cfg.model.validator_temperature,
            top_p=cfg.model.validator_top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _format_errors(errors: List[ValidationError]) -> List[str]:
    """Format validation errors into human-readable strings for repair prompt."""
    msgs = []
    for e in errors:
        if e.error_type == "missing_field":
            msgs.append(f"Field '{e.field}' is missing or empty. You MUST provide it.")
        elif e.error_type == "invalid_category":
            msgs.append(f"Category '{e.message}' is not in the allowed list.")
        elif e.error_type == "unsupported_span":
            msgs.append(
                f"The {e.field} span is NOT found in the post text. "
                f"You MUST copy it exactly from the post."
            )
        elif e.error_type == "parse_error":
            msgs.append(f"Output could not be parsed as JSON. Output valid JSON only.")
        else:
            msgs.append(f"{e.error_type}: {e.message}")
    return msgs


def auto_repair_stage1(
    output: Dict, post_text: str, schema
) -> Tuple[Dict, bool]:
    """Attempt auto-repair without LLM. Returns (fixed_output, was_fixed)."""
    fixed = False

    # Fix category
    cat = output.get("aspect_category", "")
    if cat and cat not in schema.aspect_categories:
        norm = normalize_category(cat, schema)
        if norm:
            output["aspect_category"] = norm
            fixed = True

    # Fix aspect_target span
    at = output.get("aspect_target", "")
    if at and at.lower() not in post_text.lower():
        closest = find_closest_span(at, post_text)
        if closest:
            output["aspect_target"] = closest
            fixed = True

    # Fix opinion_span
    op = output.get("opinion_span", "")
    if op and op.lower() not in post_text.lower():
        closest = find_closest_span(op, post_text)
        if closest:
            output["opinion_span"] = closest
            fixed = True

    return output, fixed


def repair_stage1(
    post_text: str,
    previous_output: Dict,
    validation_result: ValidationResult,
    cfg: ACOSHDConfig,
) -> Tuple[Optional[Dict], ValidationResult, int]:
    """Repair Stage 1 output using auto-fix then LLaMA regeneration.

    Returns: (repaired_output, validation_result, num_attempts)
    """
    # Step 1: Try auto-repair first
    repaired, was_auto_fixed = auto_repair_stage1(
        previous_output.copy(), post_text, cfg.schema
    )
    if was_auto_fixed:
        vr = validate_stage1(repaired, post_text, cfg.schema)
        if vr.valid:
            logger.info("Stage 1 auto-repair succeeded")
            return vr.output, vr, 0

    # Step 2: LLM-assisted regeneration loop
    current_output = previous_output.copy()
    current_errors = validation_result.errors

    for attempt in range(cfg.pipeline.max_repair_retries):
        error_msgs = _format_errors(current_errors)
        messages = build_stage1_repair_prompt(
            post_text, current_output, error_msgs, cfg.schema
        )

        try:
            raw = _generate_with_llama(messages, cfg)
            parsed, parse_err = parse_stage1_output(raw)

            if parsed is None:
                logger.warning("Repair attempt %d: parse failed: %s", attempt+1, parse_err)
                current_errors = [ValidationError("parse_error", "output", parse_err or "")]
                continue

            # Auto-fix minor issues in regenerated output
            parsed, _ = auto_repair_stage1(parsed, post_text, cfg.schema)
            vr = validate_stage1(parsed, post_text, cfg.schema)

            if vr.valid:
                logger.info("Stage 1 repair succeeded on attempt %d", attempt+1)
                return vr.output, vr, attempt + 1

            current_output = parsed
            current_errors = vr.errors
            logger.info("Repair attempt %d: %d errors remain", attempt+1, len(vr.errors))

        except Exception as e:
            logger.error("Repair attempt %d failed: %s", attempt+1, e)
            current_errors = [ValidationError("repair_error", "output", str(e))]

    # All retries exhausted
    logger.warning("Stage 1 repair exhausted after %d attempts", cfg.pipeline.max_repair_retries)
    final_vr = validate_stage1(current_output, post_text, cfg.schema)
    return current_output, final_vr, cfg.pipeline.max_repair_retries
