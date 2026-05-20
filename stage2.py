"""
stage2.py - Stage 2 generation: stance + rationale via GPT-4.1-mini.
Predicts stance label and span-grounded explanation conditioned on Stage 1 output (paper §3.5 eq.9).
"""
import json, logging, time
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from configs import ACOSHDConfig, OPENAI_API_KEY
from prompt_stage2 import build_stage2_prompt
from validate_stage2 import parse_stage2_output, validate_stage2, ValidationResult

logger = logging.getLogger(__name__)

_total_input_tokens = 0
_total_output_tokens = 0
_total_cost_usd = 0.0


def get_cost_summary() -> Dict[str, float]:
    return {"total_input_tokens": _total_input_tokens,
            "total_output_tokens": _total_output_tokens,
            "total_cost_usd": _total_cost_usd}


def _compute_cost(inp: int, out: int, cfg: ACOSHDConfig) -> float:
    return (inp / 1000) * cfg.model.annotation_input_price_per_1k + \
           (out / 1000) * cfg.model.annotation_output_price_per_1k


def _create_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_stage2(
    post_text: str, stage1_output: Dict, cfg: ACOSHDConfig,
    few_shot_examples: Optional[List[Dict]] = None,
    client: Optional[OpenAI] = None,
) -> Tuple[Optional[Dict], float, Dict]:
    """Run Stage 2 generation. Returns (parsed_output, cost_usd, usage_info)."""
    global _total_input_tokens, _total_output_tokens, _total_cost_usd
    if client is None:
        client = _create_client()

    messages = build_stage2_prompt(post_text, stage1_output, cfg.schema, few_shot_examples)

    for attempt in range(cfg.pipeline.api_max_retries):
        try:
            response = client.chat.completions.create(
                model=cfg.model.annotation_model,
                messages=messages,
                temperature=cfg.model.annotation_temperature,
                max_tokens=cfg.model.annotation_max_tokens,
                top_p=cfg.model.annotation_top_p,
                seed=cfg.model.annotation_seed,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            usage = response.usage
            inp_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0
            cost = _compute_cost(inp_tok, out_tok, cfg)
            _total_input_tokens += inp_tok
            _total_output_tokens += out_tok
            _total_cost_usd += cost

            usage_info = {"input_tokens": inp_tok, "output_tokens": out_tok,
                          "cost_usd": cost, "model": cfg.model.annotation_model}

            parsed, err = parse_stage2_output(raw)
            if parsed is None:
                logger.warning("Stage 2 parse failed: %s", err)
                return None, cost, usage_info
            return parsed, cost, usage_info

        except Exception as e:
            wait = cfg.pipeline.api_retry_backoff_base ** attempt
            logger.warning("Stage 2 API error (attempt %d/%d): %s. Retry in %.1fs",
                           attempt+1, cfg.pipeline.api_max_retries, e, wait)
            time.sleep(wait)

    logger.error("Stage 2 failed after %d retries", cfg.pipeline.api_max_retries)
    return None, 0.0, {}


def generate_and_validate_stage2(
    post_text: str, stage1_output: Dict, cfg: ACOSHDConfig,
    few_shot_examples: Optional[List[Dict]] = None,
    client: Optional[OpenAI] = None,
) -> Tuple[Optional[Dict], ValidationResult, float, Dict]:
    """Generate + validate Stage 2. Returns (output, validation_result, cost, usage)."""
    parsed, cost, usage = generate_stage2(post_text, stage1_output, cfg,
                                          few_shot_examples, client)
    if parsed is None:
        vr = ValidationResult(valid=False, errors=[], output=None)
        return None, vr, cost, usage
    vr = validate_stage2(stage1_output, parsed, post_text, cfg.schema)
    return vr.output, vr, cost, usage
