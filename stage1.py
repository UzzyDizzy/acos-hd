"""
stage1.py - Stage 1 generation: structured extraction via GPT-4.1-mini.
Extracts (aspect_target, aspect_category, opinion_span) per paper §3.4 eq.8.
Uses async OpenAI calls with rate limiting and cost tracking.
"""
import asyncio, json, logging, time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from configs import ACOSHDConfig, OPENAI_API_KEY
from prompt_stage1 import build_stage1_prompt
from validate_stage1 import parse_stage1_output, validate_stage1, ValidationResult

logger = logging.getLogger(__name__)

# Cost tracking
_total_input_tokens = 0
_total_output_tokens = 0
_total_cost_usd = 0.0


def get_cost_summary() -> Dict[str, float]:
    return {
        "total_input_tokens": _total_input_tokens,
        "total_output_tokens": _total_output_tokens,
        "total_cost_usd": _total_cost_usd,
    }


def _compute_cost(input_tokens: int, output_tokens: int, cfg: ACOSHDConfig) -> float:
    """Compute cost in USD for a single API call."""
    inp_cost = (input_tokens / 1000) * cfg.model.annotation_input_price_per_1k
    out_cost = (output_tokens / 1000) * cfg.model.annotation_output_price_per_1k
    return inp_cost + out_cost


def _create_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_stage1(
    post_text: str,
    cfg: ACOSHDConfig,
    few_shot_examples: Optional[List[Dict]] = None,
    client: Optional[OpenAI] = None,
) -> Tuple[Optional[Dict], float, Dict]:
    """Run Stage 1 generation for a single post.

    Returns: (parsed_output, cost_usd, usage_info)
    """
    global _total_input_tokens, _total_output_tokens, _total_cost_usd

    if client is None:
        client = _create_client()

    messages = build_stage1_prompt(post_text, cfg.schema, few_shot_examples)

    # Retry with exponential backoff
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

            raw_content = response.choices[0].message.content or ""
            usage = response.usage
            input_tok = usage.prompt_tokens if usage else 0
            output_tok = usage.completion_tokens if usage else 0
            cost = _compute_cost(input_tok, output_tok, cfg)

            _total_input_tokens += input_tok
            _total_output_tokens += output_tok
            _total_cost_usd += cost

            usage_info = {
                "input_tokens": input_tok,
                "output_tokens": output_tok,
                "cost_usd": cost,
                "model": cfg.model.annotation_model,
            }

            # Parse
            parsed, parse_err = parse_stage1_output(raw_content)
            if parsed is None:
                logger.warning("Stage 1 parse failed: %s", parse_err)
                return None, cost, usage_info

            return parsed, cost, usage_info

        except Exception as e:
            wait = cfg.pipeline.api_retry_backoff_base ** attempt
            logger.warning("Stage 1 API error (attempt %d/%d): %s. Retrying in %.1fs",
                           attempt + 1, cfg.pipeline.api_max_retries, e, wait)
            time.sleep(wait)

    logger.error("Stage 1 generation failed after %d retries", cfg.pipeline.api_max_retries)
    return None, 0.0, {}


def generate_and_validate_stage1(
    post_text: str,
    cfg: ACOSHDConfig,
    few_shot_examples: Optional[List[Dict]] = None,
    client: Optional[OpenAI] = None,
) -> Tuple[Optional[Dict], ValidationResult, float, Dict]:
    """Generate Stage 1 output and validate it.

    Returns: (validated_output, validation_result, cost_usd, usage_info)
    """
    parsed, cost, usage = generate_stage1(post_text, cfg, few_shot_examples, client)
    if parsed is None:
        vr = ValidationResult(valid=False, errors=[], output=None)
        return None, vr, cost, usage

    vr = validate_stage1(parsed, post_text, cfg.schema)
    return vr.output, vr, cost, usage
