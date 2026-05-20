"""
prompt_stage1.py - Stage 1 prompt construction for ACOS-HD.
Builds the instruction-following prompt for structured extraction of
(aspect_target, aspect_category, opinion_span) per paper §3.2, §3.4.
"""
import json, random
from typing import Dict, List, Optional
import pandas as pd
from configs import SchemaConfig

# ── System prompt (paper §3.2: Instruction Construction) ──────────────────

STAGE1_SYSTEM = """You are an expert annotator for the ACOS-HD (Aspect-Category-Opinion-Stance) framework for homelessness discourse analysis.

Your task is Stage 1: Structured Extraction. Given a homelessness-related social media post, you must extract three discourse components.

TASK INSTRUCTION (per ACOS-HD §3.4):
Extract the following from the post:

1. **Aspect Target (at)**: The stance-bearing entity, policy, practice, place, group, or intervention being evaluated in the post. This is NOT just any entity mentioned — it is the specific target toward which the attitude is directed. It MUST be an exact substring copied from the input post.

2. **Aspect Category (c)**: Select EXACTLY ONE category from the inventory below. The category captures the homelessness-related discourse dimension.

3. **Opinion Span (o)**: The textual expression that conveys the evaluative attitude toward the aspect target. It MUST be an exact substring copied from the input post.

ASPECT CATEGORY INVENTORY (C):
{category_inventory}

SCHEMA CONSTRAINTS:
- aspect_target MUST be an exact substring of the input post (span-copy constraint)
- opinion_span MUST be an exact substring of the input post (span-copy constraint)
- aspect_category MUST be exactly one of the 8 categories listed above
- Output MUST be valid JSON matching the schema below
- Do NOT hallucinate or paraphrase spans — copy them exactly from the post

OUTPUT SCHEMA:
{{
  "aspect_target": "<exact span from post>",
  "aspect_category": "<one of 8 categories>",
  "opinion_span": "<exact span from post>"
}}"""

# ── Few-shot examples ─────────────────────────────────────────────────────

DEFAULT_FEW_SHOT_STAGE1 = [
    {
        "post": "The city should invest in shelters instead of pushing people out of parks.",
        "output": {
            "aspect_target": "shelters",
            "aspect_category": "Shelter & Housing",
            "opinion_span": "invest in shelters"
        }
    },
    {
        "post": "They are ruining downtown and should be kicked out immediately.",
        "output": {
            "aspect_target": "downtown",
            "aspect_category": "Public Space",
            "opinion_span": "ruining downtown and should be kicked out"
        }
    },
    {
        "post": "The report says shelter capacity is below demand this winter.",
        "output": {
            "aspect_target": "shelter capacity",
            "aspect_category": "Shelter & Housing",
            "opinion_span": "below demand"
        }
    },
]

# ── Repair prompt variant (stricter, paper §3.6) ─────────────────────────

STAGE1_REPAIR_SYSTEM = """You are an expert annotator for ACOS-HD. Your PREVIOUS extraction had errors.

You MUST fix the following issues:
{error_feedback}

STRICT CONSTRAINTS (you failed these before — follow them exactly):
- aspect_target MUST be an EXACT substring that appears verbatim in the post
- opinion_span MUST be an EXACT substring that appears verbatim in the post
- aspect_category MUST be EXACTLY one of: {category_list}
- Copy spans CHARACTER-FOR-CHARACTER from the post. Do NOT add, remove, or change any words.
- Output valid JSON only. No explanation text outside the JSON.

ASPECT CATEGORY INVENTORY:
{category_inventory}

OUTPUT SCHEMA:
{{
  "aspect_target": "<exact span from post>",
  "aspect_category": "<one of 8 categories>",
  "opinion_span": "<exact span from post>"
}}"""


def _format_category_inventory(schema: SchemaConfig) -> str:
    lines = []
    for i, cat in enumerate(schema.aspect_categories, 1):
        defn = schema.category_definitions.get(cat, "")
        lines.append(f"{i}. {cat}: {defn}")
    return "\n".join(lines)


def build_stage1_prompt(
    post_text: str,
    schema: SchemaConfig,
    few_shot_examples: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """Build the chat-format messages for Stage 1 generation.
    Returns list of {"role": ..., "content": ...} dicts for OpenAI API.
    """
    cat_inv = _format_category_inventory(schema)
    system_msg = STAGE1_SYSTEM.format(category_inventory=cat_inv)

    messages = [{"role": "system", "content": system_msg}]

    # Few-shot examples
    examples = few_shot_examples or DEFAULT_FEW_SHOT_STAGE1
    for ex in examples:
        messages.append({
            "role": "user",
            "content": f"POST: {ex['post']}"
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(ex["output"], indent=2)
        })

    # Actual input
    messages.append({"role": "user", "content": f"POST: {post_text}"})
    return messages


def build_stage1_repair_prompt(
    post_text: str,
    previous_output: Dict,
    errors: List[str],
    schema: SchemaConfig,
) -> List[Dict[str, str]]:
    """Build repair prompt for Stage 1 with explicit error feedback (paper §3.6)."""
    cat_inv = _format_category_inventory(schema)
    cat_list = ", ".join(schema.aspect_categories)
    error_feedback = "\n".join(f"- {e}" for e in errors)

    system_msg = STAGE1_REPAIR_SYSTEM.format(
        error_feedback=error_feedback,
        category_list=cat_list,
        category_inventory=cat_inv,
    )

    messages = [{"role": "system", "content": system_msg}]
    messages.append({
        "role": "user",
        "content": (
            f"POST: {post_text}\n\n"
            f"YOUR PREVIOUS (INCORRECT) OUTPUT:\n{json.dumps(previous_output, indent=2)}\n\n"
            f"Fix all errors and re-extract. Output valid JSON only."
        )
    })
    return messages


def load_few_shot_from_gold(
    gold_df: pd.DataFrame, n: int = 3, seed: int = 42
) -> List[Dict]:
    """Sample few-shot examples from gold_dataset.csv for Stage 1."""
    rng = random.Random(seed)
    examples = []
    # Try to get one per stance class
    for stance in ["Hate", "Neutral", "Hopeful"]:
        pool = gold_df[gold_df["stance"].str.strip().str.lower() == stance.lower()]
        if len(pool) > 0:
            row = pool.sample(1, random_state=seed).iloc[0]
            examples.append({
                "post": row["text"],
                "output": {
                    "aspect_target": row.get("aspect_span", ""),
                    "aspect_category": row.get("aspect_category", ""),
                    "opinion_span": row.get("opinion_span", ""),
                }
            })
    # Fill remaining if needed
    while len(examples) < n:
        row = gold_df.sample(1, random_state=seed + len(examples)).iloc[0]
        examples.append({
            "post": row["text"],
            "output": {
                "aspect_target": row.get("aspect_span", ""),
                "aspect_category": row.get("aspect_category", ""),
                "opinion_span": row.get("opinion_span", ""),
            }
        })
    return examples[:n]
