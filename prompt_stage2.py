"""
prompt_stage2.py - Stage 2 prompt construction for ACOS-HD.
Builds the instruction-following prompt for stance prediction and
span-grounded rationale generation, conditioned on Stage 1 output (paper §3.5).
"""
import json
from typing import Dict, List, Optional
from configs import SchemaConfig

# ── System prompt (paper §3.5: Stance and Rationale Generation) ───────────

STAGE2_SYSTEM = """You are an expert annotator for the ACOS-HD framework for homelessness discourse analysis.

Your task is Stage 2: Stance and Rationale Generation. Given a post AND its extracted discourse structure from Stage 1 (aspect target, aspect category, opinion span), you must predict the stance label and provide span-grounded explanation evidence.

STAGE 1 OUTPUT (already extracted):
- Aspect Target: {aspect_target}
- Aspect Category: {aspect_category}
- Opinion Span: {opinion_span}

TASK INSTRUCTION (per ACOS-HD §3.5):

1. **Stance (y)**: Classify the post's attitude as EXACTLY one of three labels:
{stance_definitions}

2. **Explanation Evidence (e)**: The MINIMAL text span from the post that justifies the predicted stance. This MUST be an exact substring copied from the input post. It should contain the smallest piece of evidence needed to explain WHY the stance is what it is. Do NOT include unnecessary context — select the minimal justifying span.

SCHEMA CONSTRAINTS:
- stance MUST be exactly one of: HATE, NEUTRAL, HOPEFUL
- explanation MUST be an exact substring of the input post (span-copy constraint)
- explanation should be the MINIMAL evidence justifying the stance
- The explanation must be GROUNDED in the input — do not hallucinate or paraphrase
- Output MUST be valid JSON matching the schema below

OUTPUT SCHEMA:
{{
  "stance": "<HATE|NEUTRAL|HOPEFUL>",
  "explanation": "<exact minimal span from post>"
}}"""

# ── Few-shot examples ─────────────────────────────────────────────────────

DEFAULT_FEW_SHOT_STAGE2 = [
    {
        "post": "The city should invest in shelters instead of pushing people out of parks.",
        "stage1": {"aspect_target": "shelters", "aspect_category": "Shelter & Housing",
                   "opinion_span": "invest in shelters"},
        "output": {"stance": "HOPEFUL",
                   "explanation": "invest in shelters instead of pushing people out"}
    },
    {
        "post": "They are ruining downtown and should be kicked out immediately.",
        "stage1": {"aspect_target": "downtown", "aspect_category": "Public Space",
                   "opinion_span": "ruining downtown and should be kicked out"},
        "output": {"stance": "HATE",
                   "explanation": "ruining downtown and should be kicked out immediately"}
    },
    {
        "post": "The report says shelter capacity is below demand this winter.",
        "stage1": {"aspect_target": "shelter capacity", "aspect_category": "Shelter & Housing",
                   "opinion_span": "below demand"},
        "output": {"stance": "NEUTRAL",
                   "explanation": "shelter capacity is below demand this winter"}
    },
]

# ── Repair prompt variant (paper §3.6) ────────────────────────────────────

STAGE2_REPAIR_SYSTEM = """You are an expert ACOS-HD annotator. Your PREVIOUS stance/explanation output had errors.

STAGE 1 CONTEXT:
- Aspect Target: {aspect_target}
- Aspect Category: {aspect_category}
- Opinion Span: {opinion_span}

ERRORS TO FIX:
{error_feedback}

STRICT CONSTRAINTS (you failed these before — follow exactly):
- stance MUST be EXACTLY one of: HATE, NEUTRAL, HOPEFUL (no other labels)
- explanation MUST be an EXACT substring that appears VERBATIM in the post
- Copy the explanation span CHARACTER-FOR-CHARACTER from the post
- Select the MINIMAL span that justifies the stance
- Output valid JSON only. No text outside the JSON.

STANCE DEFINITIONS:
{stance_definitions}

OUTPUT SCHEMA:
{{
  "stance": "<HATE|NEUTRAL|HOPEFUL>",
  "explanation": "<exact minimal span from post>"
}}"""


def _format_stance_definitions(schema: SchemaConfig) -> str:
    lines = []
    for label in schema.stance_labels:
        defn = schema.stance_definitions.get(label, "")
        lines.append(f"   - {label}: {defn}")
    return "\n".join(lines)


def build_stage2_prompt(
    post_text: str,
    stage1_output: Dict[str, str],
    schema: SchemaConfig,
    few_shot_examples: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """Build chat messages for Stage 2 generation (paper §3.5, eq. 9)."""
    stance_defs = _format_stance_definitions(schema)
    system_msg = STAGE2_SYSTEM.format(
        aspect_target=stage1_output.get("aspect_target", ""),
        aspect_category=stage1_output.get("aspect_category", ""),
        opinion_span=stage1_output.get("opinion_span", ""),
        stance_definitions=stance_defs,
    )

    messages = [{"role": "system", "content": system_msg}]

    # Few-shot
    examples = few_shot_examples or DEFAULT_FEW_SHOT_STAGE2
    for ex in examples:
        user_msg = (
            f"POST: {ex['post']}\n\n"
            f"STAGE 1 OUTPUT:\n"
            f"- Aspect Target: {ex['stage1']['aspect_target']}\n"
            f"- Aspect Category: {ex['stage1']['aspect_category']}\n"
            f"- Opinion Span: {ex['stage1']['opinion_span']}"
        )
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": json.dumps(ex["output"], indent=2)})

    # Actual input
    user_msg = (
        f"POST: {post_text}\n\n"
        f"STAGE 1 OUTPUT:\n"
        f"- Aspect Target: {stage1_output.get('aspect_target', '')}\n"
        f"- Aspect Category: {stage1_output.get('aspect_category', '')}\n"
        f"- Opinion Span: {stage1_output.get('opinion_span', '')}"
    )
    messages.append({"role": "user", "content": user_msg})
    return messages


def build_stage2_repair_prompt(
    post_text: str,
    stage1_output: Dict[str, str],
    previous_output: Dict,
    errors: List[str],
    schema: SchemaConfig,
) -> List[Dict[str, str]]:
    """Build repair prompt for Stage 2 with explicit error feedback."""
    stance_defs = _format_stance_definitions(schema)
    error_feedback = "\n".join(f"- {e}" for e in errors)

    system_msg = STAGE2_REPAIR_SYSTEM.format(
        aspect_target=stage1_output.get("aspect_target", ""),
        aspect_category=stage1_output.get("aspect_category", ""),
        opinion_span=stage1_output.get("opinion_span", ""),
        error_feedback=error_feedback,
        stance_definitions=stance_defs,
    )

    messages = [{"role": "system", "content": system_msg}]
    messages.append({
        "role": "user",
        "content": (
            f"POST: {post_text}\n\n"
            f"YOUR PREVIOUS (INCORRECT) OUTPUT:\n{json.dumps(previous_output, indent=2)}\n\n"
            f"Fix all errors and regenerate. Output valid JSON only."
        )
    })
    return messages


def load_few_shot_stage2_from_gold(gold_df, n=3, seed=42):
    """Sample few-shot examples from gold dataset for Stage 2."""
    import random
    examples = []
    for stance in ["Hate", "Neutral", "Hopeful"]:
        pool = gold_df[gold_df["stance"].str.strip().str.lower() == stance.lower()]
        if len(pool) > 0:
            row = pool.sample(1, random_state=seed).iloc[0]
            examples.append({
                "post": row["text"],
                "stage1": {
                    "aspect_target": row.get("aspect_span", ""),
                    "aspect_category": row.get("aspect_category", ""),
                    "opinion_span": row.get("opinion_span", ""),
                },
                "output": {
                    "stance": row.get("stance", "").upper().strip(),
                    "explanation": row.get("evidence_span", ""),
                }
            })
    while len(examples) < n:
        row = gold_df.sample(1, random_state=seed + len(examples)).iloc[0]
        examples.append({
            "post": row["text"],
            "stage1": {
                "aspect_target": row.get("aspect_span", ""),
                "aspect_category": row.get("aspect_category", ""),
                "opinion_span": row.get("opinion_span", ""),
            },
            "output": {
                "stance": row.get("stance", "").upper().strip(),
                "explanation": row.get("evidence_span", ""),
            }
        })
    return examples[:n]
