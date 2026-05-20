"""
validate_stage2.py - Parsing and validation for Stage 2 outputs.
Implements checks (i), (iii), (v) from paper §3.6 for stance + rationale outputs.
Also performs label-consistency filtering between Stage 1 and Stage 2.
"""
import json, logging, re
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz
from configs import SchemaConfig
from validate_stage1 import (
    ValidationError, ValidationResult, span_grounded,
    find_closest_span, _token_overlap_f1,
)

logger = logging.getLogger(__name__)
REQUIRED_FIELDS_S2 = ["stance", "explanation"]


def parse_stage2_output(raw: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse raw LLM output into dict. Returns (parsed_dict, error_msg)."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()), None
        except json.JSONDecodeError:
            pass
    return None, f"Could not parse JSON: {raw[:200]}"


def normalize_stance(raw_stance: str, schema: SchemaConfig) -> Optional[str]:
    """Normalise stance label using the map from SchemaConfig (paper §3.6)."""
    if not raw_stance:
        return None
    s = raw_stance.strip()
    # Direct match (case-sensitive)
    if s in schema.stance_labels:
        return s
    # Upper-case match
    if s.upper() in schema.stance_labels:
        return s.upper()
    # Normalisation map
    key = s.lower().strip()
    if key in schema.stance_normalisation_map:
        return schema.stance_normalisation_map[key]
    # Also check the raw value in the map
    if s in schema.stance_normalisation_map:
        return schema.stance_normalisation_map[s]
    return None


def check_label_consistency(
    stage1: Dict, stage2: Dict, post_text: str
) -> Tuple[bool, str]:
    """Check if stance is consistent with opinion span valence.
    Returns (is_consistent, reason).
    """
    stance = stage2.get("stance", "").upper()
    opinion = stage1.get("opinion_span", "").lower()
    category = stage1.get("aspect_category", "")

    # Heuristic consistency checks
    hostile_markers = ["kicked out", "ruining", "disgusting", "get rid", "remove",
                       "forced out", "destroy", "trash", "filth", "vermin", "scum"]
    hopeful_markers = ["invest", "support", "help", "expand", "improve", "should be",
                       "need more", "deserve", "dignity", "compassion", "solution"]

    if stance == "HATE":
        if any(m in opinion for m in hopeful_markers) and not any(m in opinion for m in hostile_markers):
            return False, "Stance is HATE but opinion span contains hopeful language"
    elif stance == "HOPEFUL":
        if any(m in opinion for m in hostile_markers) and not any(m in opinion for m in hopeful_markers):
            return False, "Stance is HOPEFUL but opinion span contains hostile language"

    return True, ""


def rationale_grounding_score(explanation: str, post_text: str) -> float:
    """Compute rationale grounding score (paper §3.6, eq. for rationale F1).
    Returns token-level F1 between explanation and post text.
    """
    return _token_overlap_f1(explanation, post_text)


def validate_stage2(
    stage1_output: Dict, stage2_output: Dict, post_text: str, schema: SchemaConfig
) -> ValidationResult:
    """Full validation for Stage 2 output (paper §3.6 checks i, iii, v)."""
    errors = []
    auto_fixed = False

    # (i) All required fields present
    for f in REQUIRED_FIELDS_S2:
        if f not in stage2_output or not stage2_output[f] or not str(stage2_output[f]).strip():
            errors.append(ValidationError("missing_field", f, f"Field '{f}' is missing or empty"))

    if errors:
        return ValidationResult(valid=False, errors=errors, output=stage2_output)

    # (iii) Stance belongs to allowed label set
    raw_stance = str(stage2_output.get("stance", "")).strip()
    if raw_stance.upper() not in schema.stance_labels:
        normalized = normalize_stance(raw_stance, schema)
        if normalized:
            stage2_output["stance"] = normalized
            auto_fixed = True
            logger.debug("Auto-normalised stance: '%s' → '%s'", raw_stance, normalized)
        else:
            errors.append(ValidationError(
                "invalid_label", "stance",
                f"'{raw_stance}' is not valid. Must be one of: {schema.stance_labels}"
            ))
    else:
        stage2_output["stance"] = raw_stance.upper()

    # (v) Explanation evidence textually grounded in input
    explanation = str(stage2_output.get("explanation", "")).strip()
    if not span_grounded(explanation, post_text, schema.rationale_grounding_threshold):
        closest = find_closest_span(explanation, post_text)
        if closest:
            stage2_output["explanation"] = closest
            auto_fixed = True
            logger.debug("Auto-fixed explanation: '%s' → '%s'", explanation, closest)
        else:
            grounding_score = rationale_grounding_score(explanation, post_text)
            errors.append(ValidationError(
                "ungrounded_rationale", "explanation",
                f"Explanation not grounded in post (grounding_F1={grounding_score:.2f})"
            ))

    # Label-consistency check
    is_consistent, reason = check_label_consistency(stage1_output, stage2_output, post_text)
    if not is_consistent:
        errors.append(ValidationError("label_inconsistency", "stance", reason))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        output=stage2_output,
        auto_fixed=auto_fixed,
    )
