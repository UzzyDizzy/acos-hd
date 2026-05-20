"""
validate_stage1.py - Parsing and validation for Stage 1 outputs.
Implements checks (i)-(iv) from paper §3.6 for structured extraction outputs.
"""
import json, logging, re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz
from configs import SchemaConfig

logger = logging.getLogger(__name__)

REQUIRED_FIELDS_S1 = ["aspect_target", "aspect_category", "opinion_span"]


@dataclass
class ValidationError:
    error_type: str   # missing_field, invalid_category, unsupported_span, parse_error
    field: str
    message: str = ""


@dataclass
class ValidationResult:
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    output: Optional[Dict] = None
    auto_fixed: bool = False  # True if any field was auto-normalised


def parse_stage1_output(raw: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse raw LLM output into a dict. Returns (parsed_dict, error_msg)."""
    raw = raw.strip()
    # Try to extract JSON from markdown code blocks
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    # Try direct JSON parse
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, None
    except json.JSONDecodeError:
        pass
    # Try to find first { ... } block
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()), None
        except json.JSONDecodeError:
            pass
    return None, f"Could not parse JSON from output: {raw[:200]}"


def _token_overlap_f1(span: str, text: str) -> float:
    """Compute token-level F1 between span tokens and text tokens."""
    span_toks = set(span.lower().split())
    text_toks = set(text.lower().split())
    if not span_toks:
        return 0.0
    common = span_toks & text_toks
    if not common:
        return 0.0
    precision = len(common) / len(span_toks)
    recall = len(common) / len(text_toks) if text_toks else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def span_grounded(span: str, text: str, threshold: float = 0.6) -> bool:
    """Check if span is grounded in text (paper §3.6 span verification).
    (a) exact substring match (case-insensitive), OR
    (b) token-level overlap F1 >= threshold
    """
    if not span or not text:
        return False
    span_l, text_l = span.lower().strip(), text.lower().strip()
    # (a) Exact substring
    if span_l in text_l:
        return True
    # (b) Token overlap F1
    if _token_overlap_f1(span, text) >= threshold:
        return True
    # (c) Fuzzy partial ratio as fallback
    if fuzz.partial_ratio(span_l, text_l) >= 85:
        return True
    return False


def normalize_category(raw_cat: str, schema: SchemaConfig) -> Optional[str]:
    """Normalise aspect category using alias map + fuzzy matching (paper §3.6)."""
    if not raw_cat:
        return None
    raw_stripped = raw_cat.strip()
    # Direct match
    if raw_stripped in schema.aspect_categories:
        return raw_stripped
    # Alias map
    if raw_stripped in schema.category_alias_map:
        return schema.category_alias_map[raw_stripped]
    # Case-insensitive match
    for cat in schema.aspect_categories:
        if raw_stripped.lower() == cat.lower():
            return cat
    # Fuzzy match (unambiguous only)
    best_score, best_cat = 0, None
    for cat in schema.aspect_categories:
        score = fuzz.ratio(raw_stripped.lower(), cat.lower())
        if score > best_score:
            best_score = score
            best_cat = cat
    if best_score >= 75:
        return best_cat
    return None


def find_closest_span(span: str, text: str) -> Optional[str]:
    """Find the closest matching substring in text for a given span."""
    if not span or not text:
        return None
    span_l = span.lower().strip()
    text_l = text.lower()
    # Try sliding window
    span_words = span_l.split()
    text_words = text.split()
    n = len(span_words)
    best_score, best_match = 0, None
    for i in range(len(text_words) - n + 1):
        candidate = " ".join(text_words[i:i+n])
        score = fuzz.ratio(span_l, candidate.lower())
        if score > best_score:
            best_score = score
            best_match = candidate
    # Also try +/- 1 word windows
    for delta in [-1, 1, -2, 2]:
        nn = n + delta
        if nn < 1:
            continue
        for i in range(len(text_words) - nn + 1):
            candidate = " ".join(text_words[i:i+nn])
            score = fuzz.ratio(span_l, candidate.lower())
            if score > best_score:
                best_score = score
                best_match = candidate
    if best_score >= 70:
        return best_match
    return None


def validate_stage1(
    output: Dict, post_text: str, schema: SchemaConfig
) -> ValidationResult:
    """Full validation for Stage 1 output (paper §3.6 checks i, ii, iv)."""
    errors = []
    auto_fixed = False

    # (i) All required fields present
    for f in REQUIRED_FIELDS_S1:
        if f not in output or not output[f] or not str(output[f]).strip():
            errors.append(ValidationError("missing_field", f, f"Field '{f}' is missing or empty"))

    if errors:
        return ValidationResult(valid=False, errors=errors, output=output)

    # (ii) Aspect category belongs to C
    raw_cat = str(output.get("aspect_category", "")).strip()
    if raw_cat not in schema.aspect_categories:
        normalized = normalize_category(raw_cat, schema)
        if normalized:
            output["aspect_category"] = normalized
            auto_fixed = True
            logger.debug("Auto-normalised category: '%s' → '%s'", raw_cat, normalized)
        else:
            errors.append(ValidationError(
                "invalid_category", "aspect_category",
                f"'{raw_cat}' is not a valid category. Allowed: {schema.aspect_categories}"
            ))

    # (iv) Aspect target span grounded in post
    at = str(output.get("aspect_target", "")).strip()
    if not span_grounded(at, post_text, schema.span_overlap_threshold):
        # Try auto-fix: find closest span
        closest = find_closest_span(at, post_text)
        if closest:
            output["aspect_target"] = closest
            auto_fixed = True
            logger.debug("Auto-fixed aspect_target: '%s' → '%s'", at, closest)
        else:
            errors.append(ValidationError(
                "unsupported_span", "aspect_target",
                f"'{at}' not found in post text"
            ))

    # (iv) Opinion span grounded in post
    op = str(output.get("opinion_span", "")).strip()
    if not span_grounded(op, post_text, schema.span_overlap_threshold):
        closest = find_closest_span(op, post_text)
        if closest:
            output["opinion_span"] = closest
            auto_fixed = True
            logger.debug("Auto-fixed opinion_span: '%s' → '%s'", op, closest)
        else:
            errors.append(ValidationError(
                "unsupported_span", "opinion_span",
                f"'{op}' not found in post text"
            ))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        output=output,
        auto_fixed=auto_fixed,
    )
