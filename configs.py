"""
configs.py — Central configuration for the ACOS-HD generation pipeline.

All hyperparameters, paths, model settings, schema definitions, and filtering
controls are exposed here so that the user can tune every aspect of the
pipeline from a single file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env.local
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parent / ".env.local"
load_dotenv(_ENV_PATH)

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
TWITTER_API_KEY: str = os.getenv("TWITTER_API", "")

BASE_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA PATHS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class DataConfig:
    """Paths to input / output artefacts."""

    stance_dataset_path: str = str(BASE_DIR / "stance_detection_dataset.csv")
    gold_dataset_path: str = str(BASE_DIR / "gold_dataset.csv")

    # --- output -----------------------------------------------------------
    output_dir: str = str(BASE_DIR / "output")
    accepted_csv: str = "accepted.csv"
    repaired_csv: str = "repaired.csv"
    reviewed_csv: str = "reviewed.csv"
    review_queue_csv: str = "review_queue.csv"

    # --- caching ----------------------------------------------------------
    cache_dir: str = str(BASE_DIR / "cache")

    def accepted_path(self) -> str:
        return os.path.join(self.output_dir, self.accepted_csv)

    def repaired_path(self) -> str:
        return os.path.join(self.output_dir, self.repaired_csv)

    def reviewed_path(self) -> str:
        return os.path.join(self.output_dir, self.reviewed_csv)

    def review_queue_path(self) -> str:
        return os.path.join(self.output_dir, self.review_queue_csv)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FILTERING / RELEVANCE
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class FilterConfig:
    """Controls for keyword, semantic, and deduplication filtering."""

    # --- primary keywords (homelessness-focused, per paper §4.1) ----------
    primary_keywords: List[str] = field(default_factory=lambda: [
        "homeless", "homelessness", "unhoused",
        "shelter", "housing", "encampment",
        "public safety", "addiction", "policy",
        "support", "dignity",
        # lexical variants from paper Table 11
        "hobo", "vagrant", "transient", "beggar",
        "people experiencing homelessness",

    "street",
    "rent",
    "eviction",
    "affordable housing",
    "housing crisis",
    "housing insecurity",
    "public housing",
    "tent",
    "addiction",
    "mental health",
    "poverty",
    "welfare",
    "support services",
    "transitional housing",
    "displacement"
    ])

    # --- secondary keywords (issue-specific) ------------------------------
    secondary_keywords: List[str] = field(default_factory=lambda: [
        "welfare", "poverty", "social services",
        "mental health", "drug", "rehab",
        "park", "sidewalk", "downtown",
        "tent", "sleeping rough", "panhandling",
        "food bank", "soup kitchen",
    ])

    # --- immigration-adjacent (for VAST sampling) -------------------------
    immigration_keywords: List[str] = field(default_factory=lambda: [
        "immigrant", "immigration", "refugee",
        "asylum", "undocumented", "migrant",
        "deport", "border", "alien",
        "illegal labor",
    ])

    # Minimum keyword hits in a post to pass keyword filter
    min_keyword_hits: int = 1

    # --- semantic-similarity filtering ------------------------------------
    use_semantic_similarity: bool = False
    similarity_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.35

    # Homelessness reference sentences for semantic filter
    reference_sentences: List[str] = field(default_factory=lambda: [
        "People experiencing homelessness need more shelters and housing.",
        "The city should invest in programs for the unhoused population.",
        "Encampments are a growing public safety concern downtown.",
        "Addiction and mental health services for homeless individuals.",
        "Government policy on homelessness and housing-first programs.",
        "Employment opportunities and economic support for people without homes.",
        "Community impact of homelessness on neighborhoods and public spaces.",
        "Empathy and dignity for vulnerable populations on the streets.",
    ])

    # --- dataset sampling -------------------------------------------------
    datasets_to_sample: List[str] = field(
        default_factory=lambda: ["VAST", "PStance", "COVID19", "SemEval2016"]
    )
    vast_priority_weight: float = 2.0  # over-sample VAST

    # --- deduplication ----------------------------------------------------
    dedup_fuzzy_threshold: float = 1  # Jaccard / MinHash threshold
    dedup_num_perm: int = 128  # MinHash permutations


# ═══════════════════════════════════════════════════════════════════════════
# 3.  TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class CleaningConfig:
    """Controls for the multi-step text cleaning pipeline."""

    min_text_length: int = 20  # characters after cleaning
    max_text_length: int = 512

    # URL handling
    resolve_urls: bool = True
    url_importance_keywords: List[str] = field(default_factory=lambda: [
        "homeless", "shelter", "housing", "encampment", "policy",
        "addiction", "mental health", "welfare",
    ])
    url_request_timeout: int = 5  # seconds

    # Hashtag / mention handling
    resolve_hashtags: bool = True
    resolve_mentions: bool = True  # now enabled since user has TWITTER_API
    twitter_api_base_url: str = "https://api.twitterapi.io"

    # Cleaning steps (each can be toggled)
    fix_unicode: bool = True
    expand_contractions: bool = True
    expand_abbreviations: bool = True
    expand_slang: bool = True
    substitute_emojis: bool = True
    fix_spelling: bool = True
    remove_repeated_chars: bool = True
    remove_repeated_words: bool = True
    remove_repeated_sentences: bool = True
    remove_numbers: bool = True
    remove_garbage_syntax: bool = True
    normalize_offensive: bool = True

    # Ekphrasis settings
    ekphrasis_segmenter_corpus: str = "twitter"
    ekphrasis_corrector_corpus: str = "twitter"


# ═══════════════════════════════════════════════════════════════════════════
# 4.  MODELS
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    """LLM configurations for annotation and validation/repair."""

    # --- Annotation model (GPT-4.1-mini via OpenAI API) -------------------
    annotation_model: str = "gpt-4.1-mini"
    annotation_temperature: float = 0.3
    annotation_max_tokens: int = 512
    annotation_top_p: float = 0.95
    annotation_seed: Optional[int] = 42

    # pricing per 1K tokens (USD) — GPT-4.1-mini
    annotation_input_price_per_1k: float = 0.0004
    annotation_output_price_per_1k: float = 0.0016

    # --- Validation / repair model (LLaMA 3.1 8B Instruct, local) --------
    validator_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    validator_quantization: str = "4bit"  # BitsAndBytes NF4
    validator_max_tokens: int = 512
    validator_temperature: float = 0.1
    validator_top_p: float = 0.9
    validator_device: str = "cuda"
    validator_torch_dtype: str = "float16"

    # LoRA settings (for future fine-tuning, per paper §4.4)
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# 5.  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class PipelineConfig:
    """Top-level pipeline control knobs."""

    # How many samples to generate per stance class
    samples_per_class: int = 1000  # set to 1000 for production run

    stance_classes: List[str] = field(
        default_factory=lambda: ["HATE", "NEUTRAL", "HOPEFUL"]
    )

    # Validation / repair loop limits
    max_validation_retries: int = 3
    max_repair_retries: int = 3

    # Concurrency
    api_batch_size: int = 5  # concurrent OpenAI calls
    api_rate_limit_rpm: int = 500  # requests per minute
    api_retry_backoff_base: float = 2.0
    api_max_retries: int = 5

    # Cost guardrails
    log_cost_per_sample: bool = True
    total_budget_limit_usd: float = 50.0  # hard cap

    # Checkpointing
    checkpoint_every_n: int = 25  # save state every N accepted samples

    # Logging
    log_level: str = "INFO"
    log_file: str = "pipeline.log"

    # Few-shot examples for prompts (drawn from gold dataset)
    num_few_shot_examples: int = 3


# ═══════════════════════════════════════════════════════════════════════════
# 6.  SCHEMA (ACOS-HD)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SchemaConfig:
    """ACOS-HD output schema and label inventories (paper §3.1)."""

    aspect_categories: List[str] = field(default_factory=lambda: [
        "Shelter & Housing",
        "Public Space",
        "Public Safety",
        "Addiction & Health",
        "Policy & Governance",
        "Employment & Economy",
        "Empathy & Support",
        "Community Impact & Social Cohesion",
    ])

    stance_labels: List[str] = field(
        default_factory=lambda: ["HATE", "NEUTRAL", "HOPEFUL"]
    )

    # --- stance definitions (used in prompts) -----------------------------
    stance_definitions: dict = field(default_factory=lambda: {
        "HATE": (
            "Hostile, dehumanizing, exclusionary, or stigmatizing discourse "
            "toward people experiencing homelessness or related issues."
        ),
        "NEUTRAL": (
            "Descriptive, factual, or non-committal discourse about "
            "homelessness-related topics without overt stance."
        ),
        "HOPEFUL": (
            "Supportive, empathetic, constructive, or solution-oriented "
            "discourse expressing dignity, empathy, policy support, or "
            "constructive solutions."
        ),
    })

    # --- category definitions (used in prompts, from paper Table 8) -------
    category_definitions: dict = field(default_factory=lambda: {
        "Shelter & Housing": (
            "Physical living arrangements, availability, and quality of "
            "temporary or permanent housing solutions for people experiencing "
            "homelessness. E.g., shelters, housing units, tents, encampments, "
            "shelter beds, permanent housing."
        ),
        "Public Space": (
            "Use, accessibility, and maintenance of shared urban spaces where "
            "homelessness is visibly present. E.g., parks, sidewalks, stations, "
            "downtown areas, public places."
        ),
        "Public Safety": (
            "Concerns related to safety, hygiene, sanitation, and public "
            "health risks associated with homelessness. E.g., crime, fear, "
            "cleanliness, trash, disease, safety risks."
        ),
        "Addiction & Health": (
            "Discussions of substance use, mental health conditions, and "
            "access to healthcare or treatment services. E.g., drugs, "
            "addiction, alcoholism, mental illness, counseling, rehabilitation."
        ),
        "Policy & Governance": (
            "Institutional actions, laws, enforcement practices, and "
            "government-level responses to homelessness. E.g., ordinances, "
            "bans, policing, funding, city council decisions, enforcement."
        ),
        "Employment & Economy": (
            "Economic participation, work opportunities, and perceived "
            "impacts on businesses and productivity. E.g., jobs, job training, "
            "employment programs, businesses, customers."
        ),
        "Empathy & Support": (
            "Moral framing, social attitudes, dignity, stigma, and "
            "expressions of compassion or dehumanization. E.g., respect, "
            "dignity, slurs, deservingness, compassion, humanization."
        ),
        "Community Impact & Social Cohesion": (
            "Perceived effects of homelessness on neighbourhood relations, "
            "community harmony, and social coexistence. E.g., resident "
            "complaints, neighbourhood tension, coexistence, community burden."
        ),
    })

    # --- span-grounding thresholds ----------------------------------------
    span_overlap_threshold: float = 0.6  # token-level F1 for span checks
    rationale_grounding_threshold: float = 0.5

    # --- label normalisation maps -----------------------------------------
    stance_normalisation_map: dict = field(default_factory=lambda: {
        # Common LLM label variants → canonical ACOS-HD labels
        "positive": "HOPEFUL", "supportive": "HOPEFUL", "pro": "HOPEFUL",
        "favor": "HOPEFUL", "favour": "HOPEFUL", "support": "HOPEFUL",
        "constructive": "HOPEFUL", "empathetic": "HOPEFUL",
        "negative": "HATE", "toxic": "HATE", "against": "HATE",
        "harmful": "HATE", "hostile": "HATE", "hateful": "HATE",
        "dehumanizing": "HATE", "exclusionary": "HATE",
        "none": "NEUTRAL", "descriptive": "NEUTRAL", "factual": "NEUTRAL",
        "neutral": "NEUTRAL", "objective": "NEUTRAL",
        # case variants
        "hate": "HATE", "hopeful": "HOPEFUL",
        "Hate": "HATE", "Hopeful": "HOPEFUL", "Neutral": "NEUTRAL",
    })

    category_alias_map: dict = field(default_factory=lambda: {
        # Typo / variant → canonical category
        "Employment &Economy": "Employment & Economy",
        "Employment & economy": "Employment & Economy",
        "Community Impact and Cohesion": "Community Impact & Social Cohesion",
        "Community Impact & Cohesion": "Community Impact & Social Cohesion",
        "Addicton and Health": "Addiction & Health",
        "Addiction and Health": "Addiction & Health",
        "Shelter and Housing": "Shelter & Housing",
        " Shelter & Housing": "Shelter & Housing",
    })


# ═══════════════════════════════════════════════════════════════════════════
# 7.  MASTER CONFIG
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ACOSHDConfig:
    """Top-level configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    schema: SchemaConfig = field(default_factory=SchemaConfig)


def get_config() -> ACOSHDConfig:
    """Return the default configuration. Edit values here or override in code."""
    return ACOSHDConfig()
