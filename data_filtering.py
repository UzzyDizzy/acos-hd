"""
data_filtering.py - Robust multi-stage filtering pipeline for ACOS-HD.
Loads data ONCE, applies keyword + semantic filtering, deduplicates against
gold_dataset.csv and all previously accepted samples. Uses vectorisation and caching.
"""
import hashlib, logging, os, pickle, re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from configs import ACOSHDConfig, FilterConfig, DataConfig

logger = logging.getLogger(__name__)

# ── Singleton data store ──────────────────────────────────────────────────
_stance_df: Optional[pd.DataFrame] = None
_gold_df: Optional[pd.DataFrame] = None
_gold_text_hashes: Optional[Set[str]] = None
_accepted_hashes: Set[str] = set()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def load_datasets(cfg: DataConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load stance + gold datasets exactly ONCE into module-level cache."""
    global _stance_df, _gold_df, _gold_text_hashes
    if _stance_df is not None and _gold_df is not None:
        return _stance_df, _gold_df

    logger.info("Loading stance_detection_dataset.csv ...")
    _stance_df = pd.read_csv(cfg.stance_dataset_path, dtype=str).fillna("")
    logger.info("  → %d rows", len(_stance_df))

    logger.info("Loading gold_dataset.csv ...")
    _gold_df = pd.read_csv(cfg.gold_dataset_path, dtype=str).fillna("")
    logger.info("  → %d rows", len(_gold_df))

    # Pre-compute gold text hashes for O(1) dedup lookups
    _gold_text_hashes = {_text_hash(t) for t in _gold_df["text"].tolist()}
    logger.info("  → %d unique gold text hashes", len(_gold_text_hashes))

    return _stance_df, _gold_df


def get_gold_df(cfg: DataConfig) -> pd.DataFrame:
    """Return the cached gold dataframe (loads if needed)."""
    global _gold_df
    if _gold_df is None:
        load_datasets(cfg)
    return _gold_df


# ── Keyword filtering (vectorised) ───────────────────────────────────────

def _build_keyword_pattern(keywords: List[str]) -> re.Pattern:
    escaped = [re.escape(k) for k in keywords]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def keyword_filter(df: pd.DataFrame, fcfg: FilterConfig) -> pd.DataFrame:
    """Vectorised keyword filtering across primary + secondary + immigration lists."""
    all_kw = fcfg.primary_keywords + fcfg.secondary_keywords + fcfg.immigration_keywords
    pattern = _build_keyword_pattern(all_kw)

    # Single vectorised pass
    text_col = df["text"].str.lower()
    mask = text_col.apply(lambda t: len(pattern.findall(t)) >= fcfg.min_keyword_hits)
    filtered = df[mask].copy()
    logger.info("Keyword filter: %d → %d rows", len(df), len(filtered))
    return filtered


# ── Semantic similarity filtering ─────────────────────────────────────────

def _load_embeddings_cache(cache_path: str):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None


def _save_embeddings_cache(embeddings: np.ndarray, cache_path: str):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)


def semantic_filter(
    df: pd.DataFrame, fcfg: FilterConfig, cache_dir: str
) -> pd.DataFrame:
    """Filter by cosine similarity to homelessness reference sentences."""
    if not fcfg.use_semantic_similarity:
        return df

    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        logger.warning("sentence-transformers not installed; skipping semantic filter.")
        return df

    cache_path = os.path.join(cache_dir, "candidate_embeddings.npy")
    ref_cache = os.path.join(cache_dir, "reference_embeddings.npy")

    logger.info("Loading sentence-transformer model: %s", fcfg.similarity_model)
    model = SentenceTransformer(fcfg.similarity_model)

    # Reference embeddings (small, always recompute or cache)
    ref_emb = _load_embeddings_cache(ref_cache)
    if ref_emb is None:
        ref_emb = model.encode(fcfg.reference_sentences, convert_to_numpy=True,
                               show_progress_bar=False)
        _save_embeddings_cache(ref_emb, ref_cache)

    # Candidate embeddings — cache by dataframe hash
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df["text"]).values.tobytes()).hexdigest()
    cand_cache = os.path.join(cache_dir, f"cand_emb_{df_hash}.npy")
    cand_emb = _load_embeddings_cache(cand_cache)
    if cand_emb is None:
        logger.info("Encoding %d candidate texts...", len(df))
        cand_emb = model.encode(df["text"].tolist(), convert_to_numpy=True,
                                show_progress_bar=True, batch_size=256)
        _save_embeddings_cache(cand_emb, cand_cache)

    # Cosine similarity: max over reference sentences
    # ref_emb: (R, D), cand_emb: (N, D)
    sim_matrix = np.dot(cand_emb, ref_emb.T) / (
        np.linalg.norm(cand_emb, axis=1, keepdims=True)
        * np.linalg.norm(ref_emb, axis=1, keepdims=True).T
        + 1e-8
    )
    max_sim = sim_matrix.max(axis=1)
    mask = max_sim >= fcfg.similarity_threshold
    filtered = df[mask].copy()
    logger.info("Semantic filter (threshold=%.2f): %d → %d rows",
                fcfg.similarity_threshold, len(df), len(filtered))
    return filtered


# ── Deduplication ─────────────────────────────────────────────────────────

def is_duplicate_of_gold(text: str) -> bool:
    """Check if text is a duplicate of any gold dataset entry."""
    global _gold_text_hashes
    if _gold_text_hashes is None:
        return False
    return _text_hash(text) in _gold_text_hashes


def is_duplicate_of_accepted(text: str) -> bool:
    """Check against all previously accepted samples (rolling dedup)."""
    return _text_hash(text) in _accepted_hashes


def register_accepted(text: str):
    """Add a newly accepted text to the rolling dedup set."""
    _accepted_hashes.add(_text_hash(text))


def fuzzy_dedup(df: pd.DataFrame, threshold: float = 0.85,
                num_perm: int = 128) -> pd.DataFrame:
    """Remove near-duplicate texts using MinHash LSH."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning("datasketch not installed; skipping fuzzy dedup.")
        return df

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep_indices = []
    seen_keys = set()

    for idx, text in tqdm(zip(df.index, df["text"]), total=len(df),
                          desc="Fuzzy dedup"):
        tokens = set(text.lower().split())
        m = MinHash(num_perm=num_perm)
        for t in tokens:
            m.update(t.encode("utf-8"))

        key = str(idx)
        result = lsh.query(m)
        if not result:
            lsh.insert(key, m)
            keep_indices.append(idx)
        # else: duplicate, skip

    filtered = df.loc[keep_indices].copy()
    logger.info("Fuzzy dedup (threshold=%.2f): %d → %d rows",
                threshold, len(df), len(filtered))
    return filtered


def exact_dedup_against_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any rows whose text exactly matches a gold dataset entry."""
    mask = ~df["text"].apply(is_duplicate_of_gold)
    filtered = df[mask].copy()
    logger.info("Gold dedup: %d → %d rows", len(df), len(filtered))
    return filtered


# ── Stance mapping ────────────────────────────────────────────────────────

STANCE_MAP = {
    "FAVOR": "HOPEFUL",
    "AGAINST": "HATE",
    "NONE": "NEUTRAL",
}


def map_stance(stance: str) -> str:
    """Map source dataset stance labels to ACOS-HD labels."""
    return STANCE_MAP.get(stance.upper().strip(), "NEUTRAL")


# ── Balanced sampling ─────────────────────────────────────────────────────

def balanced_sample(
    df: pd.DataFrame,
    samples_per_class: int,
    fcfg: FilterConfig,
) -> pd.DataFrame:
    """Sample balanced across stance classes with VAST priority weighting."""
    df = df.copy()
    df["mapped_stance"] = df["stance"].apply(map_stance)

    sampled_parts = []
    for stance_label in ["HATE", "NEUTRAL", "HOPEFUL"]:
        pool = df[df["mapped_stance"] == stance_label]
        if len(pool) == 0:
            logger.warning("No candidates for stance=%s", stance_label)
            continue

        # Weight VAST samples higher
        weights = pool["dataset"].apply(
            lambda d: fcfg.vast_priority_weight if d == "VAST" else 1.0
        )
        weights = weights / weights.sum()

        n = min(samples_per_class, len(pool))
        sampled = pool.sample(n=n, weights=weights, replace=False, random_state=42)
        sampled_parts.append(sampled)
        logger.info("Sampled %d for stance=%s (from %d candidates)",
                    n, stance_label, len(pool))

    result = pd.concat(sampled_parts, ignore_index=True)
    return result


# ── Main filtering pipeline ──────────────────────────────────────────────

def run_filtering(cfg: ACOSHDConfig) -> pd.DataFrame:
    """Execute the full filtering pipeline. Returns cleaned candidate pool."""
    # 1. Load data once
    stance_df, gold_df = load_datasets(cfg.data)

    # 2. Filter by selected datasets
    mask = stance_df["dataset"].isin(cfg.filter.datasets_to_sample)
    candidates = stance_df[mask].copy()
    logger.info("Dataset filter: %d → %d rows", len(stance_df), len(candidates))

    # 3. Keyword filtering (vectorised)
    candidates = keyword_filter(candidates, cfg.filter)

    # 4. Semantic similarity filtering (cached embeddings)
    # os.makedirs(cfg.data.cache_dir, exist_ok=True)
    # candidates = semantic_filter(candidates, cfg.filter, cfg.data.cache_dir)
    logger.info(
        "Skipping semantic similarity filtering "
        "(GPT relevance filter used instead)"
    )

    # 5. Exact dedup against gold
    candidates = exact_dedup_against_gold(candidates)

    # 6. Fuzzy dedup within candidates
    candidates = fuzzy_dedup(candidates, cfg.filter.dedup_fuzzy_threshold,
                             cfg.filter.dedup_num_perm)

    # 7. Balanced sampling
    candidates = balanced_sample(candidates, cfg.pipeline.samples_per_class * 3,
                                 cfg.filter)

    logger.info("Final candidate pool: %d rows", len(candidates))

    # Cache filtered results
    cache_path = os.path.join(cfg.data.cache_dir, "filtered_candidates.pkl")
    candidates.to_pickle(cache_path)
    logger.info("Cached filtered candidates to %s", cache_path)

    return candidates


def load_cached_candidates(cfg: ACOSHDConfig) -> Optional[pd.DataFrame]:
    """Load previously cached filtered candidates if available."""
    cache_path = os.path.join(cfg.data.cache_dir, "filtered_candidates.pkl")
    if os.path.exists(cache_path):
        logger.info("Loading cached candidates from %s", cache_path)
        return pd.read_pickle(cache_path)
    return None
