"""
preprocessing.py - Multi-step text cleaning for ACOS-HD using ekphrasis + custom logic.
Handles: URL/hashtag/mention resolution, abbreviation expansion, emoji substitution,
spelling correction, redundancy removal, length filtering, offensive-tone normalisation.
All heavy objects loaded ONCE and reused.
"""
import hashlib, html, logging, re, sys, unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from bs4 import BeautifulSoup
from configs import CleaningConfig, TWITTER_API_KEY

logger = logging.getLogger(__name__)

# Add ekphrasis to path
_EK_ROOT = Path(__file__).resolve().parent / "gitclones" / "ekphrasis"
if str(_EK_ROOT) not in sys.path:
    sys.path.insert(0, str(_EK_ROOT))

from gitclones.ekphrasis.ekphrasis.classes.preprocessor import TextPreProcessor
from gitclones.ekphrasis.ekphrasis.classes.spellcorrect import SpellCorrector
from gitclones.ekphrasis.ekphrasis.classes.tokenizer import SocialTokenizer
from gitclones.ekphrasis.ekphrasis.dicts.emoticons import emoticons as EK_EMOTICONS

_text_proc = None
_spell_cor = None

try:
    import emoji as _emoji_lib
    def _demojize(t): return _emoji_lib.demojize(t, delimiters=(" ", " "))
except ImportError:
    def _demojize(t): return t

# ── Dictionaries ──────────────────────────────────────────────────────────
INTERNET_SLANG: Dict[str,str] = {
    "tbh":"to be honest","imo":"in my opinion","imho":"in my humble opinion",
    "smh":"shaking my head","af":"as fuck","ngl":"not gonna lie",
    "irl":"in real life","fwiw":"for what it is worth","brb":"be right back",
    "btw":"by the way","idk":"I do not know","lol":"laughing out loud",
    "lmao":"laughing my ass off","omg":"oh my god","stfu":"shut up",
    "gtfo":"get out","tfw":"that feeling when","ikr":"I know right",
    "rn":"right now","tbf":"to be fair","ppl":"people","govt":"government",
    "bc":"because","b4":"before","w/":"with","w/o":"without",
    "abt":"about","thru":"through","tho":"though","prolly":"probably",
    "gonna":"going to","wanna":"want to","gotta":"got to",
    "kinda":"kind of","sorta":"sort of","cuz":"because","dunno":"do not know",
}
DOMAIN_ACRONYMS: Dict[str,str] = {
    "peh":"people experiencing homelessness","hf":"housing first",
    "coc":"continuum of care","hud":"department of housing and urban development",
    "nimby":"not in my backyard","yimby":"yes in my backyard",
    "sro":"single room occupancy","mh":"mental health","sa":"substance abuse",
    "dv":"domestic violence","snap":"supplemental nutrition assistance program",
}
OFFENSIVE_NORM: Dict[str,str] = {
    "hobo":"person experiencing homelessness","hobos":"people experiencing homelessness",
    "bum":"person experiencing homelessness","bums":"people experiencing homelessness",
    "vagrants":"people without permanent housing","vagrant":"person without permanent housing",
    "crackhead":"person with substance use disorder","crackheads":"people with substance use disorders",
    "junkie":"person with substance use disorder","junkies":"people with substance use disorders",
    "druggies":"people with substance use disorders","druggie":"person with substance use disorder",
}

# ── Regex patterns ────────────────────────────────────────────────────────
_RE_URL = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.I)
_RE_MENTION = re.compile(r"@[\w]+")
_RE_DELETED = re.compile(r"^\s*\[(deleted|removed)\]\s*$", re.I)
_RE_REPEAT_CHARS = re.compile(r"(.)\1{2,}")
_RE_REPEAT_WORDS = re.compile(r"\b(\w+)(\s+\1){1,}\b", re.I)
_RE_REPEAT_PUNCT = re.compile(r"([!?.]){2,}")
_RE_MULTI_SPACE = re.compile(r"\s+")

# ── Init helpers (load once) ──────────────────────────────────────────────
def _init_ekphrasis(cfg: CleaningConfig) -> TextPreProcessor:
    global _text_proc
    if _text_proc is not None:
        return _text_proc
    logger.info("Initialising ekphrasis TextPreProcessor...")
    _text_proc = TextPreProcessor(
        normalize=["url","email","phone","user","time","date","percent","money"],
        annotate={"hashtag","allcaps","elongated","repeated"},
        fix_bad_unicode=True, segmenter=cfg.ekphrasis_segmenter_corpus,
        corrector=cfg.ekphrasis_corrector_corpus, unpack_hashtags=cfg.resolve_hashtags,
        unpack_contractions=cfg.expand_contractions, spell_correct_elong=cfg.fix_spelling,
        spell_correction=cfg.fix_spelling,
        tokenizer=SocialTokenizer(lowercase=False).tokenize,
        dicts=[EK_EMOTICONS], remove_tags=True,
    )
    return _text_proc

def _init_spell(cfg: CleaningConfig) -> SpellCorrector:
    global _spell_cor
    if _spell_cor is not None:
        return _spell_cor
    _spell_cor = SpellCorrector(corpus=cfg.ekphrasis_corrector_corpus)
    return _spell_cor

# ── Individual cleaning steps ─────────────────────────────────────────────
def is_empty_or_deleted(text: str) -> bool:
    if not text or not text.strip():
        return True
    return bool(_RE_DELETED.match(text.strip()))

def _url_importance(url: str, text: str, kws: List[str]) -> float:
    combined = (text + " " + url).lower()
    return min(sum(0.15 for k in kws if k.lower() in combined), 1.0)

def resolve_url(url: str, timeout: int = 5) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"ACOSHD-Bot/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        fp = ""
        for p in soup.find_all("p"):
            t = p.get_text(strip=True)
            if len(t) > 30: fp = t[:200]; break
        return " ".join(p for p in [title, fp] if p) or ""
    except Exception:
        return ""

def resolve_twitter_mention(username: str) -> str:
    if not TWITTER_API_KEY:
        return username
    try:
        r = requests.get("https://api.twitterapi.io/twitter/user/info",
            params={"userName": username.lstrip("@")},
            headers={"X-API-Key": TWITTER_API_KEY}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "success" and data.get("data"):
                ud = data["data"]
                return ud.get("name", username)
        return username
    except Exception:
        return username

def handle_urls(text: str, cfg: CleaningConfig) -> str:
    for url in _RE_URL.findall(text):
        imp = _url_importance(url, text, cfg.url_importance_keywords)
        if cfg.resolve_urls and imp >= 0.3:
            resolved = resolve_url(url, cfg.url_request_timeout)
            text = text.replace(url, f"[{resolved}]" if resolved else "")
        else:
            text = text.replace(url, "")
    return text

def handle_mentions(text: str, cfg: CleaningConfig) -> str:
    for m in _RE_MENTION.findall(text):
        if cfg.resolve_mentions and TWITTER_API_KEY:
            text = text.replace(m, resolve_twitter_mention(m))
        else:
            text = text.replace(m, "")
    return text

def expand_abbreviations(text: str) -> str:
    tokens = text.split()
    out = []
    for tok in tokens:
        low = tok.lower().strip(".,!?;:()")
        if low in INTERNET_SLANG:
            out.append(INTERNET_SLANG[low])
        elif low in DOMAIN_ACRONYMS:
            out.append(DOMAIN_ACRONYMS[low])
        else:
            out.append(tok)
    return " ".join(out)

def remove_repeated_content(text: str) -> str:
    text = _RE_REPEAT_CHARS.sub(r"\1\1", text)
    text = _RE_REPEAT_WORDS.sub(r"\1", text)
    text = _RE_REPEAT_PUNCT.sub(r"\1", text)
    # Deduplicate sentences
    sents = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    deduped = []
    for s in sents:
        h = hashlib.md5(s.strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h); deduped.append(s)
    return " ".join(deduped)

def normalize_offensive(text: str) -> str:
    for term, repl in OFFENSIVE_NORM.items():
        text = re.sub(r"\b" + re.escape(term) + r"\b", repl, text, flags=re.I)
    return text

def check_language(text: str) -> bool:
    try:
        from langdetect import detect
        return detect(text) == "en"
    except Exception:
        return True

# ── Main pipeline ─────────────────────────────────────────────────────────
def clean_text(text: str, cfg: CleaningConfig) -> Optional[str]:
    """Full cleaning pipeline. Returns cleaned text or None to discard."""
    if is_empty_or_deleted(text):
        return None
    # Unicode fix
    if cfg.fix_unicode:
        try:
            import ftfy; text = ftfy.fix_text(text)
        except ImportError: pass
    # Remove control chars
    if cfg.remove_garbage_syntax:
        text = html.unescape(text)
        text = "".join(c for c in text if unicodedata.category(c)[0] != "C" or c in "\n\t")
    # URLs
    text = handle_urls(text, cfg)
    # Mentions
    text = handle_mentions(text, cfg)
    # Emojis
    if cfg.substitute_emojis:
        text = _demojize(text)
    # Ekphrasis (hashtags, elongated, contractions, spelling, emoticons)
    try:
        proc = _init_ekphrasis(cfg)
        tokens = proc.pre_process_doc(text)
        text = " ".join(tokens) if isinstance(tokens, list) else tokens
    except Exception as e:
        logger.warning("Ekphrasis failed: %s", e)
    # Remaining hashtag symbols
    text = re.sub(r"#(\w+)", r"\1", text)
    # Abbreviations
    if cfg.expand_abbreviations or cfg.expand_slang:
        text = expand_abbreviations(text)
    # Numbers
    if cfg.remove_numbers:
        text = re.sub(r"\b\d+\b", "", text)
    # Repeated content
    if cfg.remove_repeated_chars or cfg.remove_repeated_words or cfg.remove_repeated_sentences:
        text = remove_repeated_content(text)
    # Offensive terms
    if cfg.normalize_offensive:
        text = normalize_offensive(text)
    # Whitespace
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    # Length
    if len(text) < cfg.min_text_length:
        return None
    if len(text) > cfg.max_text_length:
        text = text[:cfg.max_text_length]
    # Language
    if not check_language(text):
        return None
    return text

def clean_texts_batch(texts: List[str], cfg: CleaningConfig) -> List[Tuple[int, str]]:
    """Clean batch. Returns [(orig_index, cleaned_text)] for survivors."""
    _init_ekphrasis(cfg)
    results = []
    for i, t in enumerate(texts):
        c = clean_text(t, cfg)
        if c is not None:
            results.append((i, c))
    return results
