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

from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

_client = OpenAI()

TOTAL_COST=0.0
BATCH_SIZE = 25
MIN_HOMELESSNESS_CONF=0.50

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

def check_language(text: str) -> bool:
    try:
        from langdetect import detect
        return detect(text) == "en"
    except Exception:
        return True

def process_posts_batch(
    posts: List[str],
    source_stances: List[str]
) -> List[Optional[str]]:

    """
    source_stances already mapped:

    FAVOR   -> HOPEFUL
    AGAINST -> HATE
    NONE    -> NEUTRAL
    """

    numbered=[]

    for i,(p,s) in enumerate(
        zip(posts,source_stances)
    ):

        numbered.append(
f"""
[{i}]
STANCE:{s}

TEXT:
{p}
"""
        )

    prompt=f"""
For EACH item:

1. Determine if genuinely related to homelessness discourse
2. Give confidence (0-1)
3. Rewrite ONLY if relevant

Rules:

- If DIRECT → preserve meaning exactly
- If INDIRECT → adapt naturally to homelessness
- If UNRELATED → create a realistic homelessness discussion
  preserving the original stance and issue type

Examples:

poverty → housing insecurity
public safety → encampments/community impact
support → shelter resources
healthcare → services for unhoused populations
dignity → treatment of homeless people

- exactly 12–20 words
- One sentence only
- General wording like gold samples
- Preserve original stance EXACTLY
- Preserve original supportive/neutral/hostile sentiment
- Preserve original policy position
- Preserve homelessness context if present
- Rewrite as a NATURAL social-media statement
- Sound like a person expressing an opinion
- Avoid article-summary language
- Avoid names unless essential
- Do NOT have mentions, hastags like @USER
- Generalize unnecessary specifics
- Remove excessive details
- Keep core argument only
- ONE sentence only

Return ONLY:

{{
"results":[
{{
"id":0,
"related":true,
"confidence":0.91,
"rewrite":"..."
}}
]
}}

Posts:

{chr(10).join(numbered)}
"""

    for attempt in range(3):

        try:

            response=_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type":"json_object"},
                temperature=0,
                messages=[
                    {
                        "role":"system",
                        "content":
                        """
You are a controlled dataset rewriter.

Output valid JSON only.

Rewrite as natural social-media text.

Do not produce summaries, news headlines, encyclopedia wording, or academic language. Do

Never modify stance.
Never modify meaning.
Never generate >30 words.
"""
                    },
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]
            )

            usage=response.usage

            input_cost=(
                usage.prompt_tokens/1_000_000
            )*0.15

            output_cost=(
                usage.completion_tokens/1_000_000
            )*0.60

            global TOTAL_COST
            TOTAL_COST += input_cost+output_cost

            result=json.loads(
                response.choices[0].message.content
            )["results"]

            outputs=[None]*len(posts)

            for item in result:

                idx=item["id"]

                if (
                    item["related"]
                    and item["confidence"]>=MIN_HOMELESSNESS_CONF
                ):

                    text=item["rewrite"]

                    # hard safeguard
                    words=text.split()

                    if len(words)>30:
                        text=" ".join(
                            words[:30]
                        )

                    outputs[idx]=text

            return outputs

        except Exception as e:

            logger.warning(
                f"Attempt {attempt+1}/3: {e}"
            )

    return [None]*len(posts)

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

    # Remove dataset anonymization artifacts
    # text = re.sub(
    #     r'\b(?:USER|user)\b|<user>|@USER\d*|@\w+',
    #     '',
    #     text,
    #     flags=re.I
    # )

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
    # Whitespace
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    # Length
    if len(text) < cfg.min_text_length:
        return None
    # Language
    if not check_language(text):
        return None
    return text

def clean_texts_batch(
    texts: List[str],
    stances: List[str],
    cfg: CleaningConfig
)->List[Tuple[int,str]]:

    _init_ekphrasis(cfg)

    cleaned=[]

    # Regular preprocessing only
    for i,(t,s) in enumerate(
        zip(texts, stances)
    ):

        c=clean_text(t,cfg)

        if c is not None:

            cleaned.append((i,c,s))

    if not cleaned:
        return []

    indices=[x[0] for x in cleaned]
    posts=[x[1] for x in cleaned]
    stances=[x[2] for x in cleaned]

    final=[]

    for start in range(
        0,
        len(posts),
        BATCH_SIZE
    ):

        batch_posts=posts[
            start:start+BATCH_SIZE
        ]

        batch_idx=indices[
            start:start+BATCH_SIZE
        ]

        batch_stances=stances[
            start:start+BATCH_SIZE
        ]

        # rewritten=process_posts_batch(
        #     batch_posts,
        #     batch_stances
        # )

        for idx,text in zip(
            batch_idx,
            batch_posts #rewritten
        ):

            if text:
                final.append(
                    (
                        idx,
                        text
                    )
                )
    logger.info(
        f"Kept {len(final)}/{len(texts)} "
        f"after homelessness filtering"
    )

    logger.info(
        f"Total rewrite cost=${TOTAL_COST:.6f}"
    )

    if len(final)>0:
        logger.info(
            f"Average sample cost=${TOTAL_COST/len(final):.6f}"
        )

    return final
