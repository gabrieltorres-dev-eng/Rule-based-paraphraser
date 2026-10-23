import re
from ..doc import get_nlp, PH_RE

_CACHE: dict = {}

def parse(text: str):
    d = _CACHE.get(text)
    if d is None:
        if len(_CACHE) > 60000: _CACHE.clear()
        d = _CACHE[text] = get_nlp()(text)
    return d

def warm(texts):
    """Parse many strings in one batch (much faster than one call each) and cache them."""
    todo = [t for t in dict.fromkeys(texts) if t and t not in _CACHE]
    if not todo: return
    for t, d in zip(todo, get_nlp().pipe(todo, batch_size=64)):
        _CACHE[t] = d

FINAL_PUNCT = re.compile(r"[.!?]+[\"')\]]*$")

def split_final(text: str):
    """('body', '.') — trailing sentence punctuation split off."""
    m = FINAL_PUNCT.search(text.rstrip())
    if not m: return text.rstrip(), ""
    return text.rstrip()[:m.start()], m.group(0)

def n_words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", s))

def keep_case(tok_text: str, tok=None) -> bool:
    """True if the first word must keep its capital when moved mid-sentence."""
    w = tok_text.split()[0] if tok_text.split() else tok_text
    if PH_RE.fullmatch(w) or w == "I" or w.startswith("I'") : return True
    if len(w) > 1 and w[:2].isupper(): return True
    if tok is not None and (tok.pos_ == "PROPN" or tok.ent_type_ in ("NORP", "PERSON", "GPE", "ORG", "LOC")): return True
    return False

def lower_first(text: str, tok=None) -> str:
    if not text or keep_case(text, tok): return text
    return text[0].lower() + text[1:]

def upper_first(text: str) -> str:
    if not text or PH_RE.match(text): return text
    return text[0].upper() + text[1:]

def span_text(text, toks):
    return text[toks[0].idx: toks[-1].idx + len(toks[-1].text)]
