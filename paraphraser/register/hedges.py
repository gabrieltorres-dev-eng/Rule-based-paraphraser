"""Hedge insertion (calibrating confidence, not adding facts). Claim detection: declarative sentences
carrying absolute markers and no hedge. Hedge frames come only from the human template bank."""
import re
import numpy as np
from ..routing.util import parse, split_final, lower_first, upper_first, n_words
from ..doc import PH_RE
from ..macro.discourse import first_words

ABSOLUTE = {"always", "never", "clearly", "obviously", "definitely", "certainly", "undoubtedly", "everyone", "nobody",
            "everybody", "proves", "prove", "absolutely", "completely", "totally", "entirely", "impossible",
            "guaranteed", "inevitably", "every", "all", "must", "no", "none"}
HEDGED = {"perhaps", "maybe", "probably", "might", "may", "could", "seems", "seem", "often", "usually", "generally",
          "typically", "sometimes", "arguably", "possibly", "likely", "tend", "tends", "apparently", "rarely"}
CONTRAST = ("but", "however", "yet", "still", "although", "though", "on the other hand", "nevertheless")

def is_claim(s: str) -> bool:
    body, punct = split_final(s)
    if punct != "." or n_words(body) < 8 or PH_RE.search(body) or "?" in s: return False
    words = {w.lower() for w in re.findall(r"[A-Za-z']+", body)}
    if words & HEDGED: return False
    if not (words & ABSOLUTE): return False
    return True

def _frame(text: str) -> str:
    t = text.strip().rstrip(",")
    return upper_first(t)

def apply(doc, profile, res, rng):
    rate = profile["rates"].get("hedge_front", 0.03)
    total = max(sum(len(p.sentences) for p in doc.paragraphs), 1)
    claims = [(pi, si) for pi, p in enumerate(doc.paragraphs) for si, s in enumerate(p.sentences)
              if si > 0 and is_claim(s) and not doc.paragraphs[pi].role == "title"]
    if not claims: return 0
    p_fire = float(np.clip(rate * total / len(claims), 0, 0.8))
    n = 0
    for pi, si in claims:
        if rng.random() >= p_fire: continue
        s = doc.paragraphs[pi].sentences[si]
        nxt = doc.paragraphs[pi].sentences[si + 1] if si + 1 < len(doc.paragraphs[pi].sentences) else ""
        fn = "concession" if nxt and first_words(nxt, 2).startswith(CONTRAST) and rng.random() < 0.6 else "hedge"
        tpl = res.templates.pick(fn, doc.register, rng, doc.used_templates)
        if not tpl and fn == "concession":
            tpl = res.templates.pick("hedge", doc.register, rng, doc.used_templates)
        if not tpl: continue
        new = f"{_frame(tpl['text'])}, {lower_first(s, parse(s)[0])}"
        doc.note("register", fn, pi, si, s, new)
        doc.paragraphs[pi].sentences[si] = new; n += 1
    return n
