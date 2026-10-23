"""Controlled sentence fragments, derived only from words already in the sentence: a short copular
statement 'This is not always true.' becomes 'Not always true.' Mid-paragraph only, at the measured rate."""
import re
import numpy as np
from ..routing.util import parse, split_final, upper_first, n_words
from ..doc import PH_RE

SUBJ = {"this", "that", "it", "which"}

def to_fragment(sent: str):
    body, punct = split_final(sent)
    if punct != ".": return None
    doc = parse(body)
    if len(doc) < 4 or len(doc) > 9: return None
    if doc[0].lower_ not in SUBJ or doc[1].lemma_ != "be" or doc[1].tag_ not in ("VBZ", "VBD", "VBP"): return None
    rest = body[doc[2].idx:]
    if n_words(rest) < 2 or n_words(rest) > 6 or PH_RE.search(rest): return None
    if any(t.pos_ == "VERB" for t in doc[2:]) and not any(t.dep_ in ("acomp", "attr") for t in doc): return None
    return upper_first(rest) + "."

def apply(doc, profile, rng):
    elig = []
    for pi, p in enumerate(doc.paragraphs):
        for si in range(1, len(p.sentences) - 1):                   # mid-paragraph only
            f = to_fragment(p.sentences[si])
            if f: elig.append((pi, si, f))
    total = max(sum(len(p.sentences) for p in doc.paragraphs), 1)
    cur = 0.0
    p_fire = min(profile.rate("fragment") * total / max(len(elig), 1), 0.8) if elig else 0
    n = 0
    for pi, si, f in elig:
        if rng.random() < p_fire:
            old = doc.paragraphs[pi].sentences[si]
            doc.paragraphs[pi].sentences[si] = f
            doc.note("register", "fragment", pi, si, old, f); n += 1
    return n
