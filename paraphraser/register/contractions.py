"""Expand or contract to match the register's measured contraction rate; instances chosen by seeded RNG."""
import re
import numpy as np
from ..calibration.measure import CONTRACTIONS, _APOS
from ..doc import PH_RE

# contract only when a word follows (so 'Yes, it is.' is left alone); negations contract anywhere
_NEG = {k for k in CONTRACTIONS if k.split()[-1] == "not" or k == "cannot"}
_CON_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, CONTRACTIONS), key=len, reverse=True)) + r")\b(?=(\s+\w)|(?<=not)|(?<=cannot))", re.I)
EXPAND = {"don't": "do not", "doesn't": "does not", "didn't": "did not", "isn't": "is not", "aren't": "are not",
          "wasn't": "was not", "weren't": "were not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
          "won't": "will not", "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
          "can't": "cannot", "i'm": "I am", "you're": "you are", "we're": "we are", "they're": "they are",
          "i've": "I have", "you've": "you have", "we've": "we have", "they've": "they have",
          "i'll": "I will", "you'll": "you will", "we'll": "we will", "they'll": "they will", "let's": "let us"}
_EXP_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, EXPAND), key=len, reverse=True)) + r")\b", re.I)

def _case(src, dst):
    if src.isupper() and len(src) > 1: return dst.upper()
    if src[:1].isupper(): return dst[:1].upper() + dst[1:]
    return dst

def _counts(doc):
    c = e = 0
    for _, _, s in doc.all_sentences():
        t = s.translate(_APOS)
        c += len(_EXP_RE.findall(t)); e += len(_CON_RE.findall(t))
    return c, e

def adjust(doc, profile, rng):
    con, exp = _counts(doc)
    tot = con + exp
    if tot == 0: return 0
    cur = con / tot; target = profile.rate("contraction")
    changed = 0
    if cur < target:
        p = (target - cur) / max(1 - cur, 1e-6); rx = _CON_RE
        sub = lambda m: _case(m.group(0), CONTRACTIONS[m.group(1).lower()])
    elif cur > target:
        p = (cur - target) / max(cur, 1e-6); rx = _EXP_RE
        sub = lambda m: _case(m.group(0), EXPAND[m.group(1).lower().translate(_APOS)])
    else:
        return 0
    for pi, p_ in enumerate(doc.paragraphs):
        for si, s in enumerate(p_.sentences):
            s0 = s.translate(_APOS)
            def maybe(m):
                nonlocal changed
                if rng.random() < p:
                    changed += 1; return sub(m)
                return m.group(0)
            new = rx.sub(maybe, s0)
            if new != s0:
                doc.note("register", "contraction", pi, si, s, new); p_.sentences[si] = new
    return changed
