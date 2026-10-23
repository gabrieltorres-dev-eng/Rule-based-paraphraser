"""Thematic-chain breaking: 3+ consecutive sentences with the same subject get the 3rd rewritten
(passive inversion first, then clause transposition) so the chain is broken grammatically."""
from ..doc import PH_RE
from ..routing.util import parse
from ..routing.passive import passivize
from ..routing.clause import transpose

PRON = {"he", "she", "it", "they", "we", "i", "you"}

def subject_key(sent: str):
    doc = parse(sent)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None: return None
    subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")]
    if not subj: return None
    s = subj[0]
    return s.lemma_.lower() if s.lower_ not in PRON else s.lower_

def break_chains(doc, rng):
    n = 0
    for pi, p in enumerate(doc.paragraphs):
        keys = [subject_key(s) if not PH_RE.fullmatch(s.strip()) else None for s in p.sentences]
        run = 1
        for si in range(1, len(keys)):
            run = run + 1 if keys[si] and keys[si] == keys[si - 1] else 1
            if run >= 3:
                s = p.sentences[si]
                new = passivize(s)
                if new is None:
                    t = transpose(s); new = t[0] if t else None
                if new:
                    doc.note("macro", "break_chain", pi, si, s, new)
                    p.sentences[si] = new; run = 1; n += 1
    return n
