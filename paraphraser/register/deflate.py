"""Remove inflated AI-style wording by DELETION only (no words are added, so provenance is untouched):
* intensifier adverbs ('fundamentally', 'immensely', 'dramatically' ...) and puffy modifiers ('overall unity', 'a true test')
* synonym doublets ('reshaped and transformed' -> 'reshaped'), detected with WordNet synsets or GloVe similarity."""
from __future__ import annotations
from ..routing.util import parse, split_final
from ..doc import PH_RE
from .. import intensity

INTENSIFIERS = {"fundamentally", "immensely", "dramatically", "drastically", "truly", "extremely", "incredibly", "profoundly",
                "deeply", "undeniably", "remarkably", "tremendously", "vastly", "greatly", "highly", "hugely", "massively",
                "inherently", "seamlessly", "significantly", "substantially", "intricately", "meticulously", "undoubtedly",
                "indeed", "certainly", "really", "very", "forever", "absolutely", "utterly", "entirely", "completely", "fully"}
PUFFY_ADJ = {"overall", "true", "real", "mighty", "crucial", "pivotal", "vital", "intricate", "robust"}
KEEP_HEAD_NEG = {"not", "n't", "never"}

def _syn(a, b, res) -> bool:
    if a.lemma_.lower() == b.lemma_.lower(): return True
    try:
        wn = res.wordnet
        pos = {"VERB": "v", "ADJ": "a", "NOUN": "n", "ADV": "r"}.get(a.pos_)
        if pos and wn:
            sa = set(wn.synsets(a.lemma_, pos=pos)); sb = set(wn.synsets(b.lemma_, pos=pos))
            if sa & sb: return True
    except Exception:
        pass
    v = getattr(res, "vectors", None)
    try:
        x, y = a.lemma_.lower(), b.lemma_.lower()
        if not (v and x in v and y in v): return False
        return float(v.similarity(x, y)) >= (0.5 if a.pos_ == "VERB" else 0.65)      # verb doublets are the typical AI tell
    except Exception:
        return False

def deflate_sentence(s: str, res, rng, p: float):
    body, punct = split_final(s)
    if not punct or len(body.split()) < 6: return None
    doc = parse(body)
    drop: set[int] = set()
    for t in doc:
        if t.i in drop or PH_RE.fullmatch(t.text): continue
        # intensifier adverbs
        if t.lower_ in INTENSIFIERS and t.dep_ == "advmod" and t.head.pos_ in ("VERB", "ADJ", "ADV", "AUX") \
                and not any(c.lower_ in KEEP_HEAD_NEG for c in t.head.children) and rng.random() < p:
            if t.lower_ == "very" and t.head.pos_ != "ADJ": continue
            drop.add(t.i)
            if t.i + 1 < len(doc) and doc[t.i + 1].text == "," and t.i == 0: drop.add(t.i + 1)
        # puffy adjectives in front of a noun that keeps another modifier or a determiner
        elif t.lower_ in PUFFY_ADJ and t.dep_ == "amod" and t.head.pos_ == "NOUN" and t.i < t.head.i and rng.random() < p:
            if t.lower_ in ("true", "real") and not any(c.lower_ in ("a", "an") for c in t.head.children): continue
            drop.add(t.i)
            if t.i + 1 < len(doc) and doc[t.i + 1].text == ",": drop.add(t.i + 1)
        # synonym doublets: 'X and Y' where Y means X
        elif t.dep_ == "conj" and t.pos_ == t.head.pos_ and t.pos_ in ("VERB", "ADJ", "NOUN") and t.i == t.head.i + 2:
            cc = doc[t.head.i + 1]
            if cc.lower_ in ("and", "or") and _syn(t.head, t, res) and rng.random() < p:
                if not list(t.children): drop.update({cc.i, t.i})                      # 'X and Y' -> 'X'
                elif {c.i for c in t.head.children if c.i > t.head.i} <= {cc.i, t.i}:
                    drop.update({t.head.i, cc.i})                                       # 'X and Y obj' -> 'Y obj'
    for i in range(1, len(doc) - 1):                    # surface fallback when the parser mislabels the pair
        a, cc, b = doc[i - 1], doc[i], doc[i + 1]
        if cc.lower_ == "and" and a.pos_ == b.pos_ and a.tag_[:2] == b.tag_[:2] and a.pos_ in ("VERB", "ADJ") and not drop & {a.i, cc.i, b.i} \
                and (i + 2 >= len(doc) or doc[i + 2].lower_ not in ("and", ",")) and _syn(a, b, res) and rng.random() < p:
            drop.update({cc.i, b.i})
    if not drop: return None
    out = "".join(t.text_with_ws for t in doc if t.i not in drop)
    out = " ".join(out.split()).replace(" ,", ",").replace(",,", ",").strip(" ,")
    if len(out.split()) < 4: return None
    if body[:1].isupper(): out = out[:1].upper() + out[1:]
    return out + punct

def apply(doc, res, rng):
    p = 0.9 if intensity.aggressive() else 0.5
    n = 0
    for pi, para in enumerate(doc.paragraphs):
        for si, s in enumerate(para.sentences):
            if PH_RE.fullmatch(s.strip()): continue
            new = deflate_sentence(s, res, rng, p)
            if new and new != s:
                para.sentences[si] = new; doc.note("register", "deflate", pi, si, s, new); n += 1
    return n
