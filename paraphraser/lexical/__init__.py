"""Phase 5: embedding-constrained synonym perturbation. WordNet proposes candidates; three filters
(semantic safety, frequency band, domain lock) accept or reject. Nothing is generated."""
from __future__ import annotations
from lemminflect import getInflection
from collections import Counter
import numpy as np
from ..doc import PH_RE, get_nlp
from ..routing.util import parse
from .. import intensity

POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a"}
SKIP_LEMMAS = {"be", "have", "do", "say", "make", "get", "go", "take", "give", "use", "thing", "way", "time", "people",
               "year", "day", "good", "bad", "new", "other", "many", "much", "more", "most", "such", "own", "same"}

def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0

def _ctx_vec(vecs, toks, skip):
    vs = [vecs[t.lower_] for t in toks if t.i != skip and t.is_alpha and not t.is_stop and t.lower_ in vecs]
    return np.mean(vs, axis=0) if vs else None

import re as _re
_GLOSS = {}
_STOPW = set("a an the of to in on at for with by from and or but is are was were be been that this which who whom whose it its as into than then so not no also very more most such one some any".split())

def _sense_vec(syn, vecs):
    k = syn.name()
    if k not in _GLOSS:
        words = _re.findall(r"[a-z]+", (syn.definition() + " " + " ".join(syn.examples()[:2])).lower())
        words += [l.name().lower() for l in syn.lemmas() if "_" not in l.name()]
        vs = [vecs[w] for w in words if w in vecs and w not in _STOPW]
        _GLOSS[k] = np.mean(vs, axis=0) if vs else None
    return _GLOSS[k]

def pick_sense(res, tok, ctx, vecs, pos):
    """Choose the WordNet sense whose definition best matches the sentence context (small prior for the most common sense)."""
    syns = res.wordnet.synsets(tok.lemma_, pos=pos)[:4]
    best, bs = None, -9.0
    for i, syn in enumerate(syns):
        v = _sense_vec(syn, vecs)
        if v is None or ctx is None: continue
        sc = _cos(ctx, v) + (0.05 if i == 0 else 0.0)
        if sc > bs: best, bs = (i, syn), sc
    return best

LOOSE = False        # set by the CLI (--lexical loose): more swaps, slightly higher meaning risk

def candidates(res, tok, ctx=None, vecs=None):
    """Single-word WordNet synonyms for the token's POS (no multiword, no antonym, no same lemma)."""
    pos = POS_MAP.get(tok.pos_)
    if not pos: return []
    out, seen = [], {tok.lemma_.lower()}
    nsense = 2 if LOOSE else 1
    syns = list(enumerate(res.wordnet.synsets(tok.lemma_, pos=pos)[:nsense]))
    if ctx is not None and vecs is not None and LOOSE:                  # loose mode: choose the sense from context, not by rank
        ps = pick_sense(res, tok, ctx, vecs, pos)
        if ps: syns = [ps]
    for si, syn in syns:
        for l in syn.lemmas():
            w = l.name()
            if "_" in w or "-" in w or not w.isalpha() or w.lower() in seen: continue
            seen.add(w.lower())
            alt = res.wordnet.synsets(w, pos=pos)
            if not alt or syn not in alt[:4 if LOOSE else nsense]: continue          # mutual dominance: the sense must be top-ranked for the synonym too
            out.append((w, si))
        if LOOSE:                                                   # tier 2: related senses (similar-to, see-also, parent category)
            for rel in list(syn.similar_tos()) + list(syn.also_sees()) + list(syn.hypernyms()):
                for l in rel.lemmas():
                    w = l.name()
                    if "_" in w or "-" in w or not w.isalpha() or w.lower() in seen: continue
                    seen.add(w.lower()); out.append((w, 2))
    if LOOSE:                                                           # tier 3: Moby thesaurus, mutual listing + same POS in WordNet
        for w in res.mutual_synonyms(tok.lemma_.lower()):
            if w in seen or len(w) < 3: continue
            if not res.wordnet.synsets(w, pos=pos): continue
            seen.add(w); out.append((w, 3))
    return out

def inflect_like(tok, new_lemma):
    tag = tok.tag_
    try:
        r = getInflection(new_lemma, tag)
    except Exception:
        return None
    if not r: return None
    w = r[0]
    return w[0].upper() + w[1:] if tok.text[:1].isupper() and tok.i == 0 else w

def perturb(doc, profile, res, rng, rate=None, ctx_margin=0.08, band=1.0, close=0.55, domain_k=3):
    """Replace a corpus-fitted fraction of eligible content words with filtered synonyms."""
    rate = 0.08 if rate is None else rate
    if LOOSE: ctx_margin, band, close, domain_k = 0.12, 1.5, 0.45, 5
    vecs = res.vectors
    if vecs is None: return 0
    # domain lock: lemmas frequent in this document are domain vocabulary
    cnt = Counter()
    docs = {}
    for pi, p in enumerate(doc.paragraphs):
        for si, s in enumerate(p.sentences):
            d = parse(s); docs[(pi, si)] = d
            cnt.update(t.lemma_.lower() for t in d if t.pos_ in POS_MAP)
    done = 0
    out_doc: Counter = Counter()                       # words now in the document (replacements included)
    out_par: dict = {}                                 # replacement words already used, per paragraph
    for d in docs.values():
        out_doc.update(t.lemma_.lower() for t in d if t.is_alpha)
    for (pi, si), d in docs.items():
        sent_lem = {t.lemma_.lower() for t in d if t.is_alpha}
        s = doc.paragraphs[pi].sentences[si]
        pieces, last, changed = [], 0, False
        for tok in d:
            if tok.pos_ not in POS_MAP or tok.is_stop or not tok.is_alpha or PH_RE.fullmatch(tok.text): continue
            lem = tok.lemma_.lower()
            if lem in SKIP_LEMMAS or cnt[lem] >= domain_k or lem not in vecs: continue
            if tok.dep_ in ("aux", "auxpass", "neg") or tok.ent_type_: continue
            if tok.dep_ == "compound" or (tok.pos_ == "NOUN" and tok.dep_ in ("nmod", "npadvmod") and tok.head.pos_ == "NOUN" and tok.head.i > tok.i):
                continue                                                          # 'case study', 'solar panel': part of a compound term
            if (tok.i > 0 and d[tok.i - 1].text == "-") or (tok.i + 1 < len(d) and d[tok.i + 1].text == "-") or "-" in tok.text: continue   # hyphenated compounds
            nxt = d[tok.i + 1] if tok.i + 1 < len(d) else None
            if tok.pos_ == "VERB" and (any(c.dep_ in ("prt", "xcomp") for c in tok.children) or (nxt is not None and nxt.pos_ == "ADP")):
                continue                                                          # phrasal / prepositional verbs
            if tok.pos_ == "VERB" and any(c.dep_ in ("ccomp", "xcomp") for c in tok.children): continue
            f_lem = res.zipf(lem)
            force = LOOSE and f_lem < 4.6                                          # rare / formal vocabulary: always attempt a plainer word
            if not force and rng.random() >= rate: continue
            ctx = _ctx_vec(vecs, d, tok.i)
            if ctx is None: continue
            base_sim = _cos(vecs[lem], ctx)
            f0 = res.zipf(lem)
            best = None
            for c, sense in candidates(res, tok, ctx, vecs):
                cl = c.lower()
                if cl not in vecs: continue
                if cl in intensity.RETIRED: continue                                # an earlier pass removed this word: never restore it
                if cl in sent_lem or cl in out_par.setdefault(pi, set()) or out_doc[cl] >= 2: continue    # no repeated replacements
                if cl[:5] == lem[:5] or cl.startswith(lem) or lem.startswith(cl): continue    # same word, different form
                dz = res.zipf(cl) - f0
                if LOOSE:
                    if dz < -0.5 or dz > 2.2: continue                              # allow plainer (more common) words, not rarer ones
                elif abs(dz) > band: continue                                       # Filter 2: frequency band
                sim = _cos(vecs[cl], ctx)
                if sim < base_sim - ctx_margin: continue                          # Filter 1: contextual fit vs the original
                closeness = _cos(vecs[cl], vecs[lem])
                if closeness < (close if sense == 0 else (max(close, 0.58) if sense == 1 else (0.52 if sense == 2 else 0.50))): continue   # second sense needs a closer match
                if closeness < close: continue                                    # Filter 1b: semantic closeness
                score = closeness + sim + (0.12 * min(dz, 2.0) if LOOSE else 0.0)   # prefer the plainer word
                if best is None or score > best[0]: best = (score, c)
            if not best: continue
            new = inflect_like(tok, best[1])
            if not new or new.lower() == tok.lower_: continue
            if tok.text.isupper() and len(tok.text) > 1: new = new.upper()
            elif tok.text[:1].isupper(): new = new[:1].upper() + new[1:]
            out_par.setdefault(pi, set()).add(new.lower()); out_doc[new.lower()] += 1; sent_lem.add(new.lower())
            ph = doc.lock(new, "lexical")
            pieces.append(s[last:tok.idx] + ph); last = tok.idx + len(tok.text); changed = True; done += 1
        if changed:
            new_s = "".join(pieces) + s[last:]
            if not doc.fresh(new_s): continue
            doc.note("lexical", "synonym", pi, si, s, new_s)
            doc.paragraphs[pi].sentences[si] = new_s
    return done
