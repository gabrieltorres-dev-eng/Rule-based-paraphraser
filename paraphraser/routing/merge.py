"""Cross-sentence rewrites that reuse the input's own words (plus corpus-attested glue: 'and', 'which', 'who'):
* merge_neighbours : 'Canada would send a force. Canada would field 600,000 men.' -> '... a force and would field ...'
* relative_clauses : '... the Conscription Crisis of 1917. This would widen ...' -> '..., which would widen ...'
* vary_openings    : when neighbouring sentences open the same way, front a time/place phrase:
                     'Canada was at war in August 1914.' -> 'In August 1914, Canada was at war.'"""
from __future__ import annotations
from .util import parse, split_final, lower_first, upper_first, n_words
from ..doc import PH_RE
from .. import glue

PEOPLE = {"soldier", "man", "woman", "people", "person", "citizen", "worker", "student", "child", "leader", "farmer", "officer",
          "volunteer", "veteran", "canadian", "politician", "writer", "reader", "teacher", "parent", "member", "nurse"}
TIME_PLACE_ENTS = {"DATE", "TIME", "GPE", "LOC", "EVENT", "FAC"}
PLACE_WORDS = {"front", "battlefield", "home", "abroad", "overseas", "europe", "school", "city", "country", "field"}
FRONT_PREPS = {"in", "on", "at", "during", "after", "before", "by", "throughout", "across", "within", "until", "since", "over"}
LINK_START = {"however", "but", "and", "so", "also", "then", "yet", "still", "thus", "therefore", "moreover", "furthermore",
              "ultimately", "soon", "finally", "instead", "meanwhile", "other", "another", "for", "because", "while", "although"}

def _ok_sentence(s):
    return not PH_RE.fullmatch(s.strip()) and '"' not in s and "“" not in s and ";" not in s and ":" not in s

def _subject(doc):
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"): return None, None
    subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")]
    if len(subj) > 1: subj = [c for c in subj if c.i == 0][:1]          # misparsed appositive gives two subjects; keep the first
    return root, (subj[0] if len(subj) == 1 else None)

def _rest_after_subject(doc, subj, body):
    """Text after a sentence-initial subject span (None if the subject is not a clean prefix)."""
    if subj.left_edge.i != 0: return None
    r = subj.right_edge.i
    if any(t.text == "," for t in doc[: r + 1]): return None
    if r + 1 >= len(doc) or doc[r + 1].text == ",": return None
    return body[doc[r + 1].idx:].strip()

# ------------------------------------------------------------------ 1. merge neighbours sharing a subject
def merge_pair(s1, s2, restore=lambda x: x):
    if not (_ok_sentence(s1) and _ok_sentence(s2)) or not glue.ok("and"): return None
    b1, p1 = split_final(s1); b2, p2 = split_final(s2)
    if p1 != "." or p2 != ".": return None
    if n_words(b1) + n_words(b2) > 38 or n_words(b2) < 4: return None
    d1, d2 = parse(b1), parse(b2)
    if len(list(d1.sents)) != 1 or len(list(d2.sents)) != 1: return None
    r1, s1j = _subject(d1); r2, s2j = _subject(d2)
    if s1j is None or s2j is None or d2[0].lower_ in LINK_START: return None
    if any(c.dep_ == "conj" and c.pos_ in ("VERB", "AUX") for c in r1.children): return None   # already 'X and Y'
    same = restore(" ".join(t.text for t in s1j.subtree)).lower() == restore(" ".join(t.text for t in s2j.subtree)).lower()
    pron = s2j.lower_ in ("it", "they") and s2j.i == 0 and s1j.pos_ in ("NOUN", "PROPN") and s1j.lemma_.lower() not in PEOPLE \
        and ((s2j.lower_ == "it") == (s1j.tag_ in ("NN", "NNP"))) and s1j.left_edge.i == 0
    same = same or (len(list(s2j.subtree)) == 1 and restore(s1j.text).lower() == restore(s2j.text).lower())
    if not (same or pron): return None
    rest = _rest_after_subject(d2, s2j, b2)
    if not rest or rest.split()[0].lower() in ("is", "are", "was", "were", "'s"): return None   # avoid 'X is A and is B' copula chains
    return b1 + " and " + rest + "."

def merge_neighbours(doc, rng, p=0.5):
    n = 0
    for pi, para in enumerate(doc.paragraphs):
        S = para.sentences; i = 0
        while i < len(S) - 1:
            new = merge_pair(S[i], S[i + 1], doc.restore) if rng.random() < p else None
            if new and doc.fresh(new):
                doc.note("routing", "merge_neighbours", pi, i, S[i] + " " + S[i + 1], new)
                S[i:i + 2] = [new]; n += 1
            i += 1
    return n

# ------------------------------------------------------------------ 2. fold a sentence into the noun it describes
def relcl_pair(s1, s2, restore=lambda x: x):
    if not (_ok_sentence(s1) and _ok_sentence(s2)): return None
    b1, p1 = split_final(s1); b2, p2 = split_final(s2)
    if p1 != "." or p2 not in (".", "!") or n_words(b1) + n_words(b2) > 42: return None
    d1, d2 = parse(b1), parse(b2)
    if len(list(d1.sents)) != 1 or len(list(d2.sents)) != 1: return None
    last = d1[len(d1) - 1]
    head = last if last.pos_ in ("NOUN", "PROPN", "NUM") else None
    while head is not None and head.dep_ in ("compound", "nummod", "pobj") and head.head.i > head.i: head = head.head
    if head is None or head.right_edge.i != len(d1) - 1: return None
    r2, sj = _subject(d2)
    if sj is None or sj.left_edge.i != 0: return None
    this = sj.lower_ in ("this", "these") and sj.i == 0 and sj.pos_ == "PRON"
    same = sj.pos_ in ("NOUN", "PROPN") and restore(sj.lemma_).lower() == restore(head.lemma_).lower() and d2[0].lower_ in ("the", "this", "these", sj.lower_)
    if not (this or same): return None
    rest = _rest_after_subject(d2, sj, b2)
    if not rest or rest.split()[0].lower() in ("is", "was") and this: pass
    if not rest: return None
    who = head.ent_type_ == "PERSON" or head.lemma_.lower() in PEOPLE
    rel = "who" if who and not this else "which"
    if not glue.ok(rel): return None
    if this and head.ent_type_ == "PERSON": return None
    return b1 + ", " + rel + " " + rest + p2

def relative_clauses(doc, rng, p=0.7):
    n = 0
    for pi, para in enumerate(doc.paragraphs):
        S = para.sentences; i = 0
        while i < len(S) - 1:
            new = relcl_pair(S[i], S[i + 1], doc.restore) if rng.random() < p else None
            if new and doc.fresh(new):
                doc.note("routing", "relative_clause", pi, i, S[i] + " " + S[i + 1], new)
                S[i:i + 2] = [new]; n += 1
            i += 1
    return n

# ------------------------------------------------------------------ 3. vary repeated openings
def _opening(s):
    w = s.split()
    return w[0].lower().strip(",") if w else ""

def front_time_place(s):
    if not _ok_sentence(s): return None
    body, punct = split_final(s)
    if not punct or punct[0] not in ".!" or n_words(body) < 7: return None
    d = parse(body)
    if len(list(d.sents)) != 1: return None
    root, subj = _subject(d)
    if subj is None or subj.left_edge.i != 0: return None
    cands = [c for c in root.children if c.dep_ in ("prep", "npadvmod")]
    last = [t for t in d if t.dep_ == "prep" and t.right_edge.i == len(d) - 1 and t.head.i != root.i
            and any(a.dep_ in ("dobj", "attr", "pobj") for a in t.ancestors)]
    cands += last[::-1]                                  # sentence-final PP hanging off the object ('... on the Western Front')
    for pp in cands:
        span = list(pp.subtree); l, r = span[0].i, span[-1].i
        if l < 2 or r - l > 8 or any(t.text in ",()" for t in span): continue
        if pp.dep_ == "prep" and pp.lower_ not in FRONT_PREPS: continue
        if not any(t.ent_type_ in TIME_PLACE_ENTS or t.lemma_.lower() in PLACE_WORDS or PH_RE.fullmatch(t.text) for t in span): continue
        if pp.dep_ == "npadvmod" and not any(t.ent_type_ in ("DATE", "TIME") or PH_RE.fullmatch(t.text) for t in span): continue
        if root.lemma_ == "be" and not any(c.dep_ in ("attr", "acomp", "oprd") or (c.dep_ == "prep" and c.i != pp.i)
                                          for c in root.children): continue        # 'X was [at war]' must keep its complement
        before = body[: d[l].idx].rstrip(" ,")
        after = body[d[r].idx + len(d[r].text):].strip(" ,")
        main = (before + (" " + after if after else "")).strip()
        if n_words(main) < 4: continue
        md = parse(main)
        if md[len(md) - 1].pos_ in ("ADV", "AUX", "ADP", "CCONJ", "DET"): continue
        return upper_first(body[d[l].idx: d[r].idx + len(d[r].text)]) + ", " + lower_first(main, d[0]) + punct
    return None

def vary_openings(doc, rng, p=0.8, p_any=0.25):
    n = 0
    for pi, para in enumerate(doc.paragraphs):
        S = para.sentences
        for i in range(len(S)):
            prev = {_opening(S[k]) for k in (i - 1, i - 2) if k >= 0}
            repeated = _opening(S[i]) in prev or (_opening(S[i]) in ("it", "this", "there", "they") and i > 0)
            if not ((repeated and rng.random() < p) or rng.random() < p_any): continue
            new = front_time_place(S[i])
            if new and doc.fresh(new):
                doc.note("routing", "vary_opening", pi, i, S[i], new); S[i] = new; n += 1
    return n
