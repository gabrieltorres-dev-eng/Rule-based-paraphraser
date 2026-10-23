"""Aggressive-mode reorderers that fire on ordinary long sentences:
  move_pp:     'Main verb X in the city.' <-> 'In the city, main verb X.'   (place/time prepositional phrases)
  swap_coords: 'A, and B.' -> 'B, and A.'                                   (independent clauses joined by 'and')"""
from .util import parse, split_final, lower_first, upper_first
from .clause import _has_dangling_pronoun

MOVABLE = {"in", "on", "at", "during", "after", "before", "within", "across", "throughout", "under", "among", "through",
           "over", "between", "since", "for"}
DISCOURSE_NOUNS = {"hand", "result", "addition", "contrast", "fact", "case", "general", "conclusion", "summary", "course", "example", "instance", "short", "turn", "particular", "end", "sense", "essence", "effect", "brief", "contrary", "words", "meantime", "time", "whole", "light"}
BAD = '"*_“”()[]<>;:'

def _clean(body):
    return not any(ch in body for ch in BAD)

def move_pp(text: str):
    body, punct = split_final(text)
    if not punct or punct[0] not in ".!" or not _clean(body): return None
    doc = parse(body)
    if len(list(doc.sents)) != 1: return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"): return None
    subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass", "expl")]
    if len(subj) != 1: return None
    n = len(doc)
    preps = [c for c in root.children if c.dep_ == "prep" and c.lower_ in MOVABLE]
    # trailing PP -> front
    for pp in preps:
        l, r = pp.left_edge.i, pp.right_edge.i
        if r == n - 1 and l >= 4 and (r - l + 1) >= 3 and subj[0].left_edge.i == 0 and doc[l - 1].text != ",":
            if "," in body[doc[l].idx:]: continue
            main = body[:doc[l].idx].rstrip(" ,")
            if len(main.split()) < 4: continue
            if doc[l - 1].pos_ in ("ADV", "AUX") or (root.lemma_ == "be" and not any(c.dep_ in ("attr", "acomp", "dobj", "oprd") for c in root.children)): continue
            if _has_dangling_pronoun([t for t in doc if t.i >= l], [t for t in doc if t.i < l]): continue
            return upper_first(body[doc[l].idx:]) + ", " + lower_first(main, doc[0]) + punct, "to_front"
    # leading PP -> back
    first = doc[0]
    if first.dep_ == "prep" and first.head.i == root.i and first.lower_ in MOVABLE:
        r = first.right_edge.i
        if r < 6 and doc[r].lower_ in DISCOURSE_NOUNS: return None                    # connectives ("On the other hand,") stay put
        if r + 2 < n and doc[r + 1].text == "," and subj[0].left_edge.i == r + 2 and (r + 1) >= 3:
            rest = body[doc[r + 1].idx + 1:].strip()
            if "," in rest or len(rest.split()) < 4: return None
            if _has_dangling_pronoun([t for t in doc if t.i > r + 1], [t for t in doc if t.i <= r]): return None
            return upper_first(rest) + " " + lower_first(body[:doc[r].idx + len(doc[r].text)], doc[0]) + punct, "to_back"
    return None

def swap_coords(text: str):
    body, punct = split_final(text)
    if not punct or punct[0] not in ".!" or not _clean(body): return None
    doc = parse(body)
    if len(list(doc.sents)) != 1: return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"): return None
    for cj in (c for c in root.children if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")):
        cc = [t for t in doc if t.dep_ == "cc" and t.head.i == root.i and t.lower_ == "and"]
        if len(cc) != 1: continue
        c = cc[0]
        if c.i < 2 or doc[c.i - 1].text != ",": continue
        subj2 = [t for t in cj.children if t.dep_ in ("nsubj", "nsubjpass")]
        subj1 = [t for t in root.children if t.dep_ in ("nsubj", "nsubjpass", "expl")]
        if len(subj2) != 1 or len(subj1) != 1: continue
        if cj.left_edge.i != c.i + 1 or subj2[0].left_edge.i != c.i + 1: continue
        if subj2[0].pos_ == "PRON" or subj1[0].left_edge.i != 0: continue
        a_toks = [t for t in doc if t.i < c.i - 1]
        b_toks = [t for t in doc if t.i > c.i]
        if len(a_toks) < 4 or len(b_toks) < 4 or cj.right_edge.i != len(doc) - 1: continue
        if _has_dangling_pronoun(b_toks, a_toks) or _has_dangling_pronoun(a_toks, b_toks): continue
        if any(t.lower_ in ("this", "these", "such", "former", "latter", "former") for t in doc): continue
        a = body[:doc[c.i - 1].idx].rstrip()
        b = body[doc[c.i + 1].idx:]
        return upper_first(b) + ", and " + lower_first(a, doc[0]) + punct
    return None
