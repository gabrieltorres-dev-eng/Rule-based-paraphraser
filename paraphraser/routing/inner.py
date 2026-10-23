"""Aggressive-mode rewrites INSIDE a sentence (grammar-level, nothing generated; only closed-class glue words
'the', 'of', 'that', "'s" are added, and every content word comes from the input):
  possessive <-> of-phrase     "the community's diesel use"  <->  "the diesel use of the community"
  participle <-> relative      "microgrids serving towns"    <->  "microgrids that serve towns"
  'that' toggle after reporting verbs   "concluded that X" <-> "concluded X"
  two-item noun/adjective coordination swap   "autumn and winter" -> "winter and autumn"
Each function returns the new sentence or None."""
from lemminflect import getInflection
import re
from .util import parse, lower_first, upper_first
from ..doc import PH_RE
from ..glue import ok as glue_ok

SAY = {"say", "conclude", "determine", "find", "show", "suggest", "believe", "think", "note", "argue", "report", "claim",
       "indicate", "demonstrate", "notice", "state", "learn", "know", "mean", "assume", "explain", "reveal", "observe",
       "estimate", "expect", "predict", "confirm", "hope", "feel", "recognize", "realize", "understand", "discover", "mention"}
NO_PART = {"include", "follow", "regard", "concern", "depend", "accord", "exclude", "be", "have", "involve", "provide", "base"}
BAD = '"*_“”()[]<>;'

def _ok(s):
    return not any(ch in s for ch in BAD)

def _span(s, doc, a, b):
    return s[doc[a].idx: doc[b].idx + len(doc[b].text)]

def _has_ph(toks):
    return any(PH_RE.search(t.text) for t in toks)

def _splice(s, doc, a, b, new):
    return s[:doc[a].idx] + new + s[doc[b].idx + len(doc[b].text):]

# ---------------------------------------------------------------- possessive <-> of
def poss_of(s):
    if not _ok(s): return None
    doc = parse(s)
    for tok in doc:
        # possessive -> of
        if tok.dep_ == "poss" and tok.pos_ in ("NOUN", "PROPN") and tok.i + 1 < len(doc) and doc[tok.i + 1].dep_ == "case":
            head = tok.head
            ap = doc[tok.i + 1]
            if head.pos_ != "NOUN" or head.i <= ap.i or head.i - ap.i > 4: continue
            if any(c.dep_ in ("prep", "conj", "cc", "relcl", "acl") for c in head.children): continue
            rest = [t for t in doc[ap.i + 1: head.i + 1]]
            if not rest or any(t.dep_ not in ("compound", "amod", "nummod") and t.i != head.i for t in rest): continue
            ps = tok.left_edge.i
            poss_toks = list(doc[ps: tok.i + 1])
            if len(poss_toks) > 5 or any(t.text in ",;:" for t in poss_toks) or _has_ph(poss_toks + rest): continue
            if any(t.pos_ == "PRON" for t in poss_toks): continue
            poss_txt = _span(s, doc, ps, tok.i)
            rest_txt = _span(s, doc, rest[0].i, head.i)
            if not glue_ok("the", "of"): return None
            lead = "The " if ps == 0 else "the "
            if ps == 0: poss_txt = lower_first(poss_txt, doc[0])
            return _splice(s, doc, ps, head.i, lead + rest_txt + " of " + poss_txt)
        # of -> possessive   ("the A of the B" -> "the B's A")
        if tok.lower_ == "of" and tok.dep_ == "prep" and tok.head.pos_ == "NOUN":
            h = tok.head
            objs = [c for c in tok.children if c.dep_ == "pobj"]
            if len(objs) != 1: continue
            o = objs[0]
            if o.pos_ != "NOUN" or o.right_edge.i != o.i or o.tag_ == "NNS" and False: continue
            if o.i - tok.i != 2 or doc[tok.i + 1].lower_ != "the": continue         # "of the B" with a bare noun B
            if any(c.dep_ not in ("det",) for c in o.children): continue
            if any(c.dep_ in ("prep", "conj", "cc", "relcl", "acl") for c in h.children if c.i != tok.i): continue
            if h.left_edge.i >= h.i or doc[h.left_edge.i].lower_ != "the": continue
            a_mods = list(doc[h.left_edge.i + 1: h.i + 1])
            if not a_mods or len(a_mods) > 3 or any(t.dep_ not in ("compound", "amod", "nummod") and t.i != h.i for t in a_mods): continue
            if tok.i != h.i + 1 or _has_ph(a_mods + [o]): continue
            poss = o.text + ("'" if o.text.endswith("s") else "'s")
            if not glue_ok("the", "of"): return None
            lead = "The " if h.left_edge.i == 0 else "the "
            return _splice(s, doc, h.left_edge.i, o.i, lead + poss + " " + _span(s, doc, a_mods[0].i, h.i))
    return None

# ---------------------------------------------------------------- participle <-> relative clause
def part_rel(s):
    if not _ok(s): return None
    doc = parse(s)
    for tok in doc:
        if tok.dep_ == "acl" and tok.tag_ == "VBG" and tok.head.pos_ in ("NOUN", "PROPN") and tok.i == tok.head.i + 1:
            if tok.lemma_ in NO_PART or any(c.dep_ in ("aux", "auxpass", "neg") for c in tok.children) or _has_ph([tok]): continue
            pl = tok.head.tag_ in ("NNS", "NNPS")
            inf = getInflection(tok.lemma_, "VBP" if pl else "VBZ")
            if not inf: continue
            if not glue_ok("that"): return None
            return _splice(s, doc, tok.i, tok.i, "that " + inf[0])
        if (tok.dep_ == "relcl" and tok.tag_ in ("VBP", "VBZ") and tok.lemma_ not in ("be", "have", "do") and tok.i >= 2
                and doc[tok.i - 1].lower_ in ("that", "which") and doc[tok.i - 1].dep_ == "nsubj" and doc[tok.i - 2].text != ","
                and not any(c.dep_ in ("aux", "auxpass", "neg", "nsubj", "nsubjpass") for c in tok.children if c.i != tok.i - 1)):
            g = getInflection(tok.lemma_, "VBG")
            if not g or _has_ph([tok]): continue
            return _splice(s, doc, tok.i - 1, tok.i, g[0])
    return None

# ---------------------------------------------------------------- 'that' toggle
def that_toggle(s):
    if not _ok(s): return None
    doc = parse(s)
    for v in doc:
        if v.lemma_ not in SAY or v.pos_ != "VERB": continue
        for c in v.children:
            if c.dep_ != "ccomp" or c.i < v.i: continue
            marks = [m for m in c.children if m.dep_ == "mark"]
            subj = [m for m in c.children if m.dep_ in ("nsubj", "nsubjpass")]
            if not subj: continue
            if marks and marks[0].lower_ == "that" and marks[0].i == v.i + 1 and v.i + 2 < len(doc):
                if doc[v.i + 2].lower_ in ("that", "this", "these", "those") or doc[v.i + 2].text == ",": continue
                return s[:doc[v.i + 1].idx] + s[doc[v.i + 2].idx:]
            if not marks and c.left_edge.i == v.i + 1 and subj[0].left_edge.i == v.i + 1 and doc[v.i + 1].lower_ not in ("that", "this"):
                if not glue_ok("that"): return None
                return s[:doc[v.i + 1].idx] + "that " + s[doc[v.i + 1].idx:]
    return None

# ---------------------------------------------------------------- two-item coordination swap
def coord_swap(s):
    if not _ok(s): return None
    doc = parse(s)
    for a in doc:
        if a.pos_ not in ("NOUN", "ADJ") or a.tag_ in ("NNP", "NNPS") or a.dep_ == "compound": continue
        cj = [c for c in a.children if c.dep_ == "conj"]
        cc = [c for c in a.children if c.dep_ == "cc" and c.lower_ in ("and", "or")]
        if len(cj) != 1 or len(cc) != 1: continue
        b = cj[0]
        if b.i != a.i + 2 or cc[0].i != a.i + 1 or b.pos_ != a.pos_ or b.tag_ in ("NNP", "NNPS"): continue
        if any(c.dep_ == "conj" for c in b.children) or b.right_edge.i != b.i or a.left_edge.i != a.i: continue
        if doc[a.i - 1].lower_ in ("both", "either", "neither") if a.i else False: continue
        if _has_ph([a, b]) or (a.ent_type_ not in ("", "DATE")) or (b.ent_type_ not in ("", "DATE")): continue
        return _splice(s, doc, a.i, b.i, f"{b.text} {cc[0].text} {a.text}")
    return None

REWRITES = (("poss_of", poss_of), ("part_rel", part_rel), ("that_toggle", that_toggle), ("coord_swap", coord_swap))

def apply_inner(doc, rng, p=0.9):
    n = 0
    for pi, para in enumerate(doc.paragraphs):
        for si in range(len(para.sentences)):
            for name, fn in REWRITES:
                s = para.sentences[si]
                if PH_RE.fullmatch(s.strip()) or rng.random() >= p: continue
                new = fn(s)
                if new and new != s and doc.fresh(new):
                    para.sentences[si] = new
                    doc.note("routing", name, pi, si, s, new); n += 1
    return n
