"""Active -> passive inversion with auxiliary selection and participle agreement done in-transform."""
import re
from lemminflect import getInflection
from ..glue import ok as glue_ok
from .util import parse, split_final, lower_first, upper_first, span_text, keep_case

STATIVE = {"be", "have", "seem", "become", "lack", "resemble", "cost", "weigh", "contain", "fit", "suit",
           "want", "like", "love", "hate", "know", "believe", "need", "own", "possess", "equal", "mean",
           "remain", "appear", "belong", "consist", "exist", "matter", "depend", "involve", "include",
           "comprise", "prefer", "owe", "deserve", "wish", "understand", "doubt", "mind", "surprise"}
SUBJ_TO_OBJ = {"he": "him", "she": "her", "they": "them"}
OBJ_TO_SUBJ = {"me": "I", "us": "we", "him": "he", "her": "she", "them": "they", "it": "it", "you": "you"}
SKIP_SUBJ = {"i", "we", "you", "it", "there", "this", "that", "these", "those", "who", "which", "what"}
BAD_CHILD = {"neg", "prt", "ccomp", "xcomp", "dative", "agent", "conj", "advcl", "cc", "csubj", "expl", "oprd", "npadvmod"}

AMBIGUOUS_PREP = {"to", "of", "for", "with", "about", "from", "on", "by", "as", "than", "like"}
POSSESSIVE_OPEN = {"his", "her", "their", "its", "this", "that", "these", "those", "our", "my", "your", "such"}

def _be(aux_kind, tense, plural, person1):
    if aux_kind == "simple":
        if tense == "past": return "were" if plural else "was"
        return "am" if person1 else ("are" if plural else "is")
    return None

def passivize(text: str):
    body, punct = split_final(text)
    if not punct or punct[0] not in ".": return None
    if any(ch in body for ch in '"*_\u201c\u201d()[]<>:;'): return None
    doc = parse(body)
    if len(list(doc.sents)) != 1: return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ != "VERB" or root.lemma_.lower() in STATIVE: return None
    if re.search(r"qz\d+x", root.text, re.I): return None          # a locked span is never inflected
    kids = list(root.children)
    subj = [c for c in kids if c.dep_ == "nsubj"]
    obj = [c for c in kids if c.dep_ in ("dobj", "obj")]
    if len(subj) != 1 or len(obj) != 1: return None
    if any(c.dep_ in BAD_CHILD for c in kids): return None
    subj, obj = subj[0], obj[0]
    if any(c.dep_ == "advmod" and c.i < root.i for c in kids): return None
    auxes = [c for c in kids if c.dep_ in ("aux", "auxpass")]
    if len(auxes) > 1: return None
    # tense / voice pattern
    if not auxes:
        if root.tag_ == "VBD": kind, tense = "simple", "past"
        elif root.tag_ in ("VBZ", "VBP"): kind, tense = "simple", "present"
        else: return None
    else:
        a = auxes[0]
        if a.tag_ == "MD" and root.tag_ == "VB": kind, tense, modal = "modal", None, a.text
        elif a.lemma_ == "have" and root.tag_ == "VBN": kind, tense = "perfect", ("past" if a.tag_ == "VBD" else "present")
        else: return None
    sl, sr = subj.left_edge.i, subj.right_edge.i
    ol, orr = obj.left_edge.i, obj.right_edge.i
    if sl != 0 or ol != root.i + 1: return None
    between = [t.i for t in doc if sr < t.i < root.i]
    if between != [a.i for a in auxes]: return None
    if orr - ol > 14 or sr - sl > 10: return None
    subj_text = body[doc[sl].idx: doc[sr].idx + len(doc[sr].text)]
    obj_text = body[doc[ol].idx: doc[orr].idx + len(doc[orr].text)]
    tail = body[doc[orr].idx + len(doc[orr].text):].strip()
    nxt = doc[orr + 1] if orr + 1 < len(doc) else None
    if nxt is not None and (nxt.text in (",", "-") or nxt.lower_ in AMBIGUOUS_PREP or nxt.pos_ in ("VERB", "AUX", "SCONJ")):
        return None                                      # tail may belong to the object (mis-attachment) -> skip
    if obj.lower_ in ("that", "which", "what", "whether", "how", "why") or obj.tag_ in ("IN", "WDT", "WP"): return None
    if doc[ol].lower_ in POSSESSIVE_OPEN: return None   # 'Their phones are ...' would have no antecedent
    if "," in obj_text or "," in subj_text: return None
    if subj.lower_ in SKIP_SUBJ or subj_text.lower() in SKIP_SUBJ: return None
    if subj.lower_ in SUBJ_TO_OBJ and subj.i == sr:   # pronoun subject -> accusative
        subj_text = SUBJ_TO_OBJ[subj.lower_]
    elif subj.pos_ == "PRON": return None
    else:
        subj_text = lower_first(subj_text, doc[sl])
    # new subject = old object; pronoun case + number
    first_person = False
    if obj.pos_ == "PRON" and obj.i == orr:
        if obj.lower_ not in OBJ_TO_SUBJ or obj.lower_ in ("it", "you"): return None
        new_subj = OBJ_TO_SUBJ[obj.lower_]
        plural = obj.lower_ in ("us", "them"); first_person = obj.lower_ == "me"
    else:
        new_subj = obj_text
        plural = obj.tag_ in ("NNS", "NNPS") or any(c.dep_ == "conj" for c in obj.children)
    new_subj = upper_first(new_subj)
    vbn = getInflection(root.lemma_, "VBN")
    if not vbn: return None
    vbn = vbn[0]
    if not glue_ok("by", "be"): return None                                       # 'by' and the auxiliary are inserted words
    if kind == "simple":
        aux = _be(kind, tense, plural, first_person)
        if not glue_ok(aux): return None
    elif kind == "modal":
        aux = f"{modal} be"
    else:
        aux = ("have" if plural or first_person else "has") if tense == "present" else "had"
        aux += " been"
    out = f"{new_subj} {aux} {vbn} by {subj_text}" + (f" {tail}" if tail else "") + punct
    return out
