"""Clause transposition: 'Main, because sub.' <-> 'Because sub, main.' Grammar fixes are done in the
transform (capitalisation, commas). Suppressed when the flip would leave a pronoun before its antecedent."""
from .util import parse, split_final, lower_first, upper_first, span_text

SUBORD = {"because", "while", "although", "though", "when", "if", "since", "whereas", "unless",
          "after", "before", "once", "until", "whenever", "as"}
COMMA_TRAILING = {"although", "though", "whereas", "while"}      # comma kept before a trailing clause
WH = {"what", "who", "whom", "which", "whose", "how", "why", "where"}
NEEDS_OBJ = {"prefer", "like", "love", "want", "need", "know", "think", "say", "mean", "believe", "hope",
             "see", "feel", "find", "make", "get", "take", "give", "tell", "ask", "show", "do"}
COMPLEMENT = {"dobj", "obj", "attr", "acomp", "prep", "dative", "xcomp", "ccomp", "oprd", "npadvmod", "prt", "agent"}
PRON3 = {"it", "he", "she", "they", "him", "her", "them", "his", "its", "their", "this", "that", "these", "those", "theirs"}

def _has_dangling_pronoun(toks, other_toks) -> bool:
    """First clause would contain a 3rd-person pronoun whose antecedent is in the other clause."""
    pron = [t for t in toks if t.lower_ in PRON3 and t.pos_ == "PRON" and t.dep_ != "expl"]
    if not pron: return False
    return any(t.pos_ in ("NOUN", "PROPN") for t in other_toks)

def transpose(text: str):
    """Return (new_text, direction) or None. direction: 'to_front' or 'to_back'."""
    body, punct = split_final(text)
    if not punct or punct[0] not in ".!": return None
    if any(ch in body for ch in '"*_\u201c\u201d()[]<>'): return None       # stray quotes / markup
    doc = parse(body)
    sents = list(doc.sents)
    if len(sents) != 1: return None
    toks = [t for t in doc if not t.is_space]
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"): return None
    subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass", "expl")]
    if len(subj) != 1: return None                       # main clause needs its own finite subject
    subj = subj[0]
    cands = [t for t in root.children if t.dep_ == "advcl"
             and any(c.dep_ == "mark" and c.lower_ in SUBORD and c.i == t.left_edge.i for c in t.children)]
    cands += [t for t in root.children if t.dep_ == "advcl" and t not in cands
              and t.left_edge.lower_ in SUBORD and t.left_edge.dep_ in ("mark", "advmod")]
    if len(cands) != 1: return None
    adv = cands[0]
    main_all = [t for t in doc if not (adv.left_edge.i <= t.i <= adv.right_edge.i)]
    if any(t.lower_ in WH for t in main_all): return None                       # 'is what' / embedded questions
    if any(t.dep_ == "mark" and t.lower_ in SUBORD for t in main_all): return None  # nested subordination
    STRONG = {"because", "although", "though", "while", "whereas", "unless", "until", "if", "when", "whenever"}
    if any(t.lower_ in STRONG for t in adv.subtree if t.i != adv.left_edge.i): return None
    if any(t.lower_ in STRONG for t in main_all): return None
    if any(c.dep_ in ("ccomp", "csubj") for c in root.children): return None   # clause attached to a 'that'-clause
    if not any(c.dep_ in COMPLEMENT for c in root.children if not (adv.left_edge.i <= c.i <= adv.right_edge.i)):
        if root.lemma_ == "be" or root.lemma_ in NEEDS_OBJ: return None         # 'I much prefer.' / 'It is.'
    if adv.pos_ not in ("VERB", "AUX") or not any(c.dep_ in ("nsubj", "nsubjpass", "expl") for c in adv.children):
        return None                                      # subordinate clause must be finite with a subject
    l, r = adv.left_edge.i, adv.right_edge.i
    mark = doc[l].lower_
    # --- subordinate clause trailing: 'main [,] sub'  -> 'Sub, main'
    if r == len(doc) - 1 and l > 1:
        if subj.left_edge.i != 0: return None            # main clause must open with its subject (no 'In conclusion ...')
        main_toks = [t for t in doc if t.i < l and not (t.i == l - 1 and t.text == ",")]
        if len(main_toks) < 3 or len(list(adv.subtree)) < 3: return None
        sub_toks = [t for t in doc if l <= t.i <= r]
        if _has_dangling_pronoun(sub_toks, main_toks): return None
        main = body[:doc[l].idx].rstrip(" ,;")
        sub = body[doc[l].idx:]
        if len(main.split()) < 3: return None
        return upper_first(sub) + ", " + lower_first(main, doc[0]) + punct, "to_front"
    # --- subordinate clause leading: 'Sub, main' -> 'Main [,] sub'
    if l == 0 and r < len(doc) - 2 and doc[r + 1].text == ",":
        sub = body[:doc[r].idx + len(doc[r].text)]
        main = body[doc[r + 1].idx + 1:].strip()
        main_toks = [t for t in doc if t.i > r + 1]
        if subj.left_edge.i != r + 2: return None
        sub_toks = [t for t in doc if t.i <= r]
        if len(main.split()) < 3 or len(sub.split()) < 3: return None
        if _has_dangling_pronoun(main_toks, sub_toks): return None
        if main_toks and main_toks[0].lower_ in ("then", "so", "but", "and", "yet"): return None
        if "," in main: return None                      # moved clause would attach to the wrong phrase
        sep = ", " if mark in COMMA_TRAILING else " "
        return upper_first(main) + sep + lower_first(sub, doc[0]) + punct, "to_back"
    return None
