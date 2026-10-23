"""Aggressive-mode complex restructuring. Every rewrite returns new text or None and is guarded.
  because_split:  'A because B.'            -> 'B. That is why A.'
  merge_this:     'A. This means B.'        -> 'A, which means B.'
  verb_to_noun:   'Researchers analyze the data.' -> 'Researchers carry out the analysis of the data.'
  appositive:     'Crundall, a researcher, found X.' -> 'A researcher, Crundall found X.'
Words come from the input, WordNet derivations and the closed-class glue words listed in GLUE (see README)."""
from lemminflect import getInflection
from nltk.corpus import wordnet as wn
from .util import parse, split_final, lower_first, upper_first, n_words
from ..doc import PH_RE
from ..glue import ok as glue_ok

GLUE = {"which", "that", "is", "why", "the", "an", "a", "of", "carry", "out"}   # closed-class function words, not content
MERGE_VERBS = {"means", "leads", "suggests", "shows", "results", "causes", "makes", "creates", "reduces", "increases",
               "explains", "indicates", "allows", "requires", "is", "was", "has", "can", "could", "will", "helps", "limits"}
BAD = '"*_“”()[]<>;:'

def _ok(text):
    return not any(ch in text for ch in BAD)

def because_split(s):
    body, punct = split_final(s)
    if not punct or punct[0] not in ".!" or not _ok(body): return None
    doc = parse(body + punct)                              # keep the period: the parse changes without it
    body_end = len(doc) - 2                                # index of the last body token
    if len(list(doc.sents)) != 1: return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"): return None
    for t in root.children:
        if t.dep_ != "advcl" or t.right_edge.i != body_end: continue
        mark = t.left_edge
        if mark.lower_ != "because" or mark.dep_ != "mark": continue
        cut = mark.i - 1 if doc[mark.i - 1].text == "," else mark.i
        a = body[: doc[cut].idx].rstrip(" ,")
        b = body[doc[mark.i + 1].idx:]
        if t.dep_ == "advcl" and not any(c.dep_ in ("nsubj", "nsubjpass") for c in t.children): return None
        if n_words(a) < 4 or n_words(b) < 4: return None
        a_first = doc[0]
        if not glue_ok("that is why"): return None
        return upper_first(b) + ". That is why " + lower_first(a, a_first) + punct
    return None

def merge_this(a_sent, b_sent):
    """Merge 'A.' + 'This means B.' into 'A, which means B.' Returns merged text or None."""
    a_body, a_p = split_final(a_sent)
    b_body, b_p = split_final(b_sent)
    if a_p != "." or b_p != "." or not _ok(a_body + b_body): return None
    words = b_body.split()
    if len(words) < 4 or words[0] != "This" or words[1].lower() not in MERGE_VERBS: return None
    if n_words(a_body) < 6 or PH_RE.fullmatch(a_body.strip()) or PH_RE.fullmatch(b_body.strip()): return None
    if a_body.rstrip()[-1:] in ":;": return None
    if not glue_ok("which"): return None
    return a_body.rstrip() + ", which " + " ".join(words[1:]) + "."

ACTION_SUFFIX = ("sis", "tion", "sion", "ment", "ance", "ence", "al")

def _derived_noun(lemma):
    """Action noun derived from the verb (WordNet derivation), preferring -sis/-tion/-ment forms ('analysis' over 'analyst')."""
    cands = []
    for syn in wn.synsets(lemma, pos=wn.VERB)[:1]:
        for l in syn.lemmas():
            if l.name().lower() != lemma: continue
            for d in l.derivationally_related_forms():
                nm = d.name().lower()
                if d.synset().pos() == "n" and "_" not in nm and nm.startswith(lemma[:4]) and nm != lemma:
                    cands.append(nm)
    cands.sort(key=lambda n: (0 if n.endswith(ACTION_SUFFIX) else 1, len(n)))
    return cands[0] if cands else None

def verb_to_noun(s):
    body, punct = split_final(s)
    if not punct or punct[0] not in ".!" or not _ok(body): return None
    doc = parse(body)
    if len(list(doc.sents)) != 1: return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ != "VERB" or root.tag_ not in ("VBZ", "VBP", "VBD"): return None
    if any(c.dep_ in ("aux", "auxpass", "neg", "prt", "agent", "ccomp", "xcomp", "dative") for c in root.children): return None
    subj = [c for c in root.children if c.dep_ == "nsubj"]
    obj = [c for c in root.children if c.dep_ == "dobj"]
    if len(subj) != 1 or len(obj) != 1: return None
    subj, obj = subj[0], obj[0]
    if subj.left_edge.i != 0 or root.i != subj.right_edge.i + 1: return None          # 'Subject VERB OBJECT' with nothing between
    if obj.i != root.i + 1 and obj.left_edge.i != root.i + 1: return None
    if any(t.text == "," for t in doc[obj.left_edge.i: obj.right_edge.i + 1]): return None
    if any(PH_RE.search(t.text) for t in doc[obj.left_edge.i: obj.right_edge.i + 1]): return None
    noun = _derived_noun(root.lemma_.lower())
    if not noun or root.lemma_.lower() in ("be", "have", "do", "get", "make", "take", "give"): return None
    verb = getInflection("carry", {"VBZ": "VBZ", "VBP": "VBP", "VBD": "VBD"}[root.tag_])
    if not verb or not glue_ok("carry out"): return None
    det = "the" if doc[obj.left_edge.i].lower_ == "the" else ("an" if noun[0] in "aeiou" else "a")
    obj_txt = body[doc[obj.left_edge.i].idx: doc[obj.right_edge.i].idx + len(doc[obj.right_edge.i].text)]
    if obj.left_edge.i == obj.i and obj.lower_ in ("it", "them", "him", "her", "me", "us", "you"): return None
    subj_txt = body[: doc[root.i].idx].rstrip()
    tail = body[doc[obj.right_edge.i].idx + len(doc[obj.right_edge.i].text):]
    if not _tail_ok(tail): return None
    if not glue_ok(det, "of"): return None
    return f"{subj_txt} {verb[0]} out {det} {noun} of {obj_txt}{tail}{punct}"

def appositive(s):
    body, punct = split_final(s)
    if not punct or punct[0] not in ".!" or not _ok(body): return None
    doc = parse(body)
    if len(list(doc.sents)) != 1: return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"): return None
    subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")]
    if len(subj) != 1 or subj[0].left_edge.i != 0: return None
    subj = subj[0]
    apps = [c for c in subj.children if c.dep_ == "appos"]
    if len(apps) != 1: return None
    ap = apps[0]
    l, r = ap.left_edge.i, ap.right_edge.i
    if doc[l - 1].text != "," or r + 1 >= len(doc) or doc[r + 1].text != ",": return None
    if subj.i != 0 or l != 2: return None                                            # 'Subj, appos, VERB'
    if any(t.text in ",;:" for t in doc[l: r + 1]) or n_words(body[doc[l].idx: doc[r].idx + len(doc[r].text)]) > 14: return None
    if PH_RE.search(body[doc[l].idx: doc[r].idx + len(doc[r].text)]): return None
    app_txt = body[doc[l].idx: doc[r].idx + len(doc[r].text)]
    subj_txt = body[: doc[subj.i].idx + len(doc[subj.i].text)]
    rest = body[doc[r + 1].idx + 1:].strip()
    if n_words(rest) < 4 or rest.split()[0].lower() in ("and", "but", "or", "so"): return None
    return upper_first(app_txt) + ", " + subj_txt + " " + rest + punct

TIME_TAIL = {"every", "each", "daily", "in", "on", "during", "at", "since", "throughout", "annually", "yearly", "per"}

def _tail_ok(tail: str) -> bool:
    """Trailing words may only be a plain time/place phrase; 'by/with/for/of/to' tails change what the noun governs."""
    w = [x.lower().strip(",.") for x in tail.split()]
    if not w: return True
    return w[0] in TIME_TAIL and not (set(w) & {"by", "with", "for", "of", "to", "and", "or", "that", "which"})

def apply_complex(doc, rng, p=0.7):
    """Run the complex rewrites on every eligible sentence (aggressive mode only)."""
    from ..doc import PH_RE as _P
    n = 0
    for pi, para in enumerate(doc.paragraphs):
        out = []
        S = para.sentences
        i = 0
        while i < len(S):
            s = S[i]
            if _P.fullmatch(s.strip()) or rng.random() >= p:
                out.append(s); i += 1; continue
            # 1. merge 'A.' + 'This means B.' into one sentence
            if i + 1 < len(S) and rng.random() < p:
                m = merge_this(s, S[i + 1])
                if m and doc.fresh(m):
                    doc.note("routing", "merge_this", pi, i, s + " " + S[i + 1], m)
                    out.append(m); i += 2; n += 1; continue
            # 2. 'A because B.' -> 'B. That is why A.'
            r = because_split(s)
            if r and doc.fresh(r):
                first, second = r.split(". That is why ", 1)[0] + ".", "That is why " + r.split(". That is why ", 1)[1]
                doc.note("routing", "because_split", pi, i, s, r)
                out += [first, second]; i += 1; n += 1; continue
            # 3. verb -> action noun
            v = verb_to_noun(s)
            if v and doc.fresh(v):
                doc.note("routing", "verb_to_noun", pi, i, s, v)
                out.append(v); i += 1; n += 1; continue
            # 4. fronted phrase (with commas) moved after the main clause
            f2 = front_to_back(s)
            if f2 and doc.fresh(f2):
                doc.note("routing", "front_to_back", pi, i, s, f2)
                out.append(f2); i += 1; n += 1; continue
            # 5. appositive moved to the front
            a = appositive(s)
            if a and doc.fresh(a):
                doc.note("routing", "appositive", pi, i, s, a)
                out.append(a); i += 1; n += 1; continue
            out.append(s); i += 1
        para.sentences = out
    return n


FRONT_OPEN = {"with", "in", "for", "during", "on", "at", "under", "among", "across", "despite", "given", "within", "from", "to"}

def front_to_back(s):
    """'With X, Y, etc., the main clause is Z.' -> 'The main clause is Z, with X, Y, etc.'
    Moves a fronted prepositional phrase (even one containing commas) after the main clause. Cuts at the comma that
    directly precedes the main clause's subject, so the last comma of the introduction is the boundary."""
    body, punct = split_final(s)
    if not punct or punct[0] not in ".!" or not _ok(body): return None
    doc = parse(body + punct)
    if len(list(doc.sents)) != 1: return None
    if doc[0].lower_ not in FRONT_OPEN: return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"): return None
    subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")]
    if len(subj) != 1: return None
    start = subj[0].left_edge.i
    if start < 4 or doc[start - 1].text != ",": return None
    intro = body[: doc[start - 1].idx].rstrip()
    main = body[doc[start].idx:].rstrip()
    if n_words(intro) < 3 or n_words(main) < 6 or PH_RE.search(main[:20]): return None
    if main.split()[0].lower() in ("it", "this", "that", "they", "these", "those"): return None      # must not strand a pronoun
    intro_low = lower_first(intro, doc[0])
    tail = "" if intro.endswith(".") else punct                                    # 'etc.' already ends the sentence
    return upper_first(main) + ", " + intro_low + tail
