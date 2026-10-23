"""Aggressive-mode sentence splitting that adds a connective taken from the HUMAN template bank (never typed in code):
  A, and B.            ->  A. Also, B.            (additive connective)
  A, but B.            ->  A. However, B.         (adversative)
  A, so B.             ->  A. As a result, B.     (causal)
  A, although B.       ->  A. That said, B.       (concession)
  A, which does B.     ->  A. This does B.
  A, leading to B.     ->  A. This leads to B.
The connective is chosen from bank rows whose text belongs to the right cue family; the cue lists only CLASSIFY rows."""
from lemminflect import getInflection
from .util import parse, split_final, lower_first, upper_first, n_words
from ..doc import PH_RE
from ..glue import ok as glue_ok

ADD = {"also", "plus", "similarly", "in addition", "besides", "moreover", "furthermore", "likewise", "and"}
CAUSE = {"so", "as a result", "therefore", "thus", "consequently", "hence", "as a consequence"}
ADV = {"however", "but", "nevertheless", "that said", "on the other hand", "in contrast", "still", "even so", "that being said",
       "nonetheless", "granted", "then again", "yet", "of course"}
FAMILY = {"add": ADD, "cause": CAUSE, "adv": ADV}
NO_COMMA = {"but", "so", "yet", "and"}
CASUAL = {"plus", "so", "then again", "of course", "granted", "but"}              # kept out of formal registers
THIS_VERBS = {"lead", "allow", "make", "cause", "result", "enable", "help", "show", "mean", "give", "create", "provide",
              "reduce", "increase", "explain", "suggest", "indicate", "reflect", "produce", "reveal", "support", "limit", "improve"}
AUX3 = {"are": "is", "were": "was", "have": "has", "do": "does"}
BAD = '"*_“”()[]<>;:'

def pick_connective(res, kind, register, rng, used):
    fam = FAMILY[kind]
    formal = register in ("essay", "academic", "technical")
    rows = [r for r in res.templates.rows if r["function"] in ("transition", "counterexample", "concession")
            and r["text"].strip().lower() in fam and r["id"] not in used
            and not (formal and r["text"].strip().lower() in CASUAL)]
    if not rows: return None
    pref = [r for r in rows if r["register"] == register] or rows
    w = [max(r.get("docs", 1), 1) ** 0.5 for r in pref]
    r = pref[int(rng.choice(len(pref), p=[x / sum(w) for x in w]))]
    used.add(r["id"])
    return r["text"].strip()

def _lead(conn, rest_text, first_tok):
    c = upper_first(conn)
    return f"{c} {lower_first(rest_text, first_tok)}" if conn.lower() in NO_COMMA else f"{c}, {lower_first(rest_text, first_tok)}"

def split_sentence(s, res, rng, register, used):
    """Return (first, second, kind) or None."""
    body, punct = split_final(s)
    if not punct or punct[0] not in ".!" or any(ch in body for ch in BAD) or n_words(body) < 16: return None
    doc = parse(body)
    if len(list(doc.sents)) != 1: return None
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"): return None
    n = len(doc)
    # A) coordinated independent clause
    for cj in (c for c in root.children if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")):
        subj2 = [t for t in cj.children if t.dep_ in ("nsubj", "nsubjpass")]
        ccs = [c for c in root.children if c.dep_ == "cc" and c.i < cj.left_edge.i and c.lower_ in ("and", "but", "so", "yet")]
        if not subj2 or not ccs or cj.right_edge.i != n - 1: continue
        cc = max(ccs, key=lambda c: c.i)
        if subj2[0].left_edge.i != cc.i + 1: continue
        kind = {"and": "add", "but": "adv", "yet": "adv", "so": "cause"}[cc.lower_]
        a = body[: doc[cc.i].idx].rstrip(" ,"); rest = body[doc[cc.i + 1].idx:]
        if n_words(a) < 6 or n_words(rest) < 6: continue
        conn = pick_connective(res, kind, register, rng, used)
        if not conn: continue
        return a + punct, _lead(conn, rest, doc[cc.i + 1]) + punct, "split_" + kind
    # B) trailing non-restrictive 'which' clause
    for t in doc:
        if t.dep_ in ("relcl", "advcl", "ccomp") and t.right_edge.i == n - 1 and t.left_edge.i >= 1 and doc[t.left_edge.i].lower_ == "which" and doc[t.left_edge.i].dep_ == "nsubj" \
                and doc[t.left_edge.i - 1].text == ",":
            wh = t.left_edge
            first = body[: doc[wh.i - 1].idx].rstrip()
            rest_toks = [x for x in doc if x.i > wh.i]
            if n_words(first) < 6 or len(rest_toks) < 4: continue
            new_rest = body[doc[wh.i + 1].idx:]
            tok1 = doc[wh.i + 1]
            if tok1.tag_ == "VBP" and tok1.lemma_ not in ("be",):
                inf = getInflection(tok1.lemma_, "VBZ")
                if inf: new_rest = inf[0] + body[doc[wh.i + 1].idx + len(tok1.text):]
            elif tok1.lower_ in AUX3:
                new_rest = AUX3[tok1.lower_] + body[doc[wh.i + 1].idx + len(tok1.text):]
            if not glue_ok("this"): continue
            return first + punct, "This " + new_rest + punct, "split_which"
    # C) trailing participial result phrase ', leading to ...'
    for t in doc:
        if t.tag_ == "VBG" and t.lemma_ in THIS_VERBS and t.i >= 6 and doc[t.i - 1].text == "," and t.right_edge.i == n - 1 \
                and t.dep_ in ("advcl", "acl", "xcomp", "conj") and not any(c.dep_ in ("aux", "auxpass") for c in t.children):
            first = body[: doc[t.i - 1].idx].rstrip()
            inf = getInflection(t.lemma_, "VBZ")
            if not inf or n_words(first) < 6 or n_words(body[doc[t.i].idx:]) < 3: continue
            if not glue_ok("this"): continue
            return first + punct, "This " + inf[0] + body[doc[t.i].idx + len(t.text):] + punct, "split_this"
    # D) trailing concessive clause
    for t in doc:
        if t.dep_ == "advcl" and t.right_edge.i == n - 1 and (t.left_edge.lower_ in ("although", "though", "whereas") or (t.left_edge.lower_ == "while" and doc[t.left_edge.i - 1].text == ",")) and t.left_edge.i >= 6 \
                and any(c.dep_ in ("nsubj", "nsubjpass") for c in t.children):
            mark = t.left_edge; cut = mark.i - 1 if doc[mark.i - 1].text == "," else mark.i
            first = body[: doc[cut].idx].rstrip(" ,")
            rest = body[doc[mark.i + 1].idx:]
            if n_words(first) < 6 or n_words(rest) < 4: continue
            conn = pick_connective(res, "adv", register, rng, used)
            if not conn: continue
            return first + punct, _lead(conn, rest, doc[mark.i + 1]) + punct, "split_concession"
    return None

def split_long(doc, res, rng, p=0.7):
    """Split eligible long sentences; paragraphs are rebuilt in order."""
    n = 0
    for pi, para in enumerate(doc.paragraphs):
        out = []
        for si, s in enumerate(para.sentences):
            r = None
            if not PH_RE.fullmatch(s.strip()) and rng.random() < p:
                r = split_sentence(s, res, rng, doc.register, doc.used_templates)
            if r and doc.fresh(r[0]):
                doc.note("routing", r[2], pi, si, s, r[0] + " | " + r[1])
                out += [r[0], r[1]]; n += 1
            else:
                out.append(s)
        para.sentences = out
    return n
