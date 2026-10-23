"""Paragraph-level operations: body swapping, conclusion relocation, paragraph-length normalisation."""
import re
import numpy as np
from .discourse import is_linked, starts_with, first_words, ANAPHORIC, CONCL_CUES, is_title
from ..calibration.measure import thesis_index
from ..routing.util import parse, lower_first, upper_first, n_words
from ..doc import PH_RE
from .. import intensity

SUMMARY_MARK = re.compile(r"^\s*(in conclusion|to conclude|in summary|to sum up|to summarize|all in all|overall|in short|"
                          r"ultimately|thus|therefore|finally|so)\b[,:]?\s*", re.I)
BAD_MOVE_START = ANAPHORIC | {"however", "but", "and", "so", "also", "then", "yet", "still", "instead", "moreover",
                              "furthermore", "additionally", "consequently", "thus", "therefore", "meanwhile", "otherwise",
                              "other", "another", "similarly", "likewise", "ultimately", "soon", "finally", "later", "eventually"}
PARA_BINS = [0, 25, 50, 75, 100, 150, 200, 300, 10**6]

def _bad_start(s: str) -> bool:
    return first_words(s, 1) in BAD_MOVE_START

# ------------------------------------------------------------------ body swapping
def swap_bodies(doc, profile, rng, arc_flag=False):
    base = float(profile["rates"].get("para_swap", 0.10)) * (1.5 if arc_flag else 1.0)
    if intensity.aggressive(): base = intensity.p('para', 0.85)
    P = doc.paragraphs
    i, swaps = 0, 0
    while i < len(P) - 1:
        a, b = P[i], P[i + 1]
        ok = (a.role in ("background", "elaboration", "solution") and b.role in ("background", "elaboration", "solution")
              and not is_linked(a.sentences[0]) and not is_linked(b.sentences[0])
              and not a.sentences[-1].rstrip().endswith(":")
              and not (i + 2 < len(P) and P[i + 2].role != "conclusion" and is_linked(P[i + 2].sentences[0]))
              and not any(PH_RE.fullmatch(s.strip()) for s in (a.sentences[0], b.sentences[0])))
        # an intro that feeds a linked first body paragraph must stay adjacent
        if ok and i > 0 and P[i - 1].role == "intro" and is_linked(b.sentences[0]): ok = False
        if ok and rng.random() < base:
            P[i], P[i + 1] = b, a
            doc.note("macro", "swap_bodies", i, 0, a.sentences[0][:60], b.sentences[0][:60])
            swaps += 1; i += 2
        else:
            i += 1
    return swaps

# ------------------------------------------------------------------ conclusion relocation
def relocate_conclusions(doc, profile, res, rng):
    target = profile.rate("early_thesis")
    elig = [pi for pi, p in enumerate(doc.paragraphs)
            if p.role != "title" and len(p.sentences) >= 4 and not any(PH_RE.fullmatch(x.strip()) for x in p.sentences)]
    if not elig: return 0
    # current early fraction using the same heuristic as calibration
    early = 0
    for pi in elig:
        sents = [parse(x).sents.__next__() for x in doc.paragraphs[pi].sentences]
        if thesis_index(sents) <= 1: early += 1
    cur = early / len(elig)
    p_move = float(np.clip((target - cur) / max(1 - cur, 0.05), 0, 0.7))
    if intensity.aggressive(): p_move = intensity.p('para', 0.9)
    moved = 0
    for pi in elig:
        if rng.random() >= p_move: continue
        p = doc.paragraphs[pi]; s = p.sentences
        last = s[-1]
        if _bad_start(last) and not SUMMARY_MARK.match(last): continue
        if _bad_start(s[1]) and not SUMMARY_MARK.match(s[1]): continue      # sentence 2 would lose its antecedent
        if last.rstrip()[-1:] not in ".!": continue
        body = last
        m = SUMMARY_MARK.match(body)
        if m:
            rest = body[m.end():]
            if len(rest.split()) < 5: continue
            if _bad_start(rest): continue
            body = upper_first(rest)
            tpl = None
            if rng.random() < 0.5:
                tpl = res.templates.pick("transition", doc.register, rng, doc.used_templates)
                if tpl and not re.match(r"^(in short|overall|basically|in the end|all in all)", tpl["text"], re.I): tpl = None
            if tpl: body = tpl["text"][0].upper() + tpl["text"][1:].rstrip(",") + ", " + lower_first(body, parse(body)[0])
        new = s[:1] + [body] + s[1:-1]
        doc.note("macro", "relocate_conclusion", pi, len(s) - 1, last, body)
        p.sentences = new; moved += 1
    return moved

# ------------------------------------------------------------------ paragraph length normalisation
def _pbin(w): 
    for i in range(len(PARA_BINS) - 1):
        if PARA_BINS[i] <= w < PARA_BINS[i + 1]: return i
    return len(PARA_BINS) - 2

def _wlen(p): return sum(n_words(s) for s in p.sentences)

def _dist(counts, target):
    t = counts.sum()
    return float(np.abs(counts / t - target).sum()) if t else 0.0

def _weak_cut(sentences):
    """Index (1..n-1) of the weakest boundary: low lexical overlap, discourse-marker opener preferred."""
    sets = [{w.lower() for w in re.findall(r"[A-Za-z]{4,}", s)} for s in sentences]
    best, bi = 9.9, None
    for i in range(2, len(sentences) - 1):                # keep >=2 sentences each side
        a, b = sets[i - 1], sets[i]
        j = len(a & b) / max(len(a | b), 1)
        if starts_with(sentences[i], ("however", "another", "secondly", "finally", "moreover", "in addition", "also", "next", "on the other hand")): j -= 0.3
        if first_words(sentences[i], 1) in ANAPHORIC: j += 0.5          # don't cut before a pronoun opener
        if j < best: best, bi = j, i
    return bi

ADDITIVE_OPEN = ("also", "furthermore", "moreover", "in addition", "additionally", "besides", "plus", "likewise", "similarly", "and", "but")

def attach_linked_openers(doc):
    """A paragraph may not open with an additive connective ('Also, ...'); pull it back into the previous paragraph."""
    P = doc.paragraphs; i = 1; n = 0
    while i < len(P):
        s0 = P[i].sentences[0] if P[i].sentences else ""
        if not is_title(P[i - 1]) and not is_title(P[i]) and first_words(s0, 1).strip(",") in ADDITIVE_OPEN:
            P[i - 1].sentences += P[i].sentences; del P[i]; n += 1
        else:
            i += 1
    return n

def normalize_paragraphs(doc, profile, rng, max_frac=0.3):
    wh = profile["para"].get("words_hist")
    if not wh: return 0
    target = np.array(wh, dtype=float); target /= target.sum()
    ops = 0
    for _ in range(max(1, int(max_frac * len(doc.paragraphs)))):
        P = doc.paragraphs
        lens = [_wlen(p) for p in P]
        counts = np.zeros(len(target))
        for p, w in zip(P, lens):
            if not is_title(p): counts[_pbin(w)] += 1
        base = _dist(counts, target); best = []
        for i, p in enumerate(P):
            if is_title(p): continue
            if len(p.sentences) >= 5:
                cut = _weak_cut(p.sentences)
                if cut:
                    a, b = sum(n_words(s) for s in p.sentences[:cut]), sum(n_words(s) for s in p.sentences[cut:])
                    c = counts.copy(); c[_pbin(lens[i])] -= 1; c[_pbin(a)] += 1; c[_pbin(b)] += 1
                    best.append((base - _dist(c, target), "split", i, cut))
            if i + 1 < len(P) and not is_title(P[i + 1]) and P[i].role not in ("intro",) and P[i + 1].role != "conclusion" \
               and P[i].role != "conclusion" and not is_linked(P[i + 1].sentences[0]) and lens[i] < 60 and lens[i + 1] < 60:
                c = counts.copy(); c[_pbin(lens[i])] -= 1; c[_pbin(lens[i + 1])] -= 1; c[_pbin(lens[i] + lens[i + 1])] += 1
                best.append((base - _dist(c, target), "merge", i, None))
        best = [b for b in best if b[0] > 1e-9]
        if not best: break
        best.sort(key=lambda x: -x[0]); g, kind, i, cut = best[int(rng.integers(min(3, len(best))))]
        if kind == "split":
            p = P[i]; q = type(p)(p.sentences[cut:], p.role if p.role != "intro" else "elaboration"); p.sentences = p.sentences[:cut]
            P.insert(i + 1, q); doc.note("macro", "split_paragraph", i, cut, "", "")
        else:
            P[i].sentences += P[i + 1].sentences; del P[i + 1]; doc.note("macro", "merge_paragraphs", i, 0, "", "")
        ops += 1
    return ops


# ------------------------------------------------------------------ aggressive: swap adjacent independent sentences
def swap_sentences(doc, rng, p=0.5):
    """Swap neighbouring sentences inside a paragraph when neither depends on its neighbour (no pronoun/linker start).
    The first and last sentence of every paragraph stay put so the paragraph still opens and closes correctly."""
    n = 0
    for pi, para in enumerate(doc.paragraphs):
        S = para.sentences
        i = 1
        while i < len(S) - 2:
            a, b = S[i], S[i + 1]
            nxt = S[i + 2]
            ok = (not _bad_start(a) and not _bad_start(b) and not _bad_start(nxt) and not is_linked(a) and not is_linked(b)
                  and not S[i - 1].rstrip().endswith(":") and not a.rstrip().endswith(":")
                  and not any(PH_RE.fullmatch(x.strip()) for x in (a, b)))
            if ok and rng.random() < p:
                S[i], S[i + 1] = b, a
                doc.note("macro", "swap_sentences", pi, i, a[:60], b[:60]); n += 1; i += 2
            else:
                i += 1
    return n
