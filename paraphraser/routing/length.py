"""Sentence-length sculpting: resample the document's sentence-length histogram toward the human
profile by splitting sentences at clause boundaries or fusing short neighbours with coordination."""
import numpy as np
from .util import parse, split_final, lower_first, upper_first, n_words
from ..doc import PH_RE

BINS, BW = 10, 5
KEEP_CONN = {"but": "But", "so": "So", "yet": "Yet"}
FUSE_BLOCK = {"however", "but", "so", "thus", "therefore", "also", "first", "second", "third", "finally",
              "then", "moreover", "furthermore", "this", "that", "these", "those", "it", "he", "she", "they",
              "and", "or", "next", "meanwhile", "instead", "still", "yet", "otherwise", "nevertheless"}
SUBMARK = {"while", "although", "though", "because", "since", "if", "when", "whereas", "unless", "until", "after", "before", "as", "once", "whenever", "so"}
FALLBACK_JOINERS = [", and ", "; "]

def bin_of(n): return min(n // BW, BINS - 1)

def hist(lengths):
    h = np.zeros(BINS)
    for n in lengths: h[bin_of(n)] += 1
    return h

def dist(counts, target_frac):
    tot = counts.sum()
    return float(np.abs(counts / tot - target_frac).sum()) if tot else 0.0

def split_options(text):
    """Yield (first, second) sentence splits at clause boundaries."""
    body, punct = split_final(text)
    if not punct or punct[0] not in ".!": return
    doc = parse(body)
    if len(list(doc.sents)) != 1: return
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"): return
    for t in root.children:
        if t.dep_ == "conj" and t.pos_ in ("VERB", "AUX") and any(c.dep_ in ("nsubj", "nsubjpass") for c in t.children):
            ccs = [c for c in root.children if c.dep_ == "cc" and c.i < t.left_edge.i]
            cut = t.left_edge.i
            if ccs:
                cc = max(ccs, key=lambda c: c.i); cut = cc.i
                lead = KEEP_CONN.get(cc.lower_, "")
            else:
                lead = ""
            a = body[: doc[cut].idx].rstrip(" ,;")
            rest = body[doc[t.left_edge.i].idx:]
            second = (lead + " " + rest) if lead else upper_first(rest)
            if rest.split() and rest.split()[0].lower().strip(",") in SUBMARK and not lead: continue      # would leave a subordinate fragment
            if n_words(a) >= 4 and n_words(rest) >= 4:
                yield a + punct, second.strip() + punct
    for i, tok in enumerate(doc):
        if tok.text == ";" and 4 <= i < len(doc) - 4:
            a, b = body[: tok.idx].rstrip(), body[tok.idx + 1:].strip()
            if n_words(a) >= 4 and n_words(b) >= 4:
                yield a + punct, upper_first(b) + punct

def fuse_option(s1, s2, joiners, rng, restore=lambda x: x):
    b1, p1 = split_final(s1); b2, p2 = split_final(s2)
    if p1 != "." or p2 not in (".", "!") or "?" in s1 + s2: return None
    first = b2.split()[0].lower().strip(",") if b2.split() else ""
    if first in FUSE_BLOCK or PH_RE.fullmatch(b1.strip()) or PH_RE.fullmatch(b2.strip()): return None
    if first in ("other", "another", "soon", "ultimately", "later", "eventually") or (len(b2.split()) > 1 and b2.split()[0].endswith(",")): return None
    w1, w2 = [w.lower().strip(",") for w in restore(b1).split()[:2]], [w.lower().strip(",") for w in restore(b2).split()[:2]]
    if (w1[:1] == w2[:1] and w1[0] not in ("the", "a", "an")) or (len(w1) == 2 and w1 == w2): return None            # 'X was ..., and X was ...': same opening repeated
    joiners = [j for j in joiners if j.startswith((",", ";")) and j.strip(" ,;") in ("and", "")] or [", and "]
    j = joiners[int(rng.integers(len(joiners)))]
    doc2 = parse(b2)
    return b1 + j + lower_first(b2, doc2[0]) + p2

def sculpt(doc, profile, res, rng, max_frac=0.30):
    target = np.array(profile["sent_len"]["hist"], dtype=float)
    target = target / target.sum()
    joiners = [r["text"] for r in res.templates.index.get(("coordination", doc.register), [])
               if r["text"].strip(", ;").strip() in ("and", "but", "so", "yet", "or", "") and r.get("docs", 0) >= 3] or FALLBACK_JOINERS
    joiners = [j if j.startswith((",", ";")) else j for j in joiners]
    changed = 0
    budget = max(1, int(max_frac * sum(len(p.sentences) for p in doc.paragraphs)))
    for _ in range(budget):
        lens = [[n_words(s) for s in p.sentences] for p in doc.paragraphs]
        flat = [n for l in lens for n in l]
        counts = hist(flat); base = dist(counts, target)
        best = []   # (gain, kind, pi, si, payload)
        for pi, p in enumerate(doc.paragraphs):
            for si, s in enumerate(p.sentences):
                n = lens[pi][si]
                for a, b in split_options(s) if n >= 10 else ():
                    c = counts.copy(); c[bin_of(n)] -= 1; c[bin_of(n_words(a))] += 1; c[bin_of(n_words(b))] += 1
                    best.append((base - dist(c, target), "split", pi, si, (a, b)))
                if si + 1 < len(p.sentences) and n <= 14 and lens[pi][si + 1] <= 14:
                    f = fuse_option(s, p.sentences[si + 1], joiners, rng, doc.restore)
                    if f:
                        c = counts.copy(); c[bin_of(n)] -= 1; c[bin_of(lens[pi][si + 1])] -= 1; c[bin_of(n_words(f))] += 1
                        best.append((base - dist(c, target), "fuse", pi, si, f))
        best = [b for b in best if b[0] > 1e-9]
        if not best: break
        best.sort(key=lambda x: -x[0])
        top = best[:3]
        gain, kind, pi, si, pay = top[int(rng.integers(len(top)))]
        p = doc.paragraphs[pi]
        if kind == "split":
            doc.note("routing", "split", pi, si, p.sentences[si], " | ".join(pay))
            p.sentences[si:si + 1] = list(pay)
        else:
            doc.note("routing", "fuse", pi, si, p.sentences[si] + " " + p.sentences[si + 1], pay)
            p.sentences[si:si + 2] = [pay]
        changed += 1
    return changed
