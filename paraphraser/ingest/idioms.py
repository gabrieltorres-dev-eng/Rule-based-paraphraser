"""Longest-match-first idiom replacement. Replacements are locked (never re-transformed).
Equivalents are chosen by register compatibility, human frequency in the register, and a rotation
counter so no single equivalent becomes a fingerprint across documents."""
from collections import Counter
from ..resources import TOKEN
from ..doc import PH_RE

USAGE: Counter = Counter()     # process-wide rotation counter (hygiene.rotation persists it)
MIN_SCORE = 3.0

def _case(src: str, dst: str) -> str:
    if src[:1].isupper() and not (len(src) > 1 and src.isupper()): return dst[:1].upper() + dst[1:]
    if len(src) > 1 and src.isupper(): return dst.upper()
    return dst

def choose(entries, register, rng):
    cands = [e for e in entries if e["score"] >= MIN_SCORE]
    if not cands: return None
    w = []
    for e in cands:
        f = e["freqs"].get(register, 0.0) if e["freqs"] else 0.0
        w.append((1.0 + e["score"] + min(f, 50) / 50) / (1.0 + USAGE[e["text"].lower()]))
    tot = sum(w)
    return cands[int(rng.choice(len(cands), p=[x / tot for x in w]))]

SWAP_EVERY = 8        # one idiom swap per ~N words (lower = more swaps)
PHRASE_CAP = 4        # max times one source phrase is swapped per document

def replace_idioms(doc, resources, rng, max_per_doc=None):
    idi = resources.idioms
    if idi.n_entries == 0: return
    done = 0
    nwords = sum(len(x.split()) for q in doc.paragraphs for x in q.sentences)
    budget = max_per_doc or max(3, nwords // SWAP_EVERY)
    seen = {}                                         # source phrase -> uses (each phrase swapped at most PHRASE_CAP times per document)
    for pi, p in enumerate(doc.paragraphs):
        if len(p.sentences) == 1 and len(p.sentences[0].split()) <= 16 and not p.sentences[0].rstrip()[-1:] in ".!":
            continue                                   # headings / titles are left alone
        for si, s in enumerate(p.sentences):
            toks = list(TOKEN.finditer(s))
            lower = [t.group(0).lower() for t in toks]
            out, i, last = [], 0, 0
            while i < len(toks):
                if PH_RE.fullmatch(toks[i].group(0)) or not toks[i].group(0)[0].isalpha():
                    i += 1; continue
                m = idi.match(lower, i)
                if not m: i += 1; continue
                n, entries = m
                if any(PH_RE.fullmatch(t.group(0)) for t in toks[i:i + n]):
                    i += 1; continue
                key = " ".join(lower[i:i + n])
                if done >= budget or seen.get(key, 0) >= PHRASE_CAP:
                    i += n; continue
                entry = choose(idi.allowed(entries, doc.register), doc.register, rng)
                if entry is None: i += 1; continue
                a, b = toks[i].start(), toks[i + n - 1].end()
                new = _case(s[a:b], entry["text"])
                ph = doc.lock(new, "idiom")
                out.append(s[last:a] + ph); last = b
                USAGE[entry["text"].lower()] += 1; seen[key] = seen.get(key, 0) + 1
                i += n; done += 1
            if out:
                new_s = "".join(out) + s[last:]
                doc.note("ingest", "idiom", pi, si, s, new_s)
                p.sentences[si] = new_s
