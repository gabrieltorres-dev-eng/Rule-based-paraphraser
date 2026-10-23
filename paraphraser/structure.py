"""Structural divergence: dependency-tree edit distance, sentence-order distortion, burstiness; plus the per-document
structure targets and the random transform 'recipes' that make the rewrite non-deterministic.

* Tree distance uses LEXICON-FREE labels (dependency relation | POS), so a pure synonym swap scores 0 and moving a clause,
  changing voice or splitting a sentence scores > 0. It is Selkow's top-down tree edit distance (children are aligned as an
  ordered sequence; moving a subtree costs delete + insert).
* Input and output sentences are aligned by content-stem overlap, so splits, merges and reorders are measured correctly.
* Targets are drawn per document from the distribution of structural distance between real HUMAN paraphrase pairs
  (data/profiles/structure.json, built by scripts/build_structure_profile.py)."""
from __future__ import annotations
import json, re
import numpy as np
from .routing.util import parse, warm
from .profile import PROFILE_DIR

FAMILIES = ["clause", "voice", "phrase", "coord", "list", "inner", "split", "complex", "para", "sent_order", "burst",
            "merge", "relcl", "opening"]
DEFAULT_TED_Q = [round(float(x), 4) for x in np.linspace(0.03, 0.42, 101)]
DEFAULT_BURST = {"p05": -0.42, "p10": -0.36, "p50": -0.24, "p90": -0.10}     # replaced by measured per-register bands when structure.json exists

# ------------------------------------------------------------------ trees
def _tree(tok):
    kids = tuple(_tree(c) for c in tok.children if c.dep_ != "punct")
    return (f"{tok.dep_}|{tok.pos_}", kids, 1 + sum(k[2] for k in kids))

def _dist(a, b, memo):
    k = (id(a), id(b))
    r = memo.get(k)
    if r is not None: return r
    ca, cb = a[1], b[1]
    m, n = len(ca), len(cb)
    prev = [0] * (n + 1)
    for j in range(1, n + 1): prev[j] = prev[j - 1] + cb[j - 1][2]
    for i in range(1, m + 1):
        cur = [prev[0] + ca[i - 1][2]] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + ca[i - 1][2], cur[j - 1] + cb[j - 1][2], prev[j - 1] + _dist(ca[i - 1], cb[j - 1], memo))
        prev = cur
    r = (0 if a[0] == b[0] else 1) + prev[n]
    memo[k] = r
    return r

def _forest(trees):
    return ("FOREST", tuple(trees), 1 + sum(t[2] for t in trees))

class _Info:
    __slots__ = ("tree", "size", "stems", "n")
    def __init__(self, tree, stems, n):
        self.tree, self.size, self.stems, self.n = tree, tree[2], stems, n

_INFO_CACHE: dict = {}

def infos(text: str):
    """Sentence infos (tree, content stems, word count) for a text; cached by text."""
    hit = _INFO_CACHE.get(text)
    if hit is not None: return hit
    paras = [p for p in re.split(r"\n+", text) if p.strip()]
    warm(paras)
    out = []
    for para in paras:
        d = parse(para)
        for s in d.sents:
            words = [t for t in s if t.is_alpha]
            if not words: continue
            stems = {t.lemma_.lower()[:5] for t in words if not t.is_stop and len(t.lemma_) >= 3}
            if len(stems) < 2: stems = stems | {t.lemma_.lower()[:5] for t in words}
            out.append(_Info(_tree(s.root), stems, len(words)))
    if len(_INFO_CACHE) > 200: _INFO_CACHE.clear()
    _INFO_CACHE[text] = out
    return out

def _overlap(a: _Info, b: _Info) -> float:
    shared = len(a.stems & b.stems)
    den = min(len(a.stems), len(b.stems))
    if den == 0 or (shared < 2 and den > 2): return 0.0
    return shared / den

def compare(src, out) -> dict:
    """src/out: text or list of _Info. Returns {'ted': 0..1, 'order': 0..1, 'matched': fraction of output sentences aligned}."""
    S = infos(src) if isinstance(src, str) else src
    O = infos(out) if isinstance(out, str) else out
    ns, no = len(S), len(O)
    if not ns or not no: return {"ted": 1.0, "order": 0.0, "matched": 0.0}
    parent = list(range(ns + no))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    best_src = [None] * no
    for j, o in enumerate(O):
        sc = [(_overlap(o, s), i) for i, s in enumerate(S)]
        top = max(sc, key=lambda t: (t[0], len(O[j].stems & S[t[1]].stems) / max(len(O[j].stems | S[t[1]].stems), 1), -abs(t[1] - j * ns / no)))
        if top[0] < 0.3: continue
        best_src[j] = top[1]
        parent[find(ns + j)] = find(top[1])
        for ov, i in sc:                                   # a merged output also absorbs its other source sentences
            if i != top[1] and ov >= 0.7 and len(S[i].stems & o.stems) >= 3 and len(o.stems & S[top[1]].stems) >= 3 and ov >= 0.85 * top[0]:
                parent[find(i)] = find(ns + j)
    groups: dict = {}
    for i in range(ns): groups.setdefault(find(i), ([], []))[0].append(i)
    for j in range(no): groups.setdefault(find(ns + j), ([], []))[1].append(j)
    memo: dict = {}
    keep: list = []                                      # forests must stay alive: the memo is keyed by id()
    total = 0
    for srcs, outs in groups.values():
        if not srcs: total += sum(O[j].size for j in outs); continue
        if not outs: total += sum(S[i].size for i in srcs); continue
        fa, fb = _forest([S[i].tree for i in srcs]), _forest([O[j].tree for j in outs])
        keep.append((fa, fb))
        total += _dist(fa, fb, memo)
    denom = sum(s.size for s in S) + sum(o.size for o in O)
    ted = min(1.0, total / max(denom, 1))
    seq = [b for b in best_src if b is not None]
    k = len(seq)
    inv = sum(1 for x in range(k) for y in range(x + 1, k) if seq[x] > seq[y])
    order = inv / (k * (k - 1) / 2) if k >= 2 else 0.0
    return {"ted": round(ted, 4), "order": round(order, 4), "matched": round(sum(b is not None for b in best_src) / no, 3)}

# ------------------------------------------------------------------ burstiness
def burstiness(lengths) -> float:
    """Goh-Barabasi burstiness (sigma - mu) / (sigma + mu) of sentence lengths: -1 regular, 0 random, +1 bursty."""
    a = np.array(lengths, dtype=float)
    if len(a) < 2: return 0.0
    mu, sd = a.mean(), a.std()
    return float((sd - mu) / (sd + mu)) if sd + mu else 0.0

def text_burstiness(text: str) -> tuple[float, int]:
    lens = [i.n for i in infos(text)]
    return burstiness(lens), len(lens)

# ------------------------------------------------------------------ profile, targets, recipes
_PROFILE: dict = {}

def load_profile() -> dict:
    if not _PROFILE:
        p = PROFILE_DIR / "structure.json"
        _PROFILE.update(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"default": True})
    return _PROFILE

def reload_profile():
    _PROFILE.clear()

def draw_targets(rng, register: str, min_ted=None, min_burst=None) -> dict:
    """Per-document targets. The TED target is drawn from the human-paraphrase distance distribution, so documents differ."""
    prof = load_profile()
    q = np.array(prof.get("ted_quantiles") or DEFAULT_TED_Q)
    u = float(np.clip(rng.beta(2.4, 1.6), 0.45, 0.98))             # which human-like quantile this document aims for
    target = float(np.quantile(q, u))
    floor = float(np.quantile(q, 0.20)) if min_ted is None else float(min_ted)      # human p20: the level current transforms can reach
    b = (prof.get("burst") or {}).get(register) or DEFAULT_BURST
    burst_floor = float(b["p10"] + rng.random() * 0.4 * (b["p50"] - b["p10"])) if min_burst is None else float(min_burst)
    return {"quantile": round(u, 3), "target": round(max(target, floor), 4), "floor": round(floor, 4),
            "burst_floor": round(burst_floor, 4), "profile": "measured" if not prof.get("default") else "default"}

def draw_recipe(rng, boost: float = 1.0) -> dict:
    """A random mix of transform families for ONE candidate: some families are off, the rest get random weights."""
    w = {f: (0.0 if rng.random() < 0.22 else float(np.clip(rng.lognormal(0.0, 0.45), 0.35, 1.7))) for f in FAMILIES}
    while sum(v > 0 for v in w.values()) < 4:
        w[FAMILIES[int(rng.integers(len(FAMILIES)))]] = float(np.clip(rng.lognormal(0.0, 0.45), 0.35, 1.7))
    w["_boost"] = boost
    return w

def repair_burstiness(doc, res, rng, floor: float, max_steps: int = 8):
    """Targeted fix when ONLY the burstiness floor fails: split the longest splittable sentence / fuse the shortest adjacent
    pair (the same guarded operations the length sculptor uses) until burstiness reaches the floor."""
    from .routing.length import split_options, fuse_option
    from .routing.util import n_words
    from .routing.length import FALLBACK_JOINERS
    joiners = [r["text"] for r in res.templates.index.get(("coordination", doc.register), [])
               if r["text"].strip(", ;").strip() in ("and", "but", "so", "yet", "or") and r.get("docs", 0) >= 3] or FALLBACK_JOINERS
    steps = 0
    for _ in range(max_steps):
        b, n = text_burstiness(doc.text())
        if n < 6 or b >= floor: break
        lens = [(n_words(doc.restore(s)), pi, si) for pi, p in enumerate(doc.paragraphs) for si, s in enumerate(p.sentences)]
        done = False
        for L, pi, si in sorted(lens, reverse=True)[:6]:                 # long sentences -> split (makes lengths more varied)
            opts = list(split_options(doc.paragraphs[pi].sentences[si]))
            if opts and L >= 22:
                a, c = opts[int(rng.integers(len(opts)))]
                doc.paragraphs[pi].sentences[si:si + 1] = [a, c]; doc.note("structure", "burst_split", pi, si, a, c); done = True; break
        if not done:
            for L, pi, si in sorted(lens)[:6]:                              # short neighbours -> fuse
                P = doc.paragraphs[pi].sentences
                if si + 1 < len(P):
                    f = fuse_option(P[si], P[si + 1], joiners, rng, doc.restore)
                    if f: P[si:si + 2] = [f]; doc.note("structure", "burst_fuse", pi, si, P[si][:40], f[:40]); done = True; break
        if not done: break
        steps += 1
    return steps
