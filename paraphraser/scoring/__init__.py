"""Phase 8: statistical scoring gate. Scorers only accept/reject candidates; they never write text."""
from __future__ import annotations
import difflib, math, re
from collections import Counter
import numpy as np
from scipy.stats import wasserstein_distance
from ..doc import get_nlp
from ..calibration.measure import punct_entropy

BINS, BW = 10, 5

def _analyse(text: str):
    """One cached parse per paragraph (unchanged paragraphs are free across candidates): sentence lengths, words, stopwords."""
    from ..routing.util import parse, warm
    paras = [p for p in re.split(r"\n+", text) if p.strip()]
    warm(paras)
    lens, words, stop = [], [], 0
    for para in paras:
        d = parse(para)
        for s in d.sents:
            n = sum(1 for t in s if t.is_alpha)
            if n: lens.append(n)
        for t in d:
            if t.is_alpha:
                words.append(t.lower_); stop += t.is_stop
    return lens, words, stop

def metrics(text: str, profile, res=None) -> dict:
    lens, words, stop = _analyse(text)
    hist = np.bincount(np.minimum(np.array(lens or [1]) // BW, BINS - 1), minlength=BINS) / max(len(lens), 1)
    centers = np.arange(BINS)
    m = {
        "hist_wasserstein": float(wasserstein_distance(centers, centers, hist, np.array(profile["sent_len"]["hist"]))),
        "sent_len_var": float(np.var(lens)) if len(lens) > 2 else 0.0,
        "ttr": len(set(words[:500])) / max(len(words[:500]), 1),
        "stopword_ratio": stop / max(len(words), 1),
        "punct_entropy": punct_entropy(text),
        "n_sentences": len(lens),
    }
    if res is not None:
        pp = res.perplexity(text)
        if pp is not None: m["perplexity"] = pp
    return m

def change_ratio(src: str, out: str) -> float:
    a, b = src.split(), out.split()
    return 1 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def violations(m: dict, profile, relax: dict | None = None, hist_max=0.9) -> dict:
    """Per-gate violation size (0 = pass). Bands are human p5-p95; `relax` widens named bands."""
    relax = relax or {}
    v = {}
    if m.get("n_sentences", 99) < 8:       # per-document statistics are meaningless on very short texts: only perplexity is gated
        if "perplexity" in m and profile.band("perplexity"):
            lo, hi = profile.band("perplexity"); x = m["perplexity"]
            v["perplexity"] = 0.0 if lo <= x <= hi else (lo - x if x < lo else x - hi) / max(hi - lo, 1e-9)
        v["hist_wasserstein"] = 0.0
        return v
    for name in ("sent_len_var", "ttr", "stopword_ratio", "punct_entropy", "perplexity"):
        band = profile.band(name)
        if not band or name not in m: continue
        lo, hi = band; w = max(hi - lo, 1e-9) * relax.get(name, 1.0)
        mid = (lo + hi) / 2; lo, hi = mid - w / 2 if relax.get(name) else lo, mid + w / 2 if relax.get(name) else hi
        x = m[name]
        v[name] = 0.0 if lo <= x <= hi else (lo - x if x < lo else x - hi) / max(hi - lo, 1e-9)
    lim = hist_max * relax.get("hist_wasserstein", 1.0)
    v["hist_wasserstein"] = max(0.0, m["hist_wasserstein"] - lim) / max(lim, 1e-9)
    return v

def total_score(m, vio, change):
    # lower is better; reward real change (but capped) so identical text never wins
    return sum(vio.values()) * 3 + (m["hist_wasserstein"] if m.get("n_sentences", 99) >= 8 else 0.0) - 0.5 * min(change, 0.5)

def calibrate_perplexity(register: str, res, profile, max_docs=150):
    from ..profile import ROOT
    files = sorted((ROOT / "data" / "corpus" / register).glob("*.txt"))[:max_docs]
    pp = []
    for f in files:
        t = f.read_text(encoding="utf-8", errors="ignore")[:4000]
        v = res.perplexity(t)
        if v: pp.append(v)
    if len(pp) >= 10:
        profile.d["bands"]["perplexity"] = [round(float(np.percentile(pp, 5)), 2), round(float(np.percentile(pp, 95)), 2)]
        profile.save(register)
    return profile.band("perplexity")
