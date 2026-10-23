"""Phase 0: measure human style statistics from a corpus of genuine human writing.

    python -m paraphraser.calibration.measure --register blog          # one register
    python -m paraphraser.calibration.measure                          # all with a corpus

Reads data/corpus/<register>/*.txt (one doc per file; paragraphs split on newlines).
Writes data/profiles/<register>.json (see profile.DEFAULT for the schema).
"""
from __future__ import annotations
import argparse, json, math, re
from collections import Counter
from pathlib import Path
import numpy as np
from ..doc import get_nlp
from ..profile import RegisterProfile, ROOT, REGISTERS, DEFAULT
from ..cues import HEDGE_CUES, CONCESSION_CUES, lead_cue

CORPUS = ROOT / "data" / "corpus"
BINS = 10          # sentence-length hist: 0-4, 5-9, ..., 45+
BW = 5

CONTRACTIONS = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't", "is not": "isn't", "are not": "aren't",
    "was not": "wasn't", "were not": "weren't", "have not": "haven't", "has not": "hasn't",
    "had not": "hadn't", "will not": "won't", "would not": "wouldn't", "could not": "couldn't",
    "should not": "shouldn't", "can not": "can't", "cannot": "can't", "it is": "it's", "that is": "that's",
    "there is": "there's", "he is": "he's", "she is": "she's", "i am": "i'm", "you are": "you're",
    "we are": "we're", "they are": "they're", "i have": "i've", "you have": "you've", "we have": "we've",
    "they have": "they've", "i will": "i'll", "you will": "you'll", "we will": "we'll", "they will": "they'll",
    "i would": "i'd", "let us": "let's", "what is": "what's",
}
_EXP = re.compile(r"\b(" + "|".join(sorted(map(re.escape, CONTRACTIONS), key=len, reverse=True)) + r")\b", re.I)
_CON = re.compile(r"\b(" + "|".join(sorted(map(re.escape, set(CONTRACTIONS.values())), key=len, reverse=True)) + r")\b", re.I)
_APOS = str.maketrans({"’": "'", "‘": "'"})

def contraction_counts(text: str):
    t = text.translate(_APOS)
    return len(_CON.findall(t)), len(_EXP.findall(t))

def is_fragment(sent) -> bool:
    finite = any(t.pos_ in ("VERB", "AUX") and "Fin" in t.morph.get("VerbForm") for t in sent)
    return (not finite) or sent.root.pos_ not in ("VERB", "AUX")

def is_passive(sent) -> bool:
    return any(t.dep_ in ("nsubjpass", "auxpass") for t in sent)

def adverbial_front(sent) -> bool:
    t = sent[0] if len(sent) else None
    return bool(t is not None and t.dep_ in ("mark", "advmod") and t.head.dep_ == "advcl" and t.head.i > t.i)

def content_lemmas(sent) -> set:
    return {t.lemma_.lower() for t in sent if t.pos_ in ("NOUN", "PROPN", "VERB", "ADJ") and not t.is_stop}

def thesis_index(sents) -> int:
    sets = [content_lemmas(s) for s in sents]
    best, bi = -1.0, 0
    for i, a in enumerate(sets):
        sc = sum(len(a & b) / math.sqrt(len(a) * len(b)) for j, b in enumerate(sets) if j != i and a and b)
        if sc > best: best, bi = sc, i
    return bi

def punct_entropy(text: str) -> float:
    c = Counter(ch for ch in text if not ch.isalnum() and not ch.isspace())
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0

def pct(a, lo=5, hi=95):
    a = [x for x in a if x is not None]
    return [round(float(np.percentile(a, lo)), 4), round(float(np.percentile(a, hi)), 4)] if len(a) >= 5 else None

def measure_texts(texts: list[str], progress=None) -> dict:
    nlp = get_nlp()
    sent_lens, para_s, para_w = [], [], []
    buckets = ("first", "middle", "last")
    cnt = {k: {b: [0, 0] for b in buckets} for k in ("passive", "adverbial_front", "fragment")}
    thesis_hit = thesis_n = 0
    hedge_n = conc_n = sent_n = 0
    con_yes = con_no = 0
    emd = par = semi = words_total = 0
    doc_ttr, doc_stop, doc_pe, doc_slv = [], [], [], []
    n_docs = 0
    for text_i, text in enumerate(texts):
        paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
        d_words, d_types, d_stop, d_lens = [], set(), 0, []
        for doc in nlp.pipe(paras, batch_size=64):
            sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
            if not sents: continue
            wl = [sum(1 for t in s if t.is_alpha) for s in sents]
            para_s.append(len(sents)); para_w.append(sum(wl))
            for i, s in enumerate(sents):
                b = "first" if i == 0 else ("last" if i == len(sents) - 1 else "middle")
                n = wl[i]; sent_lens.append(n); d_lens.append(n)
                sent_n += 1; hedge_n += lead_cue(s.text, HEDGE_CUES); conc_n += lead_cue(s.text, CONCESSION_CUES)
                for k, fn in (("passive", is_passive), ("adverbial_front", adverbial_front), ("fragment", is_fragment)):
                    cnt[k][b][1] += 1; cnt[k][b][0] += int(fn(s))
            if len(sents) >= 3:
                thesis_n += 1; thesis_hit += int(thesis_index(sents) <= 1)
            for t in doc:
                if t.is_alpha:
                    d_words.append(t.lower_); d_stop += int(t.is_stop)
        text_all = " ".join(paras)
        y, n = contraction_counts(text_all); con_yes += y; con_no += n
        nw = len(d_words); words_total += nw
        emd += text_all.count("—") + text_all.count(" -- ") + text_all.count("--")
        par += text_all.count("("); semi += text_all.count(";")
        if nw >= 120:
            n_docs += 1
            w = d_words[:500]
            doc_ttr.append(len(set(w)) / len(w)); doc_stop.append(d_stop / nw)
            doc_pe.append(punct_entropy(text_all)); doc_slv.append(float(np.var(d_lens)) if len(d_lens) > 2 else None)
        if progress: progress(text_i + 1)
    sl = np.array(sent_lens) if sent_lens else np.array([20])
    hist = np.bincount(np.minimum(sl // BW, BINS - 1), minlength=BINS) / len(sl)
    rate = lambda k: sum(v[0] for v in cnt[k].values()) / max(sum(v[1] for v in cnt[k].values()), 1)
    posr = lambda k: {b: round(cnt[k][b][0] / max(cnt[k][b][1], 1), 4) for b in buckets}
    per1k = lambda x: round(1000 * x / max(words_total, 1), 3)
    return {
        "placeholder": False,
        "sent_len": {"mean": round(float(sl.mean()), 3), "var": round(float(sl.var()), 3), "bin_width": BW,
                     "hist": [round(float(x), 4) for x in hist]},
        "para": {"sents_mean": round(float(np.mean(para_s)), 3), "sents_var": round(float(np.var(para_s)), 3),
                 "words_mean": round(float(np.mean(para_w)), 3), "words_var": round(float(np.var(para_w)), 3),
                 "words_hist": [round(float(x), 4) for x in np.histogram(para_w, bins=[0, 25, 50, 75, 100, 150, 200, 300, 10**6])[0] / max(len(para_w), 1)]},
        "rates": {"passive": round(rate("passive"), 4), "adverbial_front": round(rate("adverbial_front"), 4),
                  "early_thesis": round(thesis_hit / max(thesis_n, 1), 4),
                  "contraction": round(con_yes / max(con_yes + con_no, 1), 4),
                  "fragment": round(rate("fragment"), 4), "emdash_per_1k": per1k(emd),
                  "paren_per_1k": per1k(par), "semicolon_per_1k": per1k(semi),
                  "hedge_front": round(hedge_n / max(sent_n, 1), 4), "concession_front": round(conc_n / max(sent_n, 1), 4)},
        "position": {k: posr(k) for k in cnt},
        "bands": {"ttr": pct(doc_ttr), "stopword_ratio": pct(doc_stop), "punct_entropy": pct(doc_pe),
                  "sent_len_var": pct(doc_slv), "perplexity": None},
        "meta": {"words": words_total, "docs": n_docs, "sentences": len(sent_lens), "paragraphs": len(para_s)},
    }

def apply_fragment_labels(profile: dict, register: str):
    """Replace the heuristic fragment rate with the hand/DeepSeek-corrected one if available."""
    p = ROOT / "data" / "profiles" / "fragment_calibration.json"
    if not p.exists(): return
    fc = json.loads(p.read_text()).get(register)
    if not fc: return
    profile["rates"]["fragment_heuristic"] = profile["rates"]["fragment"]
    profile["rates"]["fragment"] = fc["fragment_rate"]
    # keep the shape of the position table but rescale it to the corrected overall rate
    pos = profile["position"]["fragment"]; mean = max(sum(pos.values()) / 3, 1e-9)
    profile["position"]["fragment"] = {b: round(v / mean * fc["fragment_rate"], 4) for b, v in pos.items()}
    profile["position"]["fragment"]["first"] = profile["position"]["fragment"]["last"] = 0.0   # never open/close
    profile["meta"]["fragment_labeled"] = fc["labeled"]

def calibrate(register: str, max_docs: int = 0, quiet: bool = False) -> dict | None:
    files = sorted((CORPUS / register).glob("*.txt")) if (CORPUS / register).exists() else []
    if not files:
        if not quiet: print(f"  {register}: no corpus files, skipping")
        return None
    if max_docs: files = files[:max_docs]
    texts = [f.read_text(encoding="utf-8", errors="ignore") for f in files]
    def prog(i):
        if not quiet and i % 25 == 0: print(f"\r  [{register}] {100*i//len(texts):3d}%  {i}/{len(texts)} docs", end="", flush=True)
    prof = measure_texts(texts, prog)
    if not quiet: print(f"\r  [{register}] 100%  {len(texts)}/{len(texts)} docs   ")
    prof["register"] = register
    apply_fragment_labels(prof, register)
    RegisterProfile(prof).save(register)
    return prof

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", choices=REGISTERS)
    ap.add_argument("--max-docs", type=int, default=0)
    a = ap.parse_args()
    for r in ([a.register] if a.register else REGISTERS):
        p = calibrate(r, a.max_docs)
        if p:
            print(f"    words={p['meta']['words']:,} sent_len mean={p['sent_len']['mean']} var={p['sent_len']['var']} "
                  f"passive={p['rates']['passive']} adv_front={p['rates']['adverbial_front']} "
                  f"early_thesis={p['rates']['early_thesis']} contraction={p['rates']['contraction']} "
                  f"fragment={p['rates']['fragment']}")
    # default profile = pooled average isn't needed; keep DEFAULT as fallback
    from ..profile import PROFILE_DIR
    d = PROFILE_DIR / "default.json"
    if not d.exists():
        PROFILE_DIR.mkdir(parents=True, exist_ok=True); d.write_text(json.dumps(DEFAULT, indent=1))

if __name__ == "__main__":
    main()
