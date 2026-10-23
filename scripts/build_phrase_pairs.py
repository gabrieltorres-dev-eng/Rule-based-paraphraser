"""Mine AI-phrase -> human-phrase replacement pairs (2-8 words) from paired human/AI text.

Inputs : data/pairs/*.jsonl  (hc3, raid, optional deepseek)  rows {"group","domain","human":[..],"ai":[..]}
Outputs: data/dictionaries/ai_tells_mined.tsv   n-grams over-used by AI (log-odds z-score)
         data/dictionaries/phrase_pairs.tsv     ai_phrase -> human_phrase with evidence

Only the HUMAN side of each pair can ever be inserted into output; it is copied verbatim
from human-written text. Models here (MiniLM, NLI cross-encoder) only score.

Pipeline:
  1. AI tells: n-gram (1-5) log-odds ratio with informative Dirichlet prior (Monroe et al. 2008).
  2. Sentence alignment inside each group: embed sentences, pair each AI sentence with its
     most similar human sentence (cos >= --align).
  3. Span extraction: token diff of aligned pairs; a 'replace' block whose two sides are 1-8
     tokens (at least one side >= 2) and which is anchored by matching words gives a candidate.
     Human-human pairs (several human answers in HC3) are mined too.
  4. Filters: seen in >= --min-groups groups, span cosine >= --span-sim, bidirectional NLI
     entailment of the sentence with the span swapped >= --nli, no numbers/negation mismatch.
"""
import argparse, csv, difflib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
from _common import DATA, PAIRS
from _progress import State, Progress, save_pickle, load_pickle, drop_pickle

ST = State("phrase_pairs")

OUT_TELLS = DATA / "dictionaries" / "ai_tells_mined.tsv"
OUT_PAIRS = DATA / "dictionaries" / "phrase_pairs.tsv"
TOK = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|[0-9]+|[^\sA-Za-z0-9]")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
NEG = {"not", "no", "never", "n't", "none", "nothing", "without", "nor"}
STOP_EDGE = {",", ".", ";", ":", "!", "?", "(", ")", '"'}

def toks(s): return TOK.findall(s)
def sents(t): return [s.strip() for s in SENT.split(re.sub(r"\s+", " ", t)) if 4 <= len(s.split()) <= 60]

def load_groups(limit):
    groups = []
    for p in sorted(PAIRS.glob("*.jsonl")):
        n = 0
        for line in open(p, encoding="utf-8"):
            g = json.loads(line); groups.append(g); n += 1
            if limit and n >= limit: break
        print(f"  {p.name}: {n} groups")
    if not groups: raise SystemExit("No data/pairs/*.jsonl - run download_data.py --only pairs")
    return groups

# ---------------------------------------------------------------- 1. AI tells
def ai_tells(groups, top, min_count=20):
    if ST.done("tells") and OUT_TELLS.exists():
        print("  AI tells already built - loading")
        rows = [l.rstrip("\n").split("\t") for l in open(OUT_TELLS, encoding="utf-8")][1:]
        return {r[0]: float(r[1]) for r in rows}
    ca, ch = Counter(), Counter()
    bar = Progress("ai-tells: counting n-grams", len(groups))
    for g in groups:
        bar.update()
        for side, c in (("ai", ca), ("human", ch)):
            for t in g[side]:
                w = [x.lower() for x in toks(t)]
                for n in range(1, 6):
                    for i in range(len(w) - n + 1):
                        gram = w[i:i + n]
                        if any(x in STOP_EDGE for x in gram): continue
                        c[" ".join(gram)] += 1
    na, nh = sum(ca.values()), sum(ch.values())
    prior = ca + ch; a0 = sum(prior.values()); A = 1000.0          # total prior mass
    rows = []
    for gram, pc in prior.items():
        ya, yh = ca[gram], ch[gram]
        if ya + yh < min_count: continue
        aw = A * pc / a0
        d = math.log((ya + aw) / (na + A - ya - aw)) - math.log((yh + aw) / (nh + A - yh - aw))
        rows.append((d / math.sqrt(1 / (ya + aw) + 1 / (yh + aw)), gram, ya, yh))
    rows.sort(reverse=True)
    OUT_TELLS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TELLS, "w", encoding="utf-8") as f:
        f.write("phrase\tz\tai_count\thuman_count\n")
        for z, gram, ya, yh in rows[:top]:
            if z < 3: break
            f.write(f"{gram}\t{z:.2f}\t{ya}\t{yh}\n")
    bar.close(); ST.mark("tells")
    print(f"  AI tells -> {OUT_TELLS}")
    return {gram: z for z, gram, *_ in rows[:top] if z >= 3}

# ---------------------------------------------------------------- 2-3. align + extract
def extract(groups, emb, align_thr, max_sents):
    import numpy as np
    saved = load_pickle("pp_cands")
    cands = defaultdict(lambda: {"groups": set(), "count": 0, "ctx": []})
    start = 0
    if saved and ST.get("extract_n") == len(groups):
        cands.update(saved["cands"]); start = saved["next"]
        print(f"  resuming alignment at group {start}")
    bar = Progress("aligning sentences", len(groups), start, " groups")
    for gi, g in enumerate(groups):
        if gi < start: continue
        bar.update()
        if gi % 500 == 0 and gi > start:
            save_pickle("pp_cands", {"cands": dict(cands), "next": gi}); ST.set("extract_n", len(groups))
        H = [s for t in g["human"] for s in sents(t)][:max_sents]
        A = [s for t in g["ai"] for s in sents(t)][:max_sents]
        if not H: continue
        sources = []
        if A: sources.append((A, H))
        if len(g["human"]) > 1: sources.append((H, H))      # human-human paraphrases
        vec = {}
        allS = list(dict.fromkeys(H + A))
        V = emb.encode(allS, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        vec = dict(zip(allS, V))
        for X, Y in sources:
            Ym = np.stack([vec[y] for y in Y])
            for x in X:
                sims = Ym @ vec[x]
                order = np.argsort(-sims)
                j = next((k for k in order if Y[k] != x), None)
                if j is None or sims[j] < align_thr: continue
                y = Y[j]
                tx, ty = toks(x), toks(y)
                lx, ly = [t.lower() for t in tx], [t.lower() for t in ty]
                for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, lx, ly, autojunk=False).get_opcodes():
                    if op != "replace": continue
                    a, b = lx[i1:i2], ly[j1:j2]
                    if not (1 <= len(a) <= 8 and 1 <= len(b) <= 8 and max(len(a), len(b)) >= 2): continue
                    if any(t in STOP_EDGE for t in a + b): continue
                    if any(t.isdigit() for t in a + b): continue
                    if (NEG & set(a)) != (NEG & set(b)): continue
                    anchored = (i1 > 0 and j1 > 0) or (i2 < len(lx) and j2 < len(ly))
                    if not anchored: continue
                    key = (" ".join(a), " ".join(ty[j1:j2]))     # human side keeps original case
                    c = cands[key]; c["groups"].add(g["group"]); c["count"] += 1
                    if len(c["ctx"]) < 3: c["ctx"].append((x, " ".join(tx[i1:i2])))
    bar.close()
    save_pickle("pp_cands", {"cands": dict(cands), "next": len(groups)}); ST.set("extract_n", len(groups))
    print(f"  {len(cands):,} raw candidates")
    return cands

# ---------------------------------------------------------------- 4. filter
def filter_pairs(cands, emb, nli, tells, a):
    import numpy as np
    keep = [(k, v) for k, v in cands.items() if len(v["groups"]) >= a.min_groups]
    print(f"  {len(keep)} candidates seen in >= {a.min_groups} groups")
    if not keep: return []
    A = emb.encode([k[0] for k, _ in keep], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    B = emb.encode([k[1] for k, _ in keep], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    sims = (A * B).sum(1)
    keep = [(k, v, float(s)) for (k, v), s in zip(keep, sims) if s >= a.span_sim]
    print(f"  {len(keep)} pass span similarity >= {a.span_sim}")
    saved = load_pickle("pp_nli") or {"out": [], "next": 0, "n": None}
    if saved["n"] != len(keep): saved = {"out": [], "next": 0, "n": len(keep)}
    out = saved["out"]
    labels = None
    bar = Progress("meaning check (NLI)", len(keep), saved["next"], " pairs")
    for i in range(saved["next"], len(keep), 128):
        if i and i % (128 * 20) == 0:
            save_pickle("pp_nli", {"out": out, "next": i, "n": len(keep)})
        bar.set(i)
        chunk = keep[i:i + 128]
        prem, hyp = [], []
        for (ap, hp), v, _ in chunk:
            sent, orig = v["ctx"][0]
            swapped = sent.replace(orig, hp, 1)
            prem += [sent, swapped]; hyp += [swapped, sent]
        scores = nli.predict(list(zip(prem, hyp)), apply_softmax=True, show_progress_bar=False)
        if labels is None:
            cfg = getattr(nli, "config", None) or getattr(getattr(nli, "model", None), "config", None)
            lab = {v.lower(): k for k, v in cfg.id2label.items()} if cfg else {}
            labels = lab.get("entailment", 1)
        for n, ((ap, hp), v, s) in enumerate(chunk):
            e = float(min(scores[2 * n][labels], scores[2 * n + 1][labels]))
            if e >= a.nli:
                src = Counter(g.split("-")[0] for g in v["groups"])
                srcs = ",".join(f"{k}:{n}" for k, n in src.most_common())
                out.append((ap, hp, len(v["groups"]), v["count"], s, e, tells.get(ap, 0.0), v["ctx"][0][0], srcs))
    bar.set(len(keep)); bar.close()
    out.sort(key=lambda r: (-r[6], -r[2]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max groups per pairs file (0 = all)")
    ap.add_argument("--align", type=float, default=0.72)
    ap.add_argument("--span-sim", type=float, default=0.55)
    ap.add_argument("--nli", type=float, default=0.80)
    ap.add_argument("--min-groups", type=int, default=2)
    ap.add_argument("--max-sents", type=int, default=40)
    ap.add_argument("--top-tells", type=int, default=3000)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer, CrossEncoder
    mdir = DATA / "models"
    emb = SentenceTransformer(str(mdir / "minilm") if (mdir / "minilm").exists() else "sentence-transformers/all-MiniLM-L6-v2")
    nli = CrossEncoder(str(mdir / "nli") if (mdir / "nli").exists() else "cross-encoder/nli-deberta-v3-small")

    groups = load_groups(a.limit)
    tells = ai_tells(groups, a.top_tells)
    cands = extract(groups, emb, a.align, a.max_sents)
    rows = filter_pairs(cands, emb, nli, tells, a)
    with open(OUT_PAIRS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ai_phrase", "human_phrase", "groups", "count", "span_sim", "nli", "ai_tell_z", "example", "sources"])
        for r in rows: w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.3f}", f"{r[5]:.3f}", f"{r[6]:.2f}", r[7], r[8]])
    drop_pickle("pp_cands"); drop_pickle("pp_nli"); ST.clear()
    print(f"  wrote {len(rows)} phrase pairs -> {OUT_PAIRS}  (review the top of the file by hand)")

if __name__ == "__main__":
    main()
