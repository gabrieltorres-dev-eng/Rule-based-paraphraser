"""Measure how structurally different real HUMAN paraphrases are, and how bursty real human prose is.
Writes data/profiles/structure.json:
  ted_quantiles : 101 quantiles of normalised dependency-tree edit distance between human paraphrase pairs (MRPC/QQP)
  burst[register]: p05/p10/p50/p90 burstiness of sentence lengths per document in data/corpus/<register>/*.txt
The pipeline draws each document's structural target/floor from this distribution; without it, built-in defaults are used."""
import json, sys, re, random
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paraphraser import structure as st
from paraphraser.routing.util import warm

def pairs(limit=4000):
    out = []
    for name in ("mrpc", "qqp"):
        f = ROOT / "data" / "pairs" / f"{name}.jsonl"
        if not f.exists(): continue
        for line in f.open(encoding="utf-8"):
            try: h = json.loads(line).get("human") or []
            except Exception: continue
            if len(h) >= 2 and all(len(x.split()) >= 10 for x in h[:2]): out.append((h[0], h[1]))
    random.Random(0).shuffle(out)
    return out[:limit]

def main():
    P = pairs()
    print(f"{len(P)} human paraphrase pairs")
    if P: warm([x for p in P for x in p])
    ted = []
    for i, (a, b) in enumerate(P):
        ted.append(st.compare(a, b)["ted"])
        st._INFO_CACHE.clear()
    prof = {}
    if len(ted) >= 100:
        prof["ted_quantiles"] = [round(float(x), 4) for x in np.quantile(ted, np.linspace(0, 1, 101))]
        print("TED p10/p50/p90:", [prof["ted_quantiles"][k] for k in (10, 50, 90)])
    burst = {}
    for d in sorted((ROOT / "data" / "corpus").glob("*")):
        if not d.is_dir(): continue
        vals = []
        for f in d.glob("*.txt"):
            t = f.read_text(encoding="utf-8", errors="ignore")
            if len(t.split()) < 150: continue
            b, n = st.text_burstiness(t[:20000]); st._INFO_CACHE.clear()
            if n >= 6: vals.append(b)
        if len(vals) >= 5:
            burst[d.name] = {k: round(float(np.quantile(vals, q)), 4) for k, q in (("p05", .05), ("p10", .10), ("p50", .5), ("p90", .9))}
            print(d.name, burst[d.name])
    if burst: prof["burst"] = burst
    if not prof: print("nothing measured; defaults stay in use"); return
    out = ROOT / "data" / "profiles" / "structure.json"
    out.write_text(json.dumps(prof, indent=1), encoding="utf-8"); print("wrote", out)

if __name__ == "__main__": main()
