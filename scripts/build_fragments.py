"""Fragment-rate calibration (spec Phase 0: needs manual annotation).

Step 1 (default):  python scripts/build_fragments.py
   Writes data/profiles/fragment_sample.csv: for each register, 75 sentences the spaCy
   heuristic flags as fragments + 75 random others, with a blank `is_fragment` column.
   Open it in Excel, fill y / n, save as CSV (same name).
Step 2:            python scripts/build_fragments.py --apply
   Reads your labels, measures heuristic precision/recall, and writes a corrected
   per-register fragment rate (overall + by paragraph position) to
   data/profiles/fragment_calibration.json, which calibration (M1) merges in.
"""
import argparse, csv, json, random
from collections import defaultdict
from _common import DATA, REGISTERS, corpus_docs, load_nlp
from _progress import Progress

SAMPLE = DATA / "profiles" / "fragment_sample.csv"
OUT = DATA / "profiles" / "fragment_calibration.json"

def is_frag(s):
    """Heuristic: no finite verb at/under the root, or root isn't a verb."""
    root = s.root
    finite = any(t.pos_ in ("VERB", "AUX") and "Fin" in t.morph.get("VerbForm") for t in s)
    return (not finite) or root.pos_ not in ("VERB", "AUX")

def pos_bucket(i, n):
    return "first" if i == 0 else ("last" if i == n - 1 else "middle")

def scan(nlp, per_reg_docs=400):
    """Yield (register, pos_bucket, sentence_text, heuristic_flag)."""
    alldocs = {r: list(corpus_docs(r))[:per_reg_docs] for r in REGISTERS}
    bar = Progress("scanning for fragments", sum(map(len, alldocs.values())), unit=" docs")
    for reg in REGISTERS:
        for _, text in alldocs[reg]:
            bar.update()
            for para in (p for p in text.split("\n") if p.strip()):
                sents = [s for s in nlp(para).sents if s.text.strip() and any(c.isalpha() for c in s.text)]
                for i, s in enumerate(sents):
                    yield reg, pos_bucket(i, len(sents)), s.text.strip(), is_frag(s)
    bar.close()

def make_sample(seed):
    rng = random.Random(seed)
    nlp = load_nlp()
    pos, neg, totals = defaultdict(list), defaultdict(list), defaultdict(lambda: [0, 0])
    for reg, b, t, flag in scan(nlp):
        (pos if flag else neg)[reg].append((b, t))
        totals[reg][0] += 1; totals[reg][1] += flag
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["register", "position", "heuristic", "is_fragment", "sentence"])
        for reg in REGISTERS:
            for flag, pool in ((1, pos[reg]), (0, neg[reg])):
                for b, t in rng.sample(pool, min(75, len(pool))):
                    w.writerow([reg, b, flag, "", t])
    json.dump({r: {"sentences": n, "heuristic_flagged": k} for r, (n, k) in totals.items()},
              open(SAMPLE.with_suffix(".totals.json"), "w"), indent=2)
    print(f"Label the is_fragment column (y/n) in {SAMPLE}, then rerun with --apply")

def apply():
    totals = json.load(open(SAMPLE.with_suffix(".totals.json")))
    rows = [r for r in csv.DictReader(open(SAMPLE, encoding="utf-8")) if r["is_fragment"].strip()]
    if not rows:
        raise SystemExit("No labels found in the CSV yet.")
    out = {}
    for reg in REGISTERS:
        R = [r for r in rows if r["register"] == reg]
        if not R or reg not in totals:
            continue
        lab = lambda r: r["is_fragment"].strip().lower().startswith("y")
        P = [r for r in R if r["heuristic"] == "1"]; N = [r for r in R if r["heuristic"] == "0"]
        prec = sum(map(lab, P)) / max(len(P), 1)          # P(frag | flagged)
        miss = sum(map(lab, N)) / max(len(N), 1)          # P(frag | not flagged)
        t = totals[reg]; flagged = t["heuristic_flagged"] / t["sentences"]
        rate = flagged * prec + (1 - flagged) * miss      # corrected population rate
        by_pos = {}
        for b in ("first", "middle", "last"):
            B = [r for r in R if r["position"] == b]
            by_pos[b] = round(sum(map(lab, B)) / len(B), 4) if B else None
        out[reg] = {"fragment_rate": round(rate, 4), "heuristic_precision": round(prec, 3),
                    "heuristic_miss_rate": round(miss, 3), "labeled": len(R),
                    "sample_rate_by_position": by_pos}
    json.dump(out, open(OUT, "w"), indent=2)
    print(json.dumps(out, indent=2)); print(f"-> {OUT}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true"); ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    apply() if a.apply else make_sample(a.seed)
