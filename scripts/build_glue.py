"""Attest the closed-class glue words the rewrites are allowed to insert ('which', 'that is why', 'carry out', 'the', 'by' ...).
Each is counted in the HUMAN calibration corpora only (never AI text). Writes data/dictionaries/glue.tsv:
  phrase <TAB> total_count <TAB> per-register counts (json).
A rewrite may insert a phrase only if it is listed here with count >= MIN_COUNT (the runtime gate in paraphraser/glue.py)."""
import json, re
from collections import Counter
from _common import DATA, REGISTERS, corpus_docs

PHRASES = ["which", "which means", "this means", "that is why", "this is why", "that is", "this is", "this",
           "that", "the", "a", "an", "of", "carry out", "is", "are", "was", "were", "be", "been", "by", "who", "where", "and", "when"]
MIN_COUNT = 5
OUT = DATA / "dictionaries" / "glue.tsv"

def main():
    pats = {p: re.compile(r"\b" + re.escape(p) + r"\b") for p in PHRASES}
    per = {p: Counter() for p in PHRASES}
    for reg in REGISTERS:
        for _, text in corpus_docs(reg):
            low = text.lower()
            for p, rx in pats.items():
                per[p][reg] += len(rx.findall(low))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for p in PHRASES:
            tot = sum(per[p].values())
            if tot >= MIN_COUNT:
                kept += 1
            f.write(f"{p}\t{tot}\t{json.dumps(dict(per[p]))}\n")
    print(f"{kept}/{len(PHRASES)} glue phrases attested (>= {MIN_COUNT} human occurrences) -> {OUT}")

if __name__ == "__main__":
    main()
