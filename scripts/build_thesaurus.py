"""Parse the Moby Thesaurus (data/raw/moby_thesaurus.txt) into data/dictionaries/thesaurus.tsv.
Format handled: one entry per line OR hard-wrapped entries (a line ending in a comma continues the entry).
Only single-word alphabetic synonyms are kept; synonym links are stored for BOTH directions so the runtime can require
that two words list each other (mutual listing = high precision)."""
import re, sys
from collections import defaultdict
from _common import DATA

RAW = DATA / "raw" / "moby_thesaurus.txt"
OUT = DATA / "dictionaries" / "thesaurus.tsv"
WORD = re.compile(r"^[a-z]+$")

def parse_entries(lines):
    entry = ""
    started = False
    for line in lines:
        if not started:
            if "*** START" in line: started = True
            continue
        if "*** END" in line: break
        line = line.strip()
        if not line: continue
        entry += line
        if not line.endswith(","):
            yield entry; entry = ""
    if entry: yield entry

def main():
    if not RAW.exists():
        print("no data/raw/moby_thesaurus.txt - run: python scripts/download_data.py --only thesaurus"); return
    text = RAW.read_text(encoding="utf-8", errors="ignore").splitlines()
    syn = defaultdict(set)
    n = 0
    for e in parse_entries(text):
        parts = [p.strip().lower() for p in e.split(",") if p.strip()]
        if len(parts) < 2: continue
        root, rest = parts[0], parts[1:]
        if not WORD.match(root): continue
        n += 1
        syn[root].update(w for w in rest if WORD.match(w) and w != root)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in sorted(syn):
            if syn[r]: f.write(r + "\t" + " ".join(sorted(syn[r])) + "\n")
    print(f"{n:,} entries -> {len(syn):,} roots written to {OUT}")

if __name__ == "__main__":
    main()
