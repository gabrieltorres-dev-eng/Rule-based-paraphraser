"""Build data/dictionaries/idioms.tsv from human-sourced resources only.

Sources, merged:
  1. PPDB 2.0 S (phrasal + lexical), filtered to high-confidence equivalence pairs
     where at least one side is multi-word.
  2. WordNet multi-word lemmas (phrasal verbs, compounds) <-> single-word synonyms.
  3. data/dictionaries/ai_tells_seed.tsv: hand-curated AI-tell phrases. Each listed
     equivalent is kept ONLY if it is attested in the human corpora.
Each entry is tagged with per-register frequencies (per million words) measured on
the calibration corpora, so the runtime can choose register-appropriate equivalents.

Output columns: phrase \t equivalent \t registers \t source \t score \t freqs(json)
"""
import argparse, gzip, json, re
from collections import defaultdict
from _common import DATA, REGISTERS, corpus_docs, need
from _progress import Progress

PPDB = DATA / "dictionaries" / "ppdb-2.0-s-all.gz"
SEED = DATA / "dictionaries" / "ai_tells_seed.tsv"      # AI-written; only used with --use-seed
MINED = DATA / "dictionaries" / "phrase_pairs.tsv"      # from build_phrase_pairs.py
OUT = DATA / "dictionaries" / "idioms.tsv"
BAD = re.compile(r"[0-9\[\]|@#$%&*_=<>/\\]")

def ppdb_pairs(min_score):
    if not PPDB.exists():
        print("  (PPDB not downloaded - skipping it)"); return
    bar = Progress("reading PPDB", PPDB.stat().st_size // (1 << 20), unit=" MB")
    raw = open(PPDB, "rb")
    with gzip.open(raw, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            bar.set(raw.tell() // (1 << 20))
            parts = line.rstrip("\n").split(" ||| ")
            if len(parts) < 6:
                continue
            _, a, b, feats, _, ent = parts[:6]
            if ent.strip() != "Equivalence" or BAD.search(a + b):
                continue
            if " " not in a and " " not in b:
                continue                      # need a phrase on at least one side
            m = re.search(r"PPDB2\.0Score=([\d.]+)", feats)
            s = float(m.group(1)) if m else 0.0
            if s < min_score:
                continue
            a, b = a.strip().lower(), b.strip().lower()
            ta, tb = set(a.split()), set(b.split())
            if a == b or ta <= tb or tb <= ta:  # drop pure insertions/deletions
                continue
            if len(a.split()) > 6 or len(b.split()) > 6:
                continue
            yield a, b, "ppdb", s

_KV = []

def _close(multi, single, thr=0.42):
    """Mean GloVe vector of the phrase vs the single word: a cheap guard against sense errors."""
    import numpy as np
    if not _KV:
        from gensim.models import KeyedVectors
        from _common import DATA
        p = DATA / "models" / "glove-300.kv"
        _KV.append(KeyedVectors.load(str(p), mmap="r") if p.exists() else None)
    kv = _KV[0]
    if kv is None: return True
    ws = [w for w in multi.split() if w in kv]
    if single not in kv or not ws: return False
    a = np.mean([kv[w] for w in ws], axis=0); b = kv[single]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) >= thr

def _common_enough(multi, single):
    from wordfreq import zipf_frequency as z
    return _close(multi, single) and min(z(w, "en") for w in multi.split()) >= 3.2 and z(single, "en") >= 3.4 and not (set(multi.split()) | {single}) & BLOCK

def wordnet_pairs():
    import nltk
    from _common import DATA
    nltk.data.path.insert(0, str(DATA / "nltk_data"))
    from nltk.corpus import wordnet as wn
    try: wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet", download_dir=str(DATA / "nltk_data")); nltk.download("omw-1.4", download_dir=str(DATA / "nltk_data")); wn.ensure_loaded()
    for syn in wn.all_synsets():
        raw = [l.name() for l in syn.lemmas()]
        if any(r[:1].isupper() for r in raw): continue                      # proper names
        try:
            doms = {d.name().split(".")[0] for d in syn.usage_domains()}
        except Exception:
            doms = set()
        if doms & {"slang", "vulgarism", "colloquialism", "obscenity", "disparagement", "derogation", "dialect", "informal", "euphemism", "jargon"}: continue
        lemmas = [r.replace("_", " ").lower() for r in raw]
        multi = [l for l in lemmas if " " in l]
        single = [l for l in lemmas if " " not in l and len(l) >= 3]
        pos = syn.pos()
        for m in multi:
            if BAD.search(m):
                continue
            # sense precision: this synset must be the MOST FREQUENT sense of the multi-word expression and of the single word,
            # otherwise the swap is only valid in a meaning the sentence probably doesn't use ('dollar bill' -> 'clam').
            ms = wn.synsets(m.replace(" ", "_"), pos=pos)
            if not ms or ms[0] != syn: continue
            for s in single:
                if s in m.split(): continue
                ss = wn.synsets(s, pos=pos)
                if ss and ss[0] == syn and _common_enough(m, s):
                    yield m, s, "wordnet", 3.0

def seed_pairs():
    if not SEED.exists():
        return
    for line in SEED.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        phrase, eqs = line.split("\t")[:2]
        for e in eqs.split("|"):
            yield phrase.strip().lower(), e.strip().lower(), "seed", 5.0

FUNC = set("a an the and or but of to in on at for with by is are was were be been am will would can could may might has have had do does did it its this that there when while if as so than then not no all any some most many much each every only also very more less".split())
BAD_START = ("we ", "i ", "there ", "how ", "what ", "why ", "who ", "when ", "where ", "you ")
CONTR = {"it's": "it is", "i'm": "i am", "i've": "i have", "don't": "do not", "can't": "cannot", "isn't": "is not", "that's": "that is",
         "they're": "they are", "we're": "we are", "you're": "you are", "doesn't": "does not", "won't": "will not", "didn't": "did not"}

def _expand(t):
    return " ".join(CONTR.get(w, w) for w in t.lower().split())

PREP = set("into onto from up out off over about around through under between against toward towards upon within without".split())
QUANT = {"few", "couple", "two", "three", "four", "pair", "number", "lot", "lots", "bunch", "dozen", "handful"}
ARTS = ("a ", "an ", "the ")

BLOCK = set("america american americans us usa uk britain british ass asshole fuck fucking shit damn crap bitch bastard jerk porn sex sexy nude suicide kill killed die died dead death murder".split())
AUXQ = set("do does did will is are was were can could would should how what why who".split())

def _stem(w):
    w = w.lower()
    for suf in ("ing", "ed", "es", "s", "e", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            return w[:-len(suf)]
    return w

def good_pair(r):
    """Quality gate for mined pairs: drop tokenizer junk, case/contraction-only changes, function-word swaps, clause-changing swaps."""
    a, h = r["ai_phrase"].strip(), r["human_phrase"].strip()
    if not a or not h or float(r["nli"]) < 0.95 or int(r["count"]) < 2: return False
    if re.search(r"[^A-Za-z' -]|\s'|' ", a + h): return False                    # digits, odd punctuation, spaced apostrophes
    if a.lower() == h.lower() or _expand(a) == _expand(h): return False          # case-only / contraction-only (handled by the contraction module)
    if all(w in FUNC or w in CONTR for w in a.lower().split()): return False     # pure function-word phrases change grammar
    if len(a.split()) == 1 and a.lower().endswith("ing"): return False           # verb-form-sensitive: -ing swaps break grammar
    if h.lower().split()[-1] in ("the", "a", "an", "of", "to", "for", "in", "on") and a.lower().split()[-1] not in ("the", "a", "an", "of", "to", "for", "in", "on"): return False   # dangling function word
    A, H = a.lower().split(), h.lower().split()
    if {_stem(w) for w in A if w not in FUNC and w not in PREP} & {_stem(w) for w in H if w not in FUNC and w not in PREP}: return False   # same word family
    if all(w in FUNC or w in PREP for w in A) or all(w in FUNC or w in PREP for w in H): return False                                 # function-word-only side
    if (set(A) | set(H)) & QUANT and a.lower().startswith(ARTS) != h.lower().startswith(ARTS): return False                           # quantity phrases must keep their article
    la, lh = (A[0] if A and A[0] in ("a", "an", "the") else None), (H[0] if H and H[0] in ("a", "an", "the") else None)
    la = "a" if la == "an" else la; lh = "a" if lh == "an" else lh
    if la != lh: return False                                                       # a/an/the must agree (or both absent)
    if H[-1] in FUNC and A[-1] not in FUNC or A[-1] in FUNC and H[-1] not in FUNC: return False   # dangling 'is', 'the', 'to' ...
    if (set(A) | set(H)) & BLOCK: return False
    if (A[0] in AUXQ) != (H[0] in AUXQ): return False                              # question / auxiliary starts must match
    if float(r["nli"]) < 0.97: return False
    if " - " in a + h or a.lower().startswith(("our ", "has anyone", "who ")): return False
    stems = {w[:4] for w in a.lower().split() if len(w) > 3}
    if stems & {w[:4] for w in h.lower().split() if len(w) > 3}: return False   # same word family (image/image segmentation, option/options available)
    if h.lower().startswith(BAD_START) and not a.lower().startswith(BAD_START): return False
    return True

def mined_pairs():
    """AI phrase -> human phrase pairs mined from paired corpora (human side verbatim)."""
    if not MINED.exists():
        print("  (no phrase_pairs.tsv yet - run build_phrase_pairs.py)"); return
    import csv
    for r in csv.DictReader(open(MINED, encoding="utf-8"), delimiter="\t"):
        if not good_pair(r): continue
        score = 3.0 + 2.0 * float(r["nli"]) + min(float(r["ai_tell_z"]), 10) / 10
        a, h = r["ai_phrase"].lower(), r["human_phrase"].lower()
        yield a, h, "mined", score
        srcs = {x.split(":")[0] for x in r.get("sources", "").split(",") if x}
        if srcs and srcs <= {"coco", "mrpc", "qqp"}:          # human-human paraphrase sets are symmetric
            yield h, a, "mined", score - 0.2

def corpus_counts(phrases):
    """Count each phrase per register (word-boundary, lowercase). Returns per-million freqs."""
    by_len = defaultdict(set)
    for p in phrases:
        by_len[len(p.split())].add(p)
    counts = {r: defaultdict(int) for r in REGISTERS}
    totals = {r: 0 for r in REGISTERS}
    docs = {r: list(corpus_docs(r)) for r in REGISTERS}
    bar = Progress("counting phrases in corpora", sum(map(len, docs.values())), unit=" docs")
    for r in REGISTERS:
        for _, text in docs[r]:
            bar.update()
            toks = re.findall(r"[a-z']+", text.lower())
            totals[r] += len(toks)
            for n, pset in by_len.items():
                for i in range(len(toks) - n + 1):
                    g = " ".join(toks[i:i + n])
                    if g in pset:
                        counts[r][g] += 1
    bar.close()
    return {r: {p: c * 1e6 / max(totals[r], 1) for p, c in counts[r].items()} for r in REGISTERS}, totals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=3.5, help="PPDB2.0Score threshold (1-5)")
    ap.add_argument("--no-ppdb", action="store_true")
    ap.add_argument("--use-seed", action="store_true", help="include the AI-written seed list (only after you reviewed it)")
    a = ap.parse_args()

    pairs = {}
    sources = [wordnet_pairs(), mined_pairs()] + ([seed_pairs()] if a.use_seed else []) + ([] if a.no_ppdb else [ppdb_pairs(a.min_score)])
    for gen in sources:
        for p, e, src, s in gen:
            k = (p, e)
            if k not in pairs or pairs[k][1] < s:
                pairs[k] = (src, s)
    print(f"candidate pairs: {len(pairs):,}")

    vocab = {p for p, _ in pairs} | {e for _, e in pairs}
    freqs, totals = corpus_counts(vocab)
    print("corpus words:", totals)

    kept = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("phrase\tequivalent\tregisters\tsource\tscore\tfreqs\n")
        for (p, e), (src, s) in sorted(pairs.items()):
            ef = {r: round(freqs[r].get(e, 0.0), 3) for r in REGISTERS}
            # the equivalent must be something humans actually write in our corpora
            regs = [r for r in REGISTERS if ef[r] > 0]
            if not regs:
                continue
            f.write(f"{p}\t{e}\t{','.join(regs)}\t{src}\t{s:.2f}\t{json.dumps(ef)}\n")
            kept += 1
    print(f"wrote {kept:,} entries -> {OUT}")

if __name__ == "__main__":
    main()
