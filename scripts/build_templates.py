"""Mine the human template bank (data/templates/bank.jsonl) from the calibration corpora.

A template is a short, real, human-written opener or frame, used verbatim at runtime.
Method (no generation):
  - Sentence-split each corpus with spaCy.
  - For each sentence, take the sentence-initial span up to the first comma (<= 6 words).
  - If that span begins with a lexicon cue for a rhetorical function (PDTB-style explicit
    connectives + hedge/concession markers), record it.
  - Keep spans attested in >= --min-docs distinct documents (filters idiosyncratic text).
  - Fragment frames: very short verbless sentences mid-paragraph (e.g. "Not always.").
Every entry keeps its count, doc spread and a real example sentence for review.
Fields: id, text, function, register, count, docs, example
"""
import argparse, json, re
from collections import defaultdict
from _common import DATA, REGISTERS, corpus_docs, load_nlp
from _progress import Progress, save_pickle, load_pickle, drop_pickle

OUT = DATA / "templates" / "bank.jsonl"

LEX = {   # function -> sentence-initial cues (lowercase)
    "hedge": ["in some cases", "in many cases", "in most cases", "arguably", "perhaps", "maybe",
              "probably", "it seems", "it appears", "apparently", "presumably", "possibly",
              "in general", "generally", "typically", "usually", "often", "sometimes",
              "for the most part", "to some extent", "to a degree", "more or less",
              "as far as i can tell", "as far as i know", "i think", "i suspect", "i guess",
              "evidence suggests", "research suggests", "it may be", "it might be"],
    "concession": ["admittedly", "that said", "having said that", "granted", "of course",
                   "to be fair", "even so", "still", "then again", "all the same",
                   "that being said", "sure", "true", "in fairness", "while that's true",
                   "although", "even though", "despite this", "nonetheless", "nevertheless"],
    "transition": ["also", "besides", "on top of that", "what's more", "plus", "then",
                   "after that", "meanwhile", "anyway", "in any case", "either way",
                   "on the other hand", "in contrast", "by contrast", "similarly",
                   "likewise", "in the same way", "so", "as a result", "because of this",
                   "that's why", "this means", "in turn", "in short", "overall", "basically",
                   "in the end", "all in all", "first", "second", "finally", "next", "now"],
    "counterexample": ["but", "however", "yet", "except", "unless", "not always",
                       "on the flip side", "the problem is", "the catch is", "the trouble is"],
    "example_intro": ["for example", "for instance", "say", "take", "consider", "imagine",
                      "like", "such as", "to give an example", "case in point", "e.g"],
}
CUES = sorted(((c, f) for f, cs in LEX.items() for c in cs), key=lambda x: -len(x[0]))

def classify(span):
    s = span.lower().strip()
    for cue, fn in CUES:
        if s == cue or s.startswith(cue + " "):
            return fn
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-docs", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=6)
    a = ap.parse_args()
    nlp = load_nlp()
    new = lambda: {"count": 0, "docs": set(), "example": None}
    stats, coord = defaultdict(new), defaultdict(new)
    ck = load_pickle("templates") or {"done": {}, "stats": {}, "coord": {}}
    stats.update(ck["stats"]); coord.update(ck["coord"])
    alldocs = {reg: list(corpus_docs(reg)) for reg in REGISTERS}
    total = sum(map(len, alldocs.values()))
    bar = Progress("mining templates", total, sum(ck["done"].values()), " docs")
    for reg in REGISTERS:
        docs = alldocs[reg][ck["done"].get(reg, 0):]
        for n, ((doc_id, text), doc) in enumerate(zip(docs, nlp.pipe((t for _, t in docs), batch_size=32))):
            bar.update()
            ck["done"][reg] = ck["done"].get(reg, 0) + 1
            if n % 200 == 199:
                save_pickle("templates", {"done": ck["done"], "stats": dict(stats), "coord": dict(coord)})
            sents = [s for s in doc.sents if s.text.strip()]
            for i, s in enumerate(sents):
                t = s.text.strip()
                # 1) comma-delimited sentence-initial frames
                if "," in t:
                    head = t.split(",", 1)[0]
                    if 1 <= len(head.split()) <= a.max_words:
                        fn = classify(head)
                        if fn:
                            k = (head.strip(), fn, reg)
                            st = stats[k]; st["count"] += 1; st["docs"].add(doc_id)
                            st["example"] = st["example"] or t[:300]
                # 2) fragment frames: short, verbless, mid-paragraph
                if 0 < i < len(sents) - 1 and 1 <= len(s) <= 6 and t[-1:] in ".!?":
                    if not any(tok.pos_ in ("VERB", "AUX") for tok in s):
                        k = (t, "fragment_frame", reg)
                        st = stats[k]; st["count"] += 1; st["docs"].add(doc_id)
                        st["example"] = st["example"] or t
                # 3) coordination joiners actually used between clauses
                for tok in s:
                    if tok.dep_ == "cc" and tok.head.pos_ in ("VERB", "AUX") and tok.i > s.start:
                        prev = doc[tok.i - 1].text
                        j = (", " if prev == "," else " ") + tok.text.lower() + " "
                        c = coord[(j, reg)]; c["count"] += 1; c["docs"].add(doc_id)
                        c["example"] = c["example"] or t[:300]

    bar.close()
    save_pickle("templates", {"done": ck["done"], "stats": dict(stats), "coord": dict(coord)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        rows = [((txt, fn, reg), st) for (txt, fn, reg), st in stats.items()]
        rows += [((j, "coordination", reg), st) for (j, reg), st in coord.items()]
        for (txt, fn, reg), st in sorted(rows, key=lambda r: (r[0][1], r[0][2], -r[1]["count"])):
            if len(st["docs"]) < a.min_docs:
                continue
            n += 1
            f.write(json.dumps({"id": f"{fn[:4]}-{reg[:3]}-{n:05d}", "text": txt, "function": fn,
                                "register": reg, "count": st["count"], "docs": len(st["docs"]),
                                "example": st["example"]}, ensure_ascii=False) + "\n")
    drop_pickle("templates")
    print(f"wrote {n} templates -> {OUT}  (review by hand; delete any bad lines)")

if __name__ == "__main__":
    main()
