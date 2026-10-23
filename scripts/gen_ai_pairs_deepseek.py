"""OPTIONAL: create extra AI-side text with DeepSeek for phrase-pair mining.

For human paragraphs in data/corpus/<register>/, DeepSeek writes its own version of the
same content. That AI text is used ONLY as a lookup key (to learn which AI phrasings map
to which human phrasings). It is never inserted into any output.
Writes data/pairs/deepseek.jsonl in the same format as hc3/raid.

    set DEEPSEEK_API_KEY=...  ;  python scripts/gen_ai_pairs_deepseek.py --per-register 600
"""
import argparse, json, os, random, sys, time, urllib.request
from _common import PAIRS, REGISTERS, corpus_docs
from _progress import Progress

URL = "https://api.deepseek.com/chat/completions"
PROMPT = ("Rewrite the following passage in your own words, keeping all of its meaning, "
          "facts and roughly its length. Output only the rewritten passage.\n\n{p}")

def call(key, text):
    body = {"model": "deepseek-chat", "temperature": 0.7,
            "messages": [{"role": "user", "content": PROMPT.format(p=text)}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  error {e}; retry"); time.sleep(2 ** a)
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-register", type=int, default=600)
    a = ap.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY") or sys.exit("Set DEEPSEEK_API_KEY first.")
    PAIRS.mkdir(parents=True, exist_ok=True)
    out = PAIRS / "deepseek.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["group"] for l in open(out, encoding="utf-8")}
    rng = random.Random(7)
    bar = Progress("deepseek AI pairs", a.per_register * len(REGISTERS), unit=" paragraphs")
    with open(out, "a", encoding="utf-8") as f:
        for reg in REGISTERS:
            paras = [(d, i, p) for d, t in corpus_docs(reg)
                     for i, p in enumerate(t.split("\n")) if 60 <= len(p.split()) <= 250]
            rng.shuffle(paras)
            n = 0
            for d, i, p in paras:
                if n >= a.per_register: break
                g = f"ds-{reg}-{d}-{i}"
                n += 1; bar.update()
                if g in done: continue
                ai = call(key, p)
                if ai:
                    f.write(json.dumps({"group": g, "domain": reg, "human": [p], "ai": [ai]}) + "\n"); f.flush()
    bar.close(); print(f"-> {out}")

if __name__ == "__main__":
    main()
