"""Fill the is_fragment column of data/profiles/fragment_sample.csv using DeepSeek
as a yes/no CLASSIFIER only (it never writes text that enters the pipeline).

    set DEEPSEEK_API_KEY=sk-...          (PowerShell: $env:DEEPSEEK_API_KEY="sk-...")
    python scripts/label_fragments_deepseek.py
    python scripts/build_fragments.py --apply

Only blank cells are filled, so any labels you typed yourself are kept. Each row also gets
`labeler` = human/deepseek. A random 10% audit list is printed: check those by hand; if you
disagree with more than ~2 of them, re-label by hand instead.
"""
import csv, json, os, random, sys, time, urllib.request
from _common import DATA
from _progress import Progress

CSV_PATH = DATA / "profiles" / "fragment_sample.csv"
URL = "https://api.deepseek.com/chat/completions"
BATCH = 25
SYSTEM = ("You are a grammar annotator. A sentence FRAGMENT is a written unit punctuated as a "
          "sentence that lacks an independent clause (no subject + finite verb as its main "
          "clause), e.g. 'Not always.', 'Which is fine.', 'Because of the rain.', "
          "'A huge mistake, honestly.'. Imperatives ('Go home.') and full clauses are NOT fragments. "
          "Return JSON {\"labels\": [\"y\" or \"n\", ...]} with one label per input, same order.")

def ask(key, sents):
    body = {"model": "deepseek-chat", "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps({"sentences": sents})}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(json.loads(r.read())["choices"][0]["message"]["content"])["labels"]
            out = [str(x).strip().lower()[:1] for x in out]
            if len(out) == len(sents) and set(out) <= {"y", "n"}:
                return out
            print("  bad answer shape, retrying")
        except Exception as e:
            print(f"  error {e}, retrying"); time.sleep(2 ** attempt)
    raise SystemExit("DeepSeek failed repeatedly; labels so far are saved.")

def main():
    key = os.environ.get("DEEPSEEK_API_KEY") or sys.exit("Set DEEPSEEK_API_KEY first.")
    if not CSV_PATH.exists():
        sys.exit("Run scripts/build_fragments.py first to create the sample CSV.")
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    for r in rows:
        r.setdefault("labeler", "")
        if r["is_fragment"].strip() and not r["labeler"]:
            r["labeler"] = "human"
    todo = [r for r in rows if not r["is_fragment"].strip()]
    print(f"{len(todo)} rows to label")
    bar = Progress("deepseek labels", len(todo), unit=" rows")
    fields = ["register", "position", "heuristic", "is_fragment", "labeler", "sentence"]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        for r, lab in zip(chunk, ask(key, [r["sentence"] for r in chunk])):
            r["is_fragment"], r["labeler"] = lab, "deepseek"
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:   # save after every batch
            w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
        bar.set(min(i + BATCH, len(todo)))
    bar.close()
    ds = [r for r in rows if r["labeler"] == "deepseek"]
    print("\nAUDIT these by hand (sentence -> deepseek label):")
    for r in random.Random(0).sample(ds, max(1, len(ds) // 10)) if ds else []:
        print(f"  [{r['is_fragment']}] {r['sentence']}")

if __name__ == "__main__":
    main()
