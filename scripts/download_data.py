"""Download every external resource for pythonParaphraser (~9 GB total).

Run on your own machine:  python scripts/download_data.py   [--only step1,step2]
Re-runnable: finished items are skipped.
Steps: lexical, corpora, technical, pairs, kenlm, ppdb
"""
import argparse, io, json, re, subprocess, sys, tarfile, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, CORPUS, PAIRS
from _progress import State, Progress

ST = State("download")
ERR = []   # sub-item failures; a step with failures is not marked done

WORD_CAP = 200_000          # words kept per register
PAIR_CAP = 20_000           # human/AI groups kept per paired dataset
EN_RAID = {"abstracts", "books", "news", "reddit", "reviews", "wiki", "recipes"}

def pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

def _ssl_ctx():
    """Use the OS certificate store (fixes 'unable to get local issuer certificate' on Windows)."""
    try:
        import truststore, ssl
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        try: pip("truststore"); return _ssl_ctx()
        except Exception: return None

def fetch(url, dest: Path):
    """Resumable download: keeps a .part file and continues it with an HTTP Range request."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists) {dest.name}"); return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, context=_ssl_ctx() if url.startswith("https") else None, timeout=60) as r:
        if have and r.status != 206:          # server ignored Range -> start over
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        bar = Progress(dest.name, total // (1 << 20), have // (1 << 20), " MB")
        with open(tmp, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk: break
                f.write(chunk); bar.update(1)
        bar.close()
    tmp.replace(dest)

def stream(name, config=None, **kw):
    from datasets import load_dataset
    return load_dataset(name, config, split="train", streaming=True, **kw)

# ---------------------------------------------------------------- corpora
def _write(register, texts, cap=WORD_CAP, prefix="", join=False):
    """Write human docs until `cap` words. Resumes: already-written stream items are skipped."""
    key = f"corpus.{register}.{prefix or 'main'}"
    st = ST.get(key, {})
    if st == "done":
        print(f"  skip (done) corpus/{register} {prefix}"); return
    out = CORPUS / register; out.mkdir(parents=True, exist_ok=True)
    words, last = st.get("words", 0), st.get("last", -1)
    bar = Progress(f"corpus/{register}{'/' + prefix.rstrip('_') if prefix else ''}", cap, words, " words")
    for i, t in enumerate(texts):
        if i <= last: continue                       # fast-forward past saved items
        t = re.sub(r"[ \t]+\n", "\n", (t or "")).strip()
        if join: t = " ".join(t.split())     # one paragraph per doc (arXiv abstracts have hard wraps)
        if len(t.split()) >= 80:
            (out / f"{prefix}{i:06d}.txt").write_text(t, encoding="utf-8")
            words += len(t.split()); bar.set(words)
        if i % 50 == 0: ST.set(key, {"words": words, "last": i})
        if words >= cap: break
    bar.close(); ST.mark(key)

def blog_posts():
    """Blog Authorship Corpus (Schler et al. 2006, blogger.com 2004). The HF copy uses a
    loading script that new `datasets` refuses, so read the identical original XML files."""
    url = "https://codeload.github.com/mdibaiee/blogs-corpus/tar.gz/refs/heads/master"
    dest = DATA / "raw" / "blogs-corpus.tar.gz"
    fetch(url, dest)
    with tarfile.open(dest, "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".xml")): continue
            raw = tf.extractfile(m).read().decode("latin-1", "ignore")
            for post in re.findall(r"<post>(.*?)</post>", raw, re.S):
                yield re.sub(r"\s*\n\s*\n\s*", "\n", post.strip())

def persuade_essays():
    """PERSUADE 2.0 (Crossley et al. 2024): ~25k student argumentative essays.
    Official training CSV is on Google Drive (link from github.com/scrosseye/persuade_corpus_2.0)."""
    import csv, zipfile
    pip("gdown")
    import gdown
    dest = DATA / "raw" / "persuade_2.0_train"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id="13phHyDzIsb0MHyJr6q-B-qIa9P2tM135", output=str(dest) + ".part", quiet=False)
        Path(str(dest) + ".part").replace(dest)
    csv.field_size_limit(10**8)
    if zipfile.is_zipfile(dest):
        z = zipfile.ZipFile(dest)
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        fh = io.TextIOWrapper(z.open(name), encoding="utf-8", errors="ignore")
    else:
        fh = open(dest, encoding="utf-8", errors="ignore")
    seen = set()
    for r in csv.DictReader(fh):                 # one row per discourse element -> dedupe essays
        eid = r.get("essay_id_comp") or r.get("essay_id")
        if eid in seen: continue
        seen.add(eid); yield r["full_text"]

def corpora():
    """General registers. Sources pre-date common LLM text."""
    pip("datasets>=2.19", "huggingface_hub")
    jobs = {
        "blog":  blog_posts,
        "essay": persuade_essays,
        # Webis-TLDR-17 Reddit 2006-2016, via HF's auto-converted parquet (no loading script)
        "forum": lambda: (r["content"] for r in stream("webis/tldr-17", revision="refs/convert/parquet")),
    }
    for reg, gen in jobs.items():
        try: _write(reg, gen())
        except Exception as e: ERR.append(1); print(f"  !! {reg} failed: {e}")

# ---------------------------------------------------------------- technical fix
RST_DROP = re.compile(r"^\s*(\.\. |:[a-z]+:|[=\-~^\"'`#*+]{3,}\s*$)")
def clean_rst(text):
    """Keep prose paragraphs from .rst/.txt docs; drop directives, code blocks, tables."""
    out, in_code = [], False
    for line in text.splitlines():
        if line.rstrip().endswith("::"):
            in_code = True; line = line.rstrip()[:-1]
        elif in_code:
            if line.strip() and not line.startswith((" ", "\t")): in_code = False
            else: continue
        if RST_DROP.match(line) or line.startswith((" ", "\t", ">>>", "|", "+")): 
            out.append(""); continue
        line = re.sub(r":[a-z:]+:`([^`<]*)(<[^>]*>)?`", r"\1", line)   # :func:`x` -> x
        line = re.sub(r"`+([^`]*)`+_?", r"\1", line).replace("**", "").replace("*", "")
        out.append(line)
    paras = [" ".join(p.split()) for p in "\n".join(out).split("\n\n")]
    return "\n".join(p for p in paras if len(p.split()) >= 12 and p[-1:] in ".:!?)")

def tar_docs(url, member_re):
    print(f"  downloading {url}")
    data = urllib.request.urlopen(url).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and re.search(member_re, m.name):
                yield clean_rst(tf.extractfile(m).read().decode("utf-8", "ignore"))

def technical():
    """Fix for the technical register: real human-written software documentation and
    Q&A instead of encyclopedia text, plus a separate 'academic' register.
      technical = CPython 3.9 docs (2020) + Django 3.2 docs (2021) + Stack Exchange answers
      academic  = arXiv abstracts (2021 snapshot)"""
    pip("datasets>=2.19")
    third = WORD_CAP // 3
    try: _write("technical", tar_docs("https://codeload.github.com/python/cpython/tar.gz/refs/tags/v3.9.0",
                                      r"/Doc/(tutorial|howto|library|faq|using)/.*\.rst$"), third, "pydoc_")
    except Exception as e: ERR.append(1); print(f"  !! cpython docs: {e}")
    try: _write("technical", tar_docs("https://codeload.github.com/django/django/tar.gz/refs/tags/3.2",
                                      r"/docs/(topics|intro|howto|ref)/.*\.txt$"), third, "django_")
    except Exception as e: ERR.append(1); print(f"  !! django docs: {e}")
    try: _write("technical", (r["response_j"] for r in stream("lvwerra/stack-exchange-paired",
                                                               data_dir="data/finetune")), third, "se_")
    except Exception as e: ERR.append(1); print(f"  !! stack exchange: {e}")
    try: _write("academic", (r["abstract"] for r in stream("gfissore/arxiv-abstracts-2021")
                             if len(r["abstract"].split()) >= 80), join=True)
    except Exception as e: ERR.append(1); print(f"  !! arxiv: {e}")

# ---------------------------------------------------------------- paired human/AI text
def pairs():
    """Human vs AI text about the SAME item, for phrase-pair mining. AI text is only a lookup key.
    Writes data/pairs/<name>.jsonl rows: {"group", "domain", "human": [...], "ai": [...]}"""
    pip("datasets>=2.19", "huggingface_hub")
    PAIRS.mkdir(parents=True, exist_ok=True)
    if not ST.done("pairs.hc3"):
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
            rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        except Exception as e:
            print(f"  hc3 jsonl failed ({e}); trying parquet")
            from datasets import load_dataset
            rows = load_dataset("Hello-SimpleAI/HC3", "all", revision="refs/convert/parquet", split="train")
        tmp = PAIRS / "hc3.jsonl.tmp"; n = 0
        bar = Progress("pairs/hc3", len(rows))
        with open(tmp, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                if r["human_answers"] and r["chatgpt_answers"]:
                    f.write(json.dumps({"group": f"hc3-{i}", "domain": r.get("source", "hc3"),
                                        "human": r["human_answers"], "ai": r["chatgpt_answers"]}) + "\n"); n += 1
                bar.update()
        bar.close(); tmp.replace(PAIRS / "hc3.jsonl"); ST.mark("pairs.hc3")
    else: print("  skip (done) hc3")

    if not ST.done("pairs.raid"):
        out = PAIRS / "raid.jsonl"
        have = set()
        if out.exists():                                       # resume: keep finished groups
            have = {json.loads(l)["group"] for l in open(out, encoding="utf-8") if l.strip().endswith("}")}
        human, ai = {}, {}
        bar = Progress("pairs/raid", PAIR_CAP, len(have), " groups")
        with open(out, "a", encoding="utf-8") as f:
            for r in stream("liamdugan/raid", "raid"):
                if len(have) >= PAIR_CAP: break
                if r["domain"] not in EN_RAID or r["attack"] != "none": continue
                sid = r["source_id"]; g = f"raid-{sid}"
                if g in have: continue
                if r["model"] == "human": human[sid] = r["generation"]
                else: ai.setdefault(sid, []).append(r["generation"])
                if sid in human and len(ai.get(sid, [])) >= 2:
                    f.write(json.dumps({"group": g, "domain": r["domain"],
                                        "human": [human.pop(sid)], "ai": ai.pop(sid)}) + "\n"); f.flush()
                    have.add(g); bar.set(len(have))
        bar.close(); ST.mark("pairs.raid")
    else: print("  skip (done) raid")

def pairs_extra():
    """Human-vs-human paraphrase sets (no AI text anywhere). Replaces PPDB as the source of
    phrase equivalents; build_phrase_pairs.py mines them the same way as HC3/RAID.
      coco : MSCOCO captions - 5 different people describe each image
      mrpc : Microsoft Research Paraphrase Corpus - news sentence pairs judged equivalent
      qqp  : Quora Question Pairs - questions marked duplicate by moderators"""
    pip("datasets>=2.19")
    PAIRS.mkdir(parents=True, exist_ok=True)
    def dump(name, rows, cap):
        if ST.done("pairs2." + name): print(f"  skip (done) {name}"); return
        tmp = PAIRS / f"{name}.jsonl.tmp"; n = 0
        bar = Progress(f"pairs/{name}", cap, unit=" groups")
        with open(tmp, "w", encoding="utf-8") as f:
            for i, texts in enumerate(rows):
                texts = [t.strip() for t in texts if t and t.strip()]
                if len(texts) < 2: continue
                f.write(json.dumps({"group": f"{name}-{i}", "domain": name, "human": texts, "ai": []}) + "\n")
                n += 1; bar.set(n)
                if n >= cap: break
        bar.close(); tmp.replace(PAIRS / f"{name}.jsonl"); ST.mark("pairs2." + name)
    from datasets import load_dataset
    jobs = {
        "coco": (lambda: (r["sentences"] for r in stream("yerevann/coco-karpathy")), 100000),
        "mrpc": (lambda: ([r["sentence1"], r["sentence2"]] for sp in ("train", "validation", "test")
                          for r in load_dataset("nyu-mll/glue", "mrpc", split=sp) if r["label"] == 1), 5000),
        "qqp":  (lambda: ([r["question1"], r["question2"]] for r in load_dataset("nyu-mll/glue", "qqp", split="train")
                          if r["label"] == 1), 150000),
    }
    for name, (gen, cap) in jobs.items():
        try: dump(name, gen(), cap)
        except Exception as e: ERR.append(1); print(f"  !! {name} failed: {e}")

# ---------------------------------------------------------------- models & lexical
def _check_long_paths():
    if sys.platform != "win32": return
    import winreg
    try:
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
        if winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1: return
    except OSError: pass
    raise RuntimeError(
        "Windows long paths are off, so PyTorch (needed by sentence-transformers) can't install.\n"
        "     Open PowerShell AS ADMINISTRATOR and run:\n"
        '     New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" '
        "-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force\n"
        "     then close this window, open a new one and rerun.")

def lexical():
    _check_long_paths()
    if not ST.done("lex.pip"):
        pip("spacy>=3.7", "nltk", "gensim", "wordfreq", "lemminflect", "sentence-transformers"); ST.mark("lex.pip")
    if not ST.done("lex.spacy"):
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"]); ST.mark("lex.spacy")
    import nltk
    if not ST.done("lex.wordnet"):
        for p in ("wordnet", "omw-1.4"): nltk.download(p, download_dir=str(DATA / "nltk_data"))
        ST.mark("lex.wordnet")
    import gensim.downloader as api
    vec = DATA / "models" / "glove-300.kv"
    if not vec.exists(): api.load("glove-wiki-gigaword-300").save(str(vec))
    # scorer/filter models for phrase mining (they only score, never write)
    from sentence_transformers import SentenceTransformer, CrossEncoder
    if not (DATA / "models" / "minilm").exists():
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").save(str(DATA / "models" / "minilm"))
    if not (DATA / "models" / "nli").exists():
        CrossEncoder("cross-encoder/nli-deberta-v3-small").save(str(DATA / "models" / "nli"))

def kenlm():
    base = "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/"
    for f in ("en.arpa.bin", "en.sp.model", "en.sp.vocab"):
        fetch(base + f, DATA / "models" / "kenlm" / f)
    pip("sentencepiece")
    if ST.done("kenlm.bindings"): return
    try:
        subprocess.check_call([sys.executable, str(Path(__file__).with_name("install_kenlm.py"))])
        ST.mark("kenlm.bindings"); return
    except Exception as e:
        print(f"  !! KenLM Python module failed to install ({e}). Non-fatal: the 4.4 GB model is downloaded and\n"
              "     the scorer skips the perplexity check until it works. Retry: python scripts/install_kenlm.py")

def ppdb():
    """PPDB is optional (its servers are down); never fail the build because of it."""
    dest = DATA / "dictionaries" / "ppdb-2.0-s-all.gz"
    if dest.exists(): print("  PPDB already present"); return
    for u in ("http://nlpgrid.seas.upenn.edu/PPDB/eng/ppdb-2.0-s-all.gz",
              "https://www.cis.upenn.edu/~ccb/ppdb/release-2.0/ppdb-2.0-s-all.gz"):
        try: fetch(u, dest); return
        except Exception as e: print(f"  (PPDB mirror unreachable: {str(e)[:80]})")
    print("  PPDB unavailable - skipping. The idiom dictionary uses WordNet + mined human pairs instead.")

def thesaurus():
    """Moby Thesaurus II (Grady Ward, public domain, human-compiled): ~30k roots, ~2.5M synonym links. Optional."""
    dest = DATA / "raw" / "moby_thesaurus.txt"
    if dest.exists() and dest.stat().st_size > 5_000_000: print("  Moby thesaurus already present"); return
    for u in ("https://www.gutenberg.org/cache/epub/3202/pg3202.txt", "https://www.gutenberg.org/files/3202/3202-0.txt",
              "https://www.gutenberg.org/files/3202/3202.txt"):
        try: fetch(u, dest); return
        except Exception as e: print(f"  (thesaurus mirror unreachable: {str(e)[:80]})")
    print("  Moby thesaurus unavailable - skipping (synonym swaps then use WordNet only).")

STEPS = dict(lexical=lexical, corpora=corpora, technical=technical, pairs=pairs, pairs_extra=pairs_extra, kenlm=kenlm, ppdb=ppdb, thesaurus=thesaurus)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=",".join(STEPS))
    ap.add_argument("--force", action="store_true", help="rerun the selected steps even if marked done")
    args = ap.parse_args()
    names = [n.strip() for n in args.only.split(",")]
    if args.force:
        for n in names: ST.set("step." + n, None)
    failed = []
    for k, name in enumerate(names):
        print(f"\n== download {k+1}/{len(names)} ({100*k//len(names)}% overall): {name}")
        if ST.done("step." + name):
            print("  skip (done)"); continue
        try:
            before = len(ERR); STEPS[name]()
            if len(ERR) > before: raise RuntimeError("some items failed")
            ST.mark("step." + name)
        except Exception as e:
            print(f"  !! step {name} failed: {e}  (rerun to resume)"); failed.append(name)
    print("\ndone." if not failed else f"\nfinished with failures: {failed} - rerun to retry them")
    sys.exit(1 if failed else 0)
