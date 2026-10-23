"""Phase 1: load-once resources. Every loader degrades gracefully if its data is missing."""
from __future__ import annotations
import csv, json, re, warnings
from collections import defaultdict
from pathlib import Path
from ..profile import ROOT

DATA = ROOT / "data"
TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")

# which registers each mined-pair source dataset is allowed in (casual caption/question wording stays out of formal text)
SOURCE_REGISTERS = {"coco": {"blog", "forum"}, "qqp": {"blog", "forum", "essay"},
                    "mrpc": None, "hc3": None, "raid": None, "ds": None}   # None = all registers

class Idioms:
    """Longest-match-first phrase -> equivalents dictionary, indexed by first token."""
    def __init__(self):
        self.by_first: dict[str, list[tuple[tuple, list]]] = defaultdict(list)   # first tok -> [(tokens, entries)]
        self.n_entries = 0

    def add(self, phrase: str, equiv: str, registers=None, score=1.0, freqs=None, source=""):
        toks = tuple(t.lower() for t in TOKEN.findall(phrase))
        if not toks or phrase.strip().lower() == equiv.strip().lower(): return
        bucket = self.by_first[toks[0]]
        for t, entries in bucket:
            if t == toks:
                entries.append({"text": equiv.strip(), "regs": registers, "score": score, "freqs": freqs or {}, "src": source})
                break
        else:
            bucket.append((toks, [{"text": equiv.strip(), "regs": registers, "score": score, "freqs": freqs or {}, "src": source}]))
        self.n_entries += 1

    def finalize(self):
        for k in self.by_first:
            self.by_first[k].sort(key=lambda x: -len(x[0]))      # longest first

    def match(self, lower_tokens: list[str], i: int):
        """Return (length, entries) for the longest phrase starting at token i, else None."""
        for toks, entries in self.by_first.get(lower_tokens[i], ()):
            n = len(toks)
            if tuple(lower_tokens[i:i + n]) == toks:
                return n, entries
        return None

    def allowed(self, entries, register: str):
        out = []
        for e in entries:
            if e["regs"] is not None and register not in e["regs"]: continue
            allow = SOURCE_REGISTERS.get(e["src"].split(":")[0], None) if e["src"] else None
            if allow is not None and register not in allow: continue
            out.append(e)
        return out

class TemplateBank:
    def __init__(self, rows=()):
        self.rows = []
        self.index = defaultdict(list)
        for r in rows: self.add(r)

    def add(self, r: dict):
        self.rows.append(r); self.index[(r["function"], r["register"])].append(r)

    def pick(self, function: str, register: str, rng, used: set, exclude=()):
        """Pick an unused template for a rhetorical function; fall back to other registers."""
        for reg in (register, None):
            pool = [r for r in (self.index[(function, register)] if reg else
                                [x for (f, _), v in self.index.items() if f == function for x in v])
                    if r["id"] not in used and r["text"] not in exclude]
            if pool:
                w = [max(r.get("docs", 1), 1) ** 0.5 for r in pool]
                r = pool[int(rng.choice(len(pool), p=[x / sum(w) for x in w]))]
                used.add(r["id"]); return r
        return None

from functools import lru_cache

@lru_cache(maxsize=200_000)
def _zipf(word: str) -> float:
    from wordfreq import zipf_frequency
    return zipf_frequency(word, "en")

class Resources:
    def __init__(self, data_dir: Path = DATA, load_heavy: bool = True, load_files: bool = True):
        self.dir = Path(data_dir)
        self.idioms = Idioms()
        self.templates = TemplateBank()
        self._wn = self._vec = self._kenlm = self._sp = None
        self.load_heavy = load_heavy
        if load_files:
            self._load_idioms()
            self._load_templates()

    # ---- dictionaries ----
    def _load_idioms(self, min_score=3.0):
        d = self.dir / "dictionaries"
        p = d / "idioms.tsv"
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
                regs = set(r["registers"].split(",")) if r.get("registers") else None
                try: freqs = json.loads(r.get("freqs") or "{}")
                except Exception: freqs = {}
                self.idioms.add(r["phrase"], r["equivalent"], regs, float(r.get("score") or 1), freqs, r.get("source", ""))
        p = d / "phrase_pairs.tsv"
        if p.exists() and self.idioms.n_entries == 0:   # raw mined pairs are unfiltered: only a fallback when idioms.tsv is missing
            for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
                srcs = (r.get("sources") or "").split(",")
                src = srcs[0].split(":")[0] if srcs and srcs[0] else ""
                score = 3.0 + 2 * float(r["nli"]) + min(float(r["ai_tell_z"]), 10) / 10
                self.idioms.add(r["ai_phrase"], r["human_phrase"], None, score, {}, src + ":mined")
        self.idioms.finalize()
        if self.idioms.n_entries == 0:
            warnings.warn("no idiom dictionary found (run scripts/build_all.py); phrase replacement disabled")

    def _load_templates(self):
        p = self.dir / "templates" / "bank.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                if line.strip(): self.templates.add(json.loads(line))
        else:
            warnings.warn("no template bank found; hedges/fragments/transitions disabled")

    # ---- lexical resources (lazy) ----
    @property
    def thesaurus(self):
        """root -> set of single-word synonyms (Moby, human-compiled). Empty dict if the file was not built."""
        if getattr(self, "_thes", None) is None:
            self._thes = {}
            p = self.dir / "dictionaries" / "thesaurus.tsv"
            if p.exists():
                for line in open(p, encoding="utf-8"):
                    r, _, rest = line.rstrip("\n").partition("\t")
                    if rest: self._thes[r] = set(rest.split())
        return self._thes

    @property
    def glue(self):
        """Human-attested glue phrases (count >= 5 in the human corpora). Empty set if build_glue.py has not run."""
        if getattr(self, "_glue", None) is None:
            self._glue = set()
            p = self.dir / "dictionaries" / "glue.tsv"
            if p.exists():
                for line in open(p, encoding="utf-8"):
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 2 and int(parts[1]) >= 5: self._glue.add(parts[0])
            else:
                warnings.warn("glue.tsv missing (run scripts/build_glue.py): glue-word rewrites disabled")
        return self._glue

    def mutual_synonyms(self, word):
        """Words that list `word` AND are listed by it."""
        t = self.thesaurus
        return [w for w in t.get(word, ()) if word in t.get(w, ())]

    @property
    def wordnet(self):
        if self._wn is None:
            import nltk
            nltk.data.path.insert(0, str(self.dir / "nltk_data"))
            from nltk.corpus import wordnet
            try: wordnet.ensure_loaded()
            except LookupError:
                import os; os.environ["NLTK_ALLOW_PROXIED_URLOPEN"] = "1"
                nltk.download("wordnet", download_dir=str(self.dir / "nltk_data"), quiet=True)
                wordnet.ensure_loaded()
            self._wn = wordnet
        return self._wn

    @property
    def vectors(self):
        if self._vec is None:
            p = self.dir / "models" / "glove-300.kv"
            if not p.exists():
                warnings.warn("GloVe vectors missing; synonym semantic filter disabled"); self._vec = False
            else:
                from gensim.models import KeyedVectors
                self._vec = KeyedVectors.load(str(p), mmap="r")
        return self._vec or None

    def zipf(self, word: str) -> float:
        return _zipf(word)

    @property
    def kenlm(self):
        """(kenlm model, sentencepiece processor) or None."""
        if self._kenlm is None:
            try:
                import kenlm, sentencepiece
                d = self.dir / "models" / "kenlm"
                try:                                    # mmap lazily: the 4.4 GB model is paged in on demand instead of read up front
                    cfg = kenlm.Config(); cfg.load_method = kenlm.LoadMethod.LAZY
                    self._kenlm = kenlm.Model(str(d / "en.arpa.bin"), cfg)
                except Exception:
                    self._kenlm = kenlm.Model(str(d / "en.arpa.bin"))
                self._sp = sentencepiece.SentencePieceProcessor(str(d / "en.sp.model"))
            except Exception as e:
                warnings.warn(f"KenLM unavailable ({type(e).__name__}); perplexity gate skipped"); self._kenlm = False
        return (self._kenlm, self._sp) if self._kenlm else None

    def perplexity(self, text: str):
        km = self.kenlm
        if not km: return None
        model, sp = km
        toks = " ".join(sp.encode(text.lower(), out_type=str))
        return model.perplexity(toks)
