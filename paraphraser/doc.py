"""Core data model. Locked spans (quotes, numbers, entities, code, URLs, replaced idioms) are
swapped for opaque placeholder tokens (e.g. QZ3X) so no later transform can alter them; they
are restored at the very end. Placeholders look like capitalised proper nouns to the parser."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

_NLP = None

def merge_bad_splits(sents):
    """Re-join spaCy splits that fall inside parentheses/brackets or before a bracket/lowercase start."""
    out = []
    for t in sents:
        if out and (out[-1].count("(") > out[-1].count(")") or out[-1].count("[") > out[-1].count("]")
                    or t[:1] in "[)]" or t[:1].islower()
                    or not re.search(r"[.!?:;\"'\u201d\u2019)\]]\s*$", out[-1])):
            out[-1] += " " + t
        else:
            out.append(t)
    return out

def get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError as e:
            raise SystemExit("spaCy model missing: python -m spacy download en_core_web_sm") from e
        _NLP.max_length = 2_000_000
    return _NLP

PH_RE = re.compile(r"\bQZ(\d+)X\b")

@dataclass
class Paragraph:
    sentences: list[str]
    role: str = ""            # discourse role, filled by macro.discourse

@dataclass
class Document:
    paragraphs: list[Paragraph]
    register: str = "essay"
    locks: dict[str, tuple[str, str]] = field(default_factory=dict)   # placeholder -> (original, reason)
    log: list[dict] = field(default_factory=list)
    used_templates: set[str] = field(default_factory=set)
    uid: str = ""

    # ---- construction -------------------------------------------------
    @classmethod
    def from_text(cls, text: str, register: str = "essay", uid: str = "") -> "Document":
        text = text.replace("\r\n", "\n").strip()
        d = cls([], register, uid=uid)
        text = re.sub(r"```.*?```", lambda m: "\n\n" + d.lock(m.group(0), "code") + "\n\n", text, flags=re.S)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        chunks = re.split(r"\n\s*\n", text) if "\n\n" in text else text.split("\n")
        nlp = get_nlp()
        paras = []
        for c in chunks:
            c = " ".join(c.split())
            if not c: continue
            sents = merge_bad_splits([s.text.strip() for s in nlp(c).sents if s.text.strip()])
            if sents: paras.append(Paragraph(sents))
        d.paragraphs = paras
        return d

    # ---- locks ---------------------------------------------------------
    def lock(self, original: str, reason: str) -> str:
        ph = f"QZ{len(self.locks)}X"
        self.locks[ph] = (original, reason)
        return ph

    def restore(self, s: str) -> str:
        for _ in range(3):                      # placeholders may nest (idiom containing a lock)
            new = PH_RE.sub(lambda m: self.locks.get(m.group(0), (m.group(0),))[0], s)
            if new == s: break
            s = new
        return s

    # ---- views ---------------------------------------------------------
    def all_sentences(self):
        for pi, p in enumerate(self.paragraphs):
            for si, s in enumerate(p.sentences):
                yield pi, si, s

    def text(self, restored: bool = True, sep: str = "\n\n") -> str:
        out = [" ".join(p.sentences) for p in self.paragraphs if p.sentences]
        t = sep.join(out)
        return self.restore(t) if restored else t

    def clone(self) -> "Document":
        return Document([Paragraph(list(p.sentences), p.role) for p in self.paragraphs], self.register,
                        dict(self.locks), [], set(self.used_templates), self.uid)

    def fresh(self, new: str) -> bool:
        """True unless `new` (restored) equals a sentence form from an earlier pass: stops repeat passes from undoing themselves."""
        from . import intensity
        return not intensity.SEEN or self.restore(new) not in intensity.SEEN

    def note(self, phase: str, transform: str, pi: int, si: int, before: str, after: str):
        self.log.append({"phase": phase, "transform": transform, "para": pi, "sent": si,
                         "before": self.restore(before), "after": self.restore(after)})

    def pos_bucket(self, pi: int, si: int) -> str:
        n = len(self.paragraphs[pi].sentences)
        return "first" if si == 0 else ("last" if si == n - 1 else "middle")
