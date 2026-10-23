"""Phase 7: rule-based grammar healing for true leftovers (a/an, agreement after swaps, spacing,
capitalisation, duplicated words). Runs on restored text segments, never inside locks."""
import re
from ..doc import PH_RE

VOWEL_SOUND = re.compile(r"^(?:[aeiou]|hour|honest|honor|heir)", re.I)
CONS_EXC = re.compile(r"^(?:uni(?:[^n]|$)|use|usu|eu|one|once|ubiq|uri|util)", re.I)
VOWEL_EXC = re.compile(r"^(?:[fhlmnrsx]\b|hono|hour|heir)", re.I)
QPH = re.compile(r"^QZ\d+X$")

def fix_articles(s: str, restore) -> str:
    def sub(m):
        art, sp, word = m.group(1), m.group(2), m.group(3)
        w = restore(word) if QPH.match(word) else word
        if not w: return m.group(0)
        if w[:1].isupper() and len(w) > 1 and w[:2].isupper() and not QPH.match(word): return m.group(0)  # acronyms
        vowel = bool(VOWEL_SOUND.match(w)) and not CONS_EXC.match(w)
        a = art.lower()
        new = ("an" if vowel else "a")
        if a not in ("a", "an") or new == a: return m.group(0)
        return (new.capitalize() if art[0].isupper() else new) + sp + word
    return re.sub(r"\b([Aa]n?)(\s+)([A-Za-z][\w'-]*)", sub, s)

ELL = "\u0001"

def fix_spacing(s: str) -> str:
    s = s.replace("...", ELL).replace("\u2026", ELL)            # protect ellipses
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"(?<!\u0001)\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])(?=[A-Za-z])", r"\1 ", s)
    s = re.sub(r"([.!?])(?=[A-Z][a-z])", r"\1 ", s)
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\.\s*\.(?!\.)", ".", s)
    s = re.sub(r"\(\s+", "(", s); s = re.sub(r"\s+\)", ")", s)
    s = s.replace(ELL, "...")
    return s.strip()

def fix_dups(s: str) -> str:
    return re.sub(r"\b(\w+)(\s+\1\b)+", lambda m: m.group(1) if m.group(1).lower() not in ("had", "that", "very") else m.group(0), s, flags=re.I)

ABBREV = re.compile(r"(?:\be\.g|\bi\.e|\betc|\bvs|\bMr|\bMrs|\bMs|\bDr|\bProf|\bSt|\bNo|\bFig|\bU\.S|\bcf|\bet al|\b[A-Za-z])\.$")

def fix_caps(s: str) -> str:
    s = re.sub(r"^(\W*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    def sub(m):
        before = s[:m.start(1) + 1]
        nxt = s[m.end(2) - 1:].split(" ")[0]
        if before.endswith("..") or ABBREV.search(before) or "/" in nxt or "http" in before[-40:] or "www" in before[-40:]: return m.group(0)
        return m.group(1) + m.group(2).upper()
    return re.sub(r"([.!?]\s+)([a-z])", sub, s)

AGREE = [(re.compile(r"\b(he|she|it) (are|were)\b", re.I), lambda m: m.group(1) + (" is" if m.group(2).lower() == "are" else " was")),
         (re.compile(r"\b(they|we|you) (is|was)\b", re.I), lambda m: m.group(1) + (" are" if m.group(2).lower() == "is" else " were")),
         (re.compile(r"\bthere are (a|an|one) ", re.I), lambda m: "there is " + m.group(1) + " "),
         (re.compile(r"\b(this|that) (are)\b", re.I), lambda m: m.group(1) + " is"),
         (re.compile(r"\b(these|those) (is)\b", re.I), lambda m: m.group(1) + " are")]

def heal_sentence(s: str, doc) -> str:
    s = fix_spacing(s)
    s = fix_articles(s, doc.restore)
    for rx, fn in AGREE: s = rx.sub(fn, s)
    s = fix_dups(s)
    return fix_caps(s)

def heal(doc, profile=None, res=None, rng=None):
    for pi, p in enumerate(doc.paragraphs):
        for si, s in enumerate(p.sentences):
            if PH_RE.fullmatch(s.strip()): continue
            new = heal_sentence(s, doc)
            if new != s:
                doc.note("healing", "heal", pi, si, s, new); p.sentences[si] = new
    return doc
