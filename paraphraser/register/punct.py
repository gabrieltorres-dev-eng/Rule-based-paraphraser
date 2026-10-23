"""Punctuation entropy: parentheticals -> dash pairs, semicolons -> sentence breaks, at corpus-fitted fractions.
Quotes/code are placeholders so they are never touched."""
import re
from ..doc import PH_RE
from ..routing.util import upper_first, n_words
from ..routing.util import parse

PAREN = re.compile(r"\s\(([a-z][^()\d,;:]{6,60})\)(?=[\s,.;:!?])")   # lowercase aside only: no digits/commas -> never a citation

def _per_1k(doc, ch):
    words = max(sum(n_words(s) for _, _, s in doc.all_sentences()), 1)
    return 1000 * sum(s.count(ch) for _, _, s in doc.all_sentences()) / words

def adjust(doc, profile, rng):
    changed = 0
    # parentheticals -> em-dash pairs
    cur = _per_1k(doc, "("); target = profile.rate("paren_per_1k")
    if cur > target and cur > 0:
        p = (cur - target) / cur
        for pi, para in enumerate(doc.paragraphs):
            for si, s in enumerate(para.sentences):
                def sub(m):
                    nonlocal changed
                    if rng.random() < p:
                        changed += 1; return "—" + m.group(1).strip() + "—"
                    return m.group(0)
                new = PAREN.sub(sub, s)
                if new != s:
                    # em-dash pair replaces ' (x)' -> '—x—' ; keep spacing readable
                    new = re.sub(r"(\w)—", r"\1 —", new) if False else new
                    doc.note("register", "paren_to_dash", pi, si, s, new); para.sentences[si] = new
    # semicolons -> periods where both sides can stand alone
    cur = _per_1k(doc, ";"); target = profile.rate("semicolon_per_1k")
    if cur > target and cur > 0:
        p = (cur - target) / cur
        for pi, para in enumerate(doc.paragraphs):
            for si, s in enumerate(list(para.sentences)):
                if ";" not in s: continue
                parts = [x for x in re.split(r";\s+", s) if x]
                if len(parts) != 2 or n_words(parts[0]) < 4 or n_words(parts[1]) < 4: continue
                d2 = parse(parts[1].rstrip(".!?"))
                if not any(t.dep_ in ("nsubj", "nsubjpass", "expl") for t in d2): continue   # second half needs its own subject
                if rng.random() < p:
                    a = parts[0].rstrip(" ,") + "."
                    b = upper_first(parts[1])
                    para.sentences[si:si + 1] = [a, b]
                    doc.note("register", "semicolon_to_period", pi, si, s, a + " " + b); changed += 1
                    break
    return changed
