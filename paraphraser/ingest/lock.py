"""Replace protected spans (quotes, code, URLs, numbers, named entities, technical terms) with
placeholders so no later phase can change them."""
import re
from ..doc import get_nlp, PH_RE

QUOTE = re.compile(r'"[^"\n]{1,400}"|“[^”\n]{1,400}”')
CODE = re.compile(r"`[^`\n]+`")
URL = re.compile(r"(?:https?://|www\.)\S+[^\s.,;:!?)\]]|\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NUM = re.compile(r"(?<![\w.])[$€£]?\d[\d,]*(?:\.\d+)?(?:%|\s?(?:percent|million|billion|thousand))?(?![\w])")
TECH = re.compile(r"\b(?:[A-Za-z]+_[A-Za-z_0-9]+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Z]{2,}[a-z0-9]*s?|\w+\.\w+\(\)?|[\w./\\-]+\.(?:py|js|txt|md|json|csv|html|cpp|java)|\w*\d\w*)\b")
ENT_LOCK = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT", "LAW", "LANGUAGE"}
_ENT_DIGIT = {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL"}

def _merge(spans):
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out = []
    for a, b, why in spans:
        if out and a < out[-1][1]:
            if b > out[-1][1]: out[-1] = (out[-1][0], b, out[-1][2])
        else:
            out.append((a, b, why))
    return out

def sentence_spans(sent: str, nlp):
    spans = []
    if sent.count('"') % 2 == 0:
        spans += [(m.start(), m.end(), "quote") for m in QUOTE.finditer(sent)]
    else:
        spans += [(m.start(), m.end(), "quote") for m in re.finditer(r"“[^”]{1,400}”", sent)]
    spans += [(m.start(), m.end(), "code") for m in CODE.finditer(sent)]
    spans += [(m.start(), m.end(), "url") for m in URL.finditer(sent)]
    spans += [(m.start(), m.end(), "number") for m in NUM.finditer(sent)]
    for m in TECH.finditer(sent):
        w = m.group(0)
        if m.start() == 0 and w.isalpha() and not w.isupper(): continue
        if w in ("I", "A", "OK"): continue
        spans.append((m.start(), m.end(), "technical"))
    for e in nlp(sent).ents:
        if e.label_ in ENT_LOCK or (e.label_ in _ENT_DIGIT and any(ch.isdigit() for ch in e.text)):
            spans.append((e.start_char, e.end_char, "entity"))
    return _merge(spans)

REF_HEAD = re.compile(r"^\s*(references?|bibliography|works cited|sources|citations?|footnotes?|endnotes?|appendix)\s*:?\s*$", re.I)

def lock_reference_sections(doc):
    """A 'References' (etc.) heading locks itself and everything after it, untouched."""
    for pi, p in enumerate(doc.paragraphs):
        if len(p.sentences) == 1 and REF_HEAD.match(doc.restore(p.sentences[0])):
            for q in doc.paragraphs[pi:]:
                q.sentences = [doc.lock(" ".join(q.sentences), "reference")]
            return pi
    return None

def lock_protected(doc):
    lock_reference_sections(doc)
    nlp = get_nlp()
    for p in doc.paragraphs:
        for i, s in enumerate(p.sentences):
            if PH_RE.fullmatch(s.strip()): continue
            spans = [sp for sp in sentence_spans(s, nlp)]
            for a, b, why in reversed(spans):
                if PH_RE.fullmatch(s[a:b]): continue
                ph = doc.lock(s[a:b], why)
                s = s[:a] + ph + s[b:]
            p.sentences[i] = s
