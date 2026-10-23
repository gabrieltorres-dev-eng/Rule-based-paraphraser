"""Aggressive within-sentence restructuring: reverse 3-item lists ("slopes, bends and corners" -> "corners, bends and slopes")."""
import re
from ..doc import PH_RE
from .util import parse

OK_POS = {"NOUN", "ADJ"}

def reverse_lists(doc, rng, p=0.85):
    n = 0
    for pi, para in enumerate(doc.paragraphs):
        for si, s in enumerate(para.sentences):
            if PH_RE.search(s) and PH_RE.fullmatch(s.strip()): continue
            d = parse(s)
            toks = list(d)
            i, edits = 0, []
            while i + 4 < len(toks):
                a, c1, b = toks[i], toks[i + 1], toks[i + 2]
                j = i + 3
                if j < len(toks) and toks[j].text == ",": j += 1                      # optional Oxford comma
                if (c1.text == "," and j < len(toks) and toks[j].lower_ in ("and", "or") and j + 1 < len(toks)):
                    cc, z = toks[j], toks[j + 1]
                    items = (a, b, z)
                    if (all(t.pos_ in OK_POS and t.is_alpha and not PH_RE.search(t.text) and not t.ent_type_ for t in items)
                            and len({t.pos_ for t in items}) == 1
                            and not any(t.dep_ in ("compound", "amod", "poss") for t in items)       # skip modifiers inside a bigger noun phrase
                            and rng.random() < p):
                        w = [t.text for t in items][::-1]
                        if a.text[:1].isupper() and a.i == 0: w[0] = w[0][:1].upper() + w[0][1:]; w[2] = w[2][:1].lower() + w[2][1:]
                        edits.append((a.idx, z.idx + len(z.text), f"{w[0]}, {w[1]}{',' if toks[j-1].text == ',' else ''} {cc.text} {w[2]}"))
                        i = j + 2; continue
                i += 1
            if edits:
                out, last = [], 0
                for st, en, rep in edits:
                    out.append(s[last:st] + rep); last = en
                new = "".join(out) + s[last:]
                if not doc.fresh(new): continue
                doc.note("routing", "reverse_list", pi, si, s, new)
                para.sentences[si] = new; n += 1
    return n
