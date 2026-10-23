import numpy as np
from ..doc import PH_RE
from ..profile import RegisterProfile
from .clause import transpose
from .passive import passivize
from .length import sculpt
from .util import parse
from ..calibration.measure import is_passive
from .. import intensity
from .shuffle import reverse_lists
from .extra import move_pp, swap_coords
from .merge import merge_neighbours, relative_clauses, vary_openings
from .inner import apply_inner
from .split_plus import split_long
from .complex import apply_complex

def _rate_from(cur, target, eligible_frac, pos_mult=1.0, cap=0.6):
    return float(np.clip((target - cur) / max(eligible_frac, 0.05) * pos_mult, 0, cap))

def route(doc, profile: RegisterProfile, res, rng, do_sculpt=True):
    """Phase 4: clause transposition, passive inversion, sentence-length sculpting."""
    sents = [(pi, si) for pi, p in enumerate(doc.paragraphs) for si in range(len(p.sentences))]
    total = max(len(sents), 1)
    # ---- clause transposition (direction chosen to move the doc toward the human adverbial-fronting rate)
    opts = {}
    for pi, si in sents:
        s = doc.paragraphs[pi].sentences[si]
        if PH_RE.fullmatch(s.strip()): continue
        r = transpose(s)
        if r: opts[(pi, si)] = r
    cur_front = sum(1 for k, v in opts.items() if v[1] == "to_back") / total       # fronted now
    target = profile.rate("adverbial_front")
    want = "to_front" if cur_front < target else "to_back"
    cand = {k: v for k, v in opts.items() if v[1] == want}
    if intensity.aggressive():                      # flip EVERY eligible clause order, in both directions
        for (pi, si), (new, _) in opts.items():
            if rng.random() < intensity.p('clause', 0.9) and doc.fresh(new):
                old = doc.paragraphs[pi].sentences[si]
                doc.paragraphs[pi].sentences[si] = new
                doc.note("routing", "clause_transpose", pi, si, old, new)
        cand = {}
    p_base = _rate_from(cur_front, target, len(cand) / total) if want == "to_front" else \
        float(np.clip((cur_front - target) / max(len(cand) / total, 0.05), 0, 0.6))
    ov = max(profile.rate("adverbial_front"), 1e-6)
    for (pi, si), (new, _) in cand.items():
        p = p_base * profile.pos_rate("adverbial_front", doc.pos_bucket(pi, si)) / ov
        if rng.random() < p and doc.fresh(new):
            old = doc.paragraphs[pi].sentences[si]
            doc.paragraphs[pi].sentences[si] = new
            doc.note("routing", "clause_transpose", pi, si, old, new)
    # ---- passive inversion
    cur_pas = sum(is_passive(parse(s)) for _, _, s in doc.all_sentences()) / total
    target = profile.rate("passive")
    pas = {}
    for pi, si in sents:
        s = doc.paragraphs[pi].sentences[si]
        r = passivize(s)
        if r: pas[(pi, si)] = r
    p_base = _rate_from(cur_pas, target, len(pas) / total, cap=0.5)
    if intensity.aggressive(): p_base, target = intensity.p('voice', 0.8), 0.8
    
    ov = max(profile.rate("passive"), 1e-6)
    for (pi, si), new in pas.items():
        if rng.random() < p_base * profile.pos_rate("passive", doc.pos_bucket(pi, si)) / ov:
            old = doc.paragraphs[pi].sentences[si]
            doc.paragraphs[pi].sentences[si] = new
            doc.note("routing", "passive", pi, si, old, new)
    # ---- length sculpting
    if intensity.aggressive():
        apply_inner(doc, rng, p=intensity.p('inner', 0.9))
        split_long(doc, res, rng, p=intensity.p('split', 0.7))
        apply_complex(doc, rng, p=intensity.p('complex', 0.7))
        reverse_lists(doc, rng, p=intensity.p('list', 0.85))
        for pi, si in sents:                              # prepositional-phrase moves and clause swaps on whatever is still unchanged
            s0 = doc.paragraphs[pi].sentences[si]
            if PH_RE.fullmatch(s0.strip()) or rng.random() >= intensity.p('phrase', 0.9): continue
            r = move_pp(s0)
            if r: new, tr = r[0], "pp_move"
            else:
                new, tr = (swap_coords(s0) if rng.random() < intensity.p("coord", 0.9) else None), "swap_coords"
            if new and doc.fresh(new):
                doc.paragraphs[pi].sentences[si] = new
                doc.note("routing", tr, pi, si, s0, new)
    if intensity.aggressive():                            # cross-sentence layers run last: they change sentence counts
        vary_openings(doc, rng, p=intensity.p('opening', 0.8), p_any=intensity.p('opening', 0.25))
        relative_clauses(doc, rng, p=intensity.p('relcl', 0.7))
        merge_neighbours(doc, rng, p=intensity.p('merge', 0.6))
    if do_sculpt: sculpt(doc, profile, res, rng, max_frac=min(0.9, 0.6 * max(intensity.RECIPE.get('burst', 1.0), 0.5) * intensity.BOOST) if intensity.aggressive() else 0.30)
    return doc
