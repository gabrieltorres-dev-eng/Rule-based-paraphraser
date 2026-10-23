from .discourse import tag_roles, uniform_arc
from .thematic import break_chains
from .paragraph_ops import swap_bodies, relocate_conclusions, normalize_paragraphs, swap_sentences
from .. import intensity

def macro(doc, profile, res, rng):
    """Phase 2: discourse tagging, chain breaking, body swap, conclusion relocation, paragraph resampling."""
    tag_roles(doc)
    flag = uniform_arc(doc)
    if flag: doc.log.append({"phase": "macro", "transform": "uniform_arc_flagged"})
    break_chains(doc, rng)
    swap_bodies(doc, profile, rng, arc_flag=flag)
    relocate_conclusions(doc, profile, res, rng)
    if intensity.aggressive(): swap_sentences(doc, rng, p=intensity.p('sent_order', 0.5))
    normalize_paragraphs(doc, profile, rng)
    return doc
