from . import contractions, punct, fragments, hedges

def tells(doc, profile, res, rng):
    """Phase 6: contractions, punctuation entropy, fragments, hedges (rule/template based only)."""
    contractions.adjust(doc, profile, rng)
    punct.adjust(doc, profile, rng)
    fragments.apply(doc, profile, rng)
    hedges.apply(doc, profile, res, rng)
    return doc
