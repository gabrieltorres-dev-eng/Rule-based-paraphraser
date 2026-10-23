from .lock import lock_protected
from .idioms import replace_idioms

def ingest(doc, resources, profile, rng, log_phase="ingest", lock=True):
    """Phase 3: lock protected spans, then deterministic longest-match idiom replacement."""
    if lock: lock_protected(doc)          # locking is deterministic: the pipeline does it once on the base document
    replace_idioms(doc, resources, rng)
    return doc
