"""Provenance gate for closed-class glue words. A rewrite may insert a word or phrase only if the HUMAN corpora attest it
(data/dictionaries/glue.tsv, built by scripts/build_glue.py). Fails closed: if the file is missing, nothing is inserted."""
_OK: set = set()

def set_glue(phrases):
    _OK.clear(); _OK.update(p.lower() for p in phrases)

def ok(*phrases) -> bool:
    return all(p.lower() in _OK for p in phrases)
