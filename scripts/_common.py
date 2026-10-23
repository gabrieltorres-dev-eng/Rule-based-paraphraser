"""Shared helpers for the data-build scripts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
REGISTERS = ("blog", "essay", "forum", "technical", "academic")
PAIRS = DATA / "pairs"

def corpus_docs(register):
    """Yield (doc_id, text) for every calibration file of a register."""
    d = CORPUS / register
    if not d.exists():
        return
    for p in sorted(d.glob("*.txt")):
        yield p.stem, p.read_text(encoding="utf-8", errors="ignore")

def load_nlp():
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except OSError:
        sys.exit("spaCy model missing: run scripts/download_data.py --only lexical")
    nlp.max_length = 2_000_000
    return nlp

def need(path: Path, hint: str):
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        sys.exit(f"Missing {path}. {hint}")
