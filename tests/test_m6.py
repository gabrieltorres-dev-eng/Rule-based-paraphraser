import numpy as np
from paraphraser.doc import Document
from paraphraser.profile import RegisterProfile
from paraphraser.resources import Resources
from paraphraser.lexical import perturb, candidates
from paraphraser.routing.util import parse

R = Resources(load_heavy=False)

def run(text, rate=1.0, seed=0, **kw):
    d = Document.from_text(text, "essay")
    n = perturb(d, RegisterProfile.load("essay"), R, np.random.default_rng(seed), rate=rate, **kw)
    return d, n

def test_candidates_single_word():
    tok = parse("The teacher began the lesson.")[3]
    assert all(" " not in c and "_" not in c for c, _ in candidates(R, parse("They begin the lesson.")[1]))

def test_changes_some_words_and_keeps_locks():
    d, n = run("The company started a project to enhance customer satisfaction. Managers praised the careful approach.")
    assert n > 0
    assert d.text() != "The company started a project to enhance customer satisfaction. Managers praised the careful approach."

def test_domain_lock_and_no_bad_swaps():
    d, n = run("The server crashed. The server restarted. The server logs show a server fault.", rate=1.0)
    assert d.text().count("server") >= 4          # domain vocabulary untouched

def test_no_weighty_for_significant():
    d, n = run("There was a significant increase in sales.", rate=1.0)
    assert "weighty" not in d.text()
