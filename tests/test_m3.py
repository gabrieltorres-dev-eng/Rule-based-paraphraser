import numpy as np
from paraphraser.doc import Document
from paraphraser.resources import Resources
from paraphraser.ingest import ingest
from paraphraser.profile import RegisterProfile

def make(res_idioms=()):
    r = Resources(load_heavy=False, load_files=False)
    for p, e in res_idioms: r.idioms.add(p, e, None, 5.0)
    r.idioms.finalize()
    return r

def run(text, idioms=(), register="essay"):
    d = Document.from_text(text, register)
    ingest(d, make(idioms), RegisterProfile.load(register), np.random.default_rng(1))
    return d

def test_quote_number_entity_locked():
    d = run('Tom Hanks said "we delve into it" in 1998. It cost $4,500 and 12% more.', [("delve into", "dig into")])
    out = d.text(restored=False)
    assert "delve" not in out and "1998" not in out and "$4,500" not in out and "Tom Hanks" not in out
    assert d.text() == 'Tom Hanks said "we delve into it" in 1998. It cost $4,500 and 12% more.'

def test_idiom_replaced_and_locked():
    d = run("We need to delve into the data today.", [("delve into", "look at")])
    assert d.text() == "We need to look at the data today."
    assert "delve" not in d.text()

def test_longest_match_and_case():
    d = run("Delve into it. Then look into the matter.", [("delve into", "dig into"), ("look into", "examine"), ("look into the matter", "check it out")])
    assert d.text() == "Dig into it. Then check it out."

def test_code_and_technical_locked():
    d = run("Call `foo_bar(x)` and then use myFunction to delve into logs.\n\n```\ndelve into code\n```", [("delve into", "look at")])
    assert "foo_bar(x)" in d.text() and "myFunction" in d.text() and "delve into code" in d.text()
    assert "look at logs" in d.text()

def test_reference_section_locked():
    d = run("Some intro text here.\n\nReferences\n\nCrundall, D. (2017, February 1). Why using a phone is so dangerous. The Conversation.")
    assert "Crundall" not in d.text(restored=False)
    assert "Crundall, D. (2017, February 1)." in d.text()
