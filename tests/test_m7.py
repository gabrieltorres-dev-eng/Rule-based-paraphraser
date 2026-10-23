import numpy as np
from paraphraser.doc import Document
from paraphraser.profile import RegisterProfile
from paraphraser.resources import Resources, TemplateBank
from paraphraser.register import contractions, punct, fragments, hedges, tells

P = RegisterProfile.load("default")
def prof(**rates): return RegisterProfile({**P.d, "rates": {**P.d["rates"], **rates}})
def doc(t): return Document.from_text(t, "essay")
RNG = lambda s=0: np.random.default_rng(s)

def test_contract_up():
    d = doc("We do not know. It is not clear. They are not sure why.")
    contractions.adjust(d, prof(contraction=1.0), RNG())
    assert d.text() == "We don't know. It isn't clear. They aren't sure why." or "don't" in d.text()

def test_expand_down_and_end_guard():
    d = doc("We don't know. They aren't sure. Yes, it is.")
    contractions.adjust(d, prof(contraction=0.0), RNG())
    assert d.text() == "We do not know. They are not sure. Yes, it is."
    d2 = doc("Yes it is. I think it is fine.")
    contractions.adjust(d2, prof(contraction=1.0), RNG())
    assert "Yes it is." in d2.text() and "it's fine" in d2.text()

def test_paren_to_dash_and_semicolon():
    d = doc("The plan (which was bold) worked well. It rained all day; the game was cancelled by officials.")
    punct.adjust(d, prof(paren_per_1k=0.0, semicolon_per_1k=0.0), RNG())
    t = d.text()
    assert "—which was bold—" in t and ";" not in t and "day. The game" in t

def test_fragment_only_midparagraph():
    d = doc("Opening line is here. This is not always true. Closing line is here.")
    n = fragments.apply(d, prof(fragment=1.0), RNG())
    assert n == 1 and "Not always true." in d.text()
    d2 = doc("This is not always true. Another sentence follows it.")
    assert fragments.apply(d2, prof(fragment=1.0), RNG()) == 0

def test_hedge_from_bank_only_and_no_reuse():
    bank = TemplateBank([{"id": "h1", "text": "In many cases,", "function": "hedge", "register": "essay", "docs": 5}])
    res = Resources(load_heavy=False); res.templates = bank
    d = doc("Intro sentence is here. Everyone always gets this wrong in the end. Nobody ever learns from the first mistake they make.")
    n = hedges.apply(d, prof(hedge_front=1.0), res, RNG())
    assert n == 1 and "In many cases, everyone always gets this wrong" in d.text()   # second claim: bank exhausted, no reuse
    d2 = doc("Intro sentence is here. Studies of 40 patients found that sleep helps recovery after surgery.")
    assert hedges.apply(d2, prof(hedge_front=1.0), res, RNG()) == 0                 # not an absolute claim

def test_citations_never_dashed():
    d = doc("Crashes rose sharply (Crundall, 2017). Other studies (see below) agree (2019). Officials (who were present) agreed.")
    punct.adjust(d, prof(paren_per_1k=0.0), RNG())
    t = d.text()
    assert "(Crundall, 2017)" in t and "(2019)" in t
