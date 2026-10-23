import numpy as np
from paraphraser.doc import Document
from paraphraser.profile import RegisterProfile
from paraphraser.resources import Resources
from paraphraser.macro import discourse as D, paragraph_ops as P, macro
from paraphraser.macro.thematic import break_chains

PROF = RegisterProfile.load("default")

def doc(text, reg="essay"):
    d = Document.from_text(text, reg); D.tag_roles(d); return d

def test_roles_and_linked():
    d = doc("My Title\n\nIntro sentence here.\n\nBackground text about the topic.\n\nMore detail follows now.\n\nWe should act now to fix it.\n\nIn conclusion, it ends.")
    assert [p.role for p in d.paragraphs][0] == "title" and d.paragraphs[1].role == "intro" and d.paragraphs[-1].role == "conclusion"
    assert D.is_linked("Because of this, things changed.") and D.is_linked("This shows the point.") and not D.is_linked("Dogs bark loudly.")

def test_swap_blocked_by_causal_and_pronoun():
    d = doc("Intro here.\n\nDogs are loyal animals that guard homes.\n\nCats are independent pets that hunt mice.\n\nBirds can fly very high above trees.\n\nEnd it all now.")
    rng = np.random.default_rng(0); prof = RegisterProfile({**PROF.d, "rates": {**PROF.d["rates"], "para_swap": 1.0}})
    P.swap_bodies(d, prof, rng)
    assert d.paragraphs[0].sentences[0] == "Intro here." and d.paragraphs[-1].sentences[0] == "End it all now."
    d2 = doc("Intro here.\n\nDogs are loyal animals that guard homes.\n\nBecause of that, cats are independent pets.\n\nEnd it all now.")
    assert P.swap_bodies(d2, prof, rng) == 0

def test_conclusion_relocation_strips_marker():
    text = ("Cities grow fast every year. Traffic gets worse as more people arrive. Public transit could reduce traffic a lot. "
            "Many cities underinvest in transit systems. In conclusion, cities must invest in transit to reduce traffic.")
    d = doc("Intro line here.\n\n" + text + "\n\nEnd it all now.")
    prof = RegisterProfile({**PROF.d, "rates": {**PROF.d["rates"], "early_thesis": 1.0}})
    r = Resources(load_heavy=False)
    P.relocate_conclusions(d, prof, r, np.random.default_rng(1))
    s = d.paragraphs[1].sentences
    assert s[1].startswith("Cities must invest in transit") and not any("In conclusion" in x for x in s)

def test_break_chain():
    d = doc("Intro line here.\n\nThe committee reviewed the budget. The committee approved the plan. The committee announced the decision.\n\nEnd it all now.")
    n = break_chains(d, np.random.default_rng(0))
    assert n == 1 and "was announced by the committee" in d.paragraphs[1].sentences[2] or "announced" in d.paragraphs[1].sentences[2]

def test_macro_runs():
    d = Document.from_text("Title Here\n\nIntro text goes here. It sets up the topic.\n\nFirst body paragraph is about dogs. Dogs bark. Dogs run fast. Dogs love people.\n\nSecond body paragraph is about cats. Cats nap. Cats hunt. Cats purr.\n\nIn conclusion, pets are great for families.")
    macro(d, RegisterProfile.load("essay"), Resources(load_heavy=False), np.random.default_rng(3))
    assert d.text().count("\n\n") >= 3
