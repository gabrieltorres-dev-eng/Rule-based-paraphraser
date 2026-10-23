import numpy as np
from paraphraser.resources import Resources, Idioms, TemplateBank, TOKEN

def test_idiom_longest_match():
    i = Idioms(); i.add("look into", "examine"); i.add("look into the matter", "investigate it"); i.finalize()
    toks = [t.lower() for t in TOKEN.findall("They look into the matter now")]
    n, e = i.match(toks, 1)
    assert n == 4 and e[0]["text"] == "investigate it"

def test_register_gating():
    i = Idioms(); i.add("a man", "a guy", None, 4, {}, "coco:mined"); i.finalize()
    toks = ["a", "man"]
    n, e = i.match(toks, 0)
    assert i.allowed(e, "blog") and not i.allowed(e, "academic")

def test_templates_no_reuse():
    tb = TemplateBank([{"id": "h1", "text": "Arguably", "function": "hedge", "register": "essay", "docs": 5},
                       {"id": "h2", "text": "Perhaps", "function": "hedge", "register": "essay", "docs": 5}])
    used, rng = set(), np.random.default_rng(0)
    a = tb.pick("hedge", "essay", rng, used); b = tb.pick("hedge", "essay", rng, used); c = tb.pick("hedge", "essay", rng, used)
    assert a["id"] != b["id"] and c is None

def test_resources_load_real_data_gracefully():
    r = Resources()
    assert r.idioms is not None and r.templates is not None
    assert r.zipf("house") > 4
