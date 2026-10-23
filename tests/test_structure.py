import numpy as np
from paraphraser import structure as st

def test_synonym_swap_scores_zero():
    a = "The committee approved the new budget after a long debate on Tuesday."
    b = "The committee approved the new budget after a long debate on Tuesday."
    assert st.compare(a, b)["ted"] == 0.0

def test_restructure_scores_higher():
    a = "The committee approved the new budget after a long debate on Tuesday."
    b = "After a long debate on Tuesday, the new budget was approved by the committee."
    assert st.compare(a, b)["ted"] > 0.1

def test_split_is_aligned():
    a = "The company reduced its prices and customers started buying far more products every week."
    b = "The company reduced its prices. Customers started buying far more products every week."
    r = st.compare(a, b)
    assert r["matched"] == 1.0 and r["ted"] > 0.0

def test_burstiness_ranges():
    assert st.burstiness([10, 10, 10, 10]) == -1.0
    assert st.burstiness([3, 40, 5, 38, 4, 41]) > st.burstiness([20, 22, 19, 21, 20, 22])

def test_targets_vary_and_floor_holds():
    ts = [st.draw_targets(np.random.default_rng(i), "essay") for i in range(20)]
    assert len({t["target"] for t in ts}) > 5
    assert all(t["target"] >= t["floor"] for t in ts)
    assert st.draw_targets(np.random.default_rng(1), "essay", min_ted=0.5)["floor"] == 0.5

def test_recipes_differ_and_disable_families():
    rs = [st.draw_recipe(np.random.default_rng(i)) for i in range(30)]
    assert len({tuple(sorted(r.items())) for r in rs}) == 30
    assert all(sum(r[f] > 0 for f in st.FAMILIES) >= 4 for r in rs)
    assert any(any(r[f] == 0 for f in st.FAMILIES) for r in rs)

def test_intensity_p_uses_recipe():
    from paraphraser import intensity
    intensity.set_recipe({"clause": 0.0, "voice": 1.0, "_boost": 1.0})
    assert intensity.p("clause", 0.9) == 0.0
    intensity.set_recipe(None)
    assert intensity.p("clause", 0.9) == 0.9

def test_gated_pipeline_reports_structure_and_varies():
    import paraphraser.intensity as I
    from paraphraser.pipeline import paraphrase
    txt = ("The city council approved the new transit plan after months of heated debate, and residents began to complain about the cost. "
           "Many shop owners feared that fewer cars would reduce their sales. Engineers argued that the buses would bring more customers downtown. "
           "Because the budget was tight, the mayor delayed the second phase until next spring. Critics said the delay showed poor planning, but supporters disagreed. "
           "In the end, the first routes opened on time and ridership grew steadily during the summer.")
    I.LEVEL = 1
    try:
        outs, recs = set(), set()
        for sd in (11, 12, 13):
            r = paraphrase(txt, "essay", 3, seed=sd, use_hygiene=False, log=False)
            s = r.report["structure"]
            assert {"ted", "order", "burstiness", "target", "floor", "burst_floor", "passed_structure_gate", "recipe"} <= set(s)
            assert 0.0 <= s["ted"] <= 1.0
            outs.add(r.text); recs.add(tuple(sorted(s["recipe"].items())))
        assert len(outs) == 3 and len(recs) >= 2
    finally:
        I.LEVEL = 0
