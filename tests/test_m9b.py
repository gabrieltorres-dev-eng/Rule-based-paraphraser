import numpy as np, tempfile
from pathlib import Path
from paraphraser import hygiene
from paraphraser.hygiene import NgramStore, ngram_hashes, make_seed
from paraphraser.pipeline import paraphrase
from paraphraser.resources import Resources

def test_seed_unique():
    assert len({make_seed("same-doc") for _ in range(50)}) == 50

def test_leakage_ignores_input_spans():
    with tempfile.TemporaryDirectory() as t:
        s = NgramStore(Path(t) / "n.bin")
        a = "one two three four five six seven eight nine ten eleven twelve"
        s.add(a, "docA")
        assert s.leakage(a, a, "docB") == 0                                    # shared with own input: not leakage
        assert s.leakage(a + " plus", "nothing related here at all", "docB") > 0   # appears in docA's output only
        s.close()

def test_end_to_end_runs_and_reports():
    text = ("Cities grow fast every year. Because traffic gets worse, people complain about delays. The council approved the new plan. "
            "Public transit could help a lot, but many riders do not trust the schedules.\n\nThe committee reviewed the budget in March. "
            "Engineers designed the new stations. In conclusion, cities must invest in transit to reduce traffic.")
    r = paraphrase(text, "essay", candidates=3, seed=7, use_hygiene=False, res=Resources(load_heavy=False))
    assert isinstance(r.text, str) and len(r.text.split()) > 30
    assert "rounds" in r.report and "metrics" in r.report and r.report["candidates"] == 3
    # numbers / names must survive untouched
    r2 = paraphrase("The report from Acme Corp. said that 1,250 cars crossed the bridge in 2019. Officials reviewed the data carefully.",
                    "essay", candidates=2, seed=3, use_hygiene=False, res=Resources(load_heavy=False))
    assert "1,250" in r2.text and "2019" in r2.text and "Acme Corp" in r2.text

def test_repeat_passes_do_not_undo_each_other():
    from paraphraser.pipeline import paraphrase_repeated
    from paraphraser.resources import Resources
    from paraphraser import intensity
    intensity.LEVEL = 1
    try:
        txt = "Output was low during autumn and winter. The study found that the results were clear. Many towns rely on diesel."
        r = paraphrase_repeated(txt, "essay", repeats=2, candidates=2, seed=3, res=Resources(load_heavy=False, load_files=False), use_hygiene=False)
        assert r.report["repeats"] == 2 and len(r.report["passes"]) == 2
        assert "during autumn and winter" not in r.text or r.report["passes"][0]["transforms"].get("routing.coord_swap", 0) == 0
    finally:
        intensity.LEVEL = 0
