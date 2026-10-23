import json, shutil
from paraphraser.calibration import measure as M

TEXTS = [
    "I don't think so. Because it rained, we stayed home. The ball was thrown by Tom. Nope.\nSecond para here. It is fine; really (mostly). We are done — finally.\nThird one. Short. Another sentence follows it.",
] * 8

def test_measure_keys_and_ranges():
    p = M.measure_texts(TEXTS)
    assert abs(sum(p["sent_len"]["hist"]) - 1) < 1e-6
    for k in ("passive", "adverbial_front", "early_thesis", "contraction", "fragment"):
        assert 0 <= p["rates"][k] <= 1
    assert p["rates"]["passive"] > 0 and p["rates"]["adverbial_front"] > 0 and p["rates"]["fragment"] > 0
    assert p["rates"]["emdash_per_1k"] > 0 and p["rates"]["semicolon_per_1k"] > 0
    assert set(p["position"]["passive"]) == {"first", "middle", "last"}
    assert p["rates"]["contraction"] == 0.0 or p["rates"]["contraction"] > 0

def test_contraction_counts():
    assert M.contraction_counts("I don't know and it is odd. We can't.") == (2, 1)
