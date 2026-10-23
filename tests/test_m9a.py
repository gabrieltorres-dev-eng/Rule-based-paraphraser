from paraphraser.profile import RegisterProfile
from paraphraser.scoring import metrics, violations, change_ratio, total_score

P = RegisterProfile.load("default")
TEXT = "The cat sat on the mat. It was a very long day for everyone who lived in the old house on the hill, and nobody slept well. Dogs bark."

def test_metrics_and_violations():
    m = metrics(TEXT, P)
    for k in ("hist_wasserstein", "sent_len_var", "ttr", "stopword_ratio", "punct_entropy"):
        assert k in m
    v = violations(m, P)
    assert all(x >= 0 for x in v.values())
    assert change_ratio("a b c d", "a b c d") == 0 and change_ratio("a b c d", "w x y z") == 1

def test_relax_widens():
    m = {"hist_wasserstein": 2.0, "sent_len_var": 5.0, "ttr": 0.9, "stopword_ratio": 0.4, "punct_entropy": 1.5}
    v0 = violations(m, P); v1 = violations(m, P, relax={"ttr": 3.0, "hist_wasserstein": 3.0})
    assert v1["ttr"] < v0["ttr"] and v1["hist_wasserstein"] < v0["hist_wasserstein"]
