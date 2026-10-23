import numpy as np
from paraphraser.register.deflate import deflate_sentence
from paraphraser.doc import merge_bad_splits

class _R:
    wordnet = None
    vectors = None

def test_intensifier_removed():
    out = deflate_sentence("The war fundamentally changed the borders of many countries.", _R(), np.random.default_rng(0), 1.0)
    assert out == "The war changed the borders of many countries."

def test_negation_kept():
    assert deflate_sentence("The plan did not fundamentally change the borders of countries.", _R(), np.random.default_rng(0), 1.0) is None

def test_unterminated_piece_merged():
    assert merge_bad_splits(["World War", "I was a turning point."]) == ["World War I was a turning point."]
