from paraphraser.doc import Document
from paraphraser.healing import heal_sentence, fix_articles, fix_spacing

D = Document.from_text("x.")
H = lambda s: heal_sentence(s, D)

def test_articles():
    assert H("It was an unique idea and a honest answer about a hour of work.") == "It was a unique idea and an honest answer about an hour of work."
    assert H("She ate an apple and a orange.") == "She ate an apple and an orange."
    assert H("This is a NASA project and an MRI scan.") == "This is a NASA project and an MRI scan."

def test_spacing_caps_dups():
    assert H("the  cat ,sat on the the mat .") == "The cat, sat on the mat."
    assert H("it works. then it stops.") == "It works. Then it stops."

def test_agreement():
    assert H("They is late and it were fine.") == "They are late and it was fine."

def test_lock_aware_article():
    D2 = Document.from_text("x."); ph = D2.lock("owl", "idiom")
    assert heal_sentence(f"He saw a {ph} today.", D2) == f"He saw an {ph} today."

def test_caps_respects_abbreviations():
    assert H("Use tools, e.g. the hammer. then rest.") == "Use tools, e.g. the hammer. Then rest."

def test_ellipsis_and_url_safe():
    assert H("Wait ... what happened here?") == "Wait ... what happened here?" or H("Wait ... what happened here?") == "Wait... what happened here?"
    assert "..." in H("It was so dangerous ... even when hands-free.")
    assert "Com/" not in H("See https://site. com/page now.")
