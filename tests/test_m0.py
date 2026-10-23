from paraphraser.doc import Document
from paraphraser.profile import RegisterProfile

TEXT = "First paragraph here. It has two sentences.\n\nSecond paragraph is short."

def test_roundtrip():
    d = Document.from_text(TEXT)
    assert len(d.paragraphs) == 2
    assert d.text() == "First paragraph here. It has two sentences.\n\nSecond paragraph is short."

def test_lock_restore():
    d = Document.from_text(TEXT)
    ph = d.lock('"quoted text"', "quote")
    d.paragraphs[0].sentences[0] = f"He said {ph} loudly."
    assert d.text().startswith('He said "quoted text" loudly.')

def test_profile_default_and_pos():
    p = RegisterProfile.load("nonexistent")
    assert 0 < p.rate("passive") < 1
    assert p.pos_rate("passive", "first") >= 0

def test_no_split_inside_parens():
    from paraphraser.doc import Document
    d = Document.from_text("Laws vary (National Conference of State Legislature [NCSL], 2019). Next one.")
    assert len(d.paragraphs[0].sentences) == 2
