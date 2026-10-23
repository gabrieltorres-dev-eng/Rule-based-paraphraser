import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from build_thesaurus import parse_entries

def test_parse_wrapped_and_single_line():
    lines = ["*** START OF THE PROJECT", "", "outline,sketch,draft,", "plan,summary", "plan,scheme,design", "*** END OF"]
    assert list(parse_entries(lines)) == ["outline,sketch,draft,plan,summary", "plan,scheme,design"]

def test_mutual_synonyms():
    from paraphraser.resources import Resources
    r = Resources(load_heavy=False, load_files=False)
    r._thes = {"show": {"display", "reveal"}, "display": {"show"}, "reveal": {"uncover"}}
    assert r.mutual_synonyms("show") == ["display"]
