import numpy as np
from paraphraser.routing.clause import transpose
from paraphraser.routing.passive import passivize
from paraphraser.routing.length import split_options, fuse_option, hist, dist

def T(s): return transpose(s)

def test_transpose_to_front():
    assert T("The team stayed home because the storm was severe.") == ("Because the storm was severe, the team stayed home.", "to_front")

def test_transpose_to_back_concessive():
    r = T("Although the road was icy, the drivers kept going.")
    assert r == ("The drivers kept going, although the road was icy.", "to_back")

def test_transpose_to_back_no_comma():
    r = T("When the bell rang, the students left the room.")
    assert r == ("The students left the room when the bell rang.", "to_back")

def test_transpose_suppressed_pronoun():
    assert T("The committee cancelled the meeting because it lacked a quorum.") is None   # 'it' would precede antecedent

def test_transpose_no_clause():
    assert T("The cat sat on the mat.") is None

def test_passive_basic_past():
    assert passivize("The committee approved the proposal.") == "The proposal was approved by the committee."

def test_passive_plural_present():
    assert passivize("Engineers design the bridges.") == "The bridges are designed by engineers." or \
           passivize("Engineers design the bridges.") == "The bridges are designed by engineers."

def test_passive_pronoun_case():
    assert passivize("He wrote the report.") == "The report was written by him."

def test_passive_modal_and_perfect():
    assert passivize("The council will ban the plan.") == "The plan will be banned by the council."
    assert passivize("The company has launched the product.") == "The product has been launched by the company."

def test_passive_suppressed():
    assert passivize("The house has three rooms.") is None            # stative
    assert passivize("I wrote the report.") is None                   # 1st person subject
    assert passivize("The report was written by him.") is None        # already passive
    assert passivize("The dog ran.") is None                          # intransitive

def test_split_and_fuse():
    s = "The engineers tested the prototype in the lab, and the results surprised the whole team."
    a, b = next(split_options(s))
    assert a == "The engineers tested the prototype in the lab." and b == "The results surprised the whole team."
    rng = np.random.default_rng(0)
    f = fuse_option("The sky was dark.", "The wind was cold.", [", and "], rng)
    assert f == "The sky was dark, and the wind was cold."
    assert fuse_option("The sky was dark.", "However, the wind was cold.", [", and "], rng) is None

def test_extra_reorderers():
    from paraphraser.routing.extra import move_pp, swap_coords
    r = move_pp("Many drivers use phones every day in busy cities across the country.")
    assert r and r[0] == "In busy cities across the country, many drivers use phones every day."
    r2 = move_pp("During the storm, the crew repaired the damaged roof.")
    assert r2 and r2[0] == "The crew repaired the damaged roof during the storm."
    s = swap_coords("The council approved the budget on Monday, and the mayor signed the order on Friday.")
    assert s == "The mayor signed the order on Friday, and the council approved the budget on Monday."

def test_inner_rewrites():
    from paraphraser.routing.inner import poss_of, part_rel, that_toggle, coord_swap
    assert poss_of("It would raise the community's diesel use by 13%.") == "It would raise the diesel use of the community by 13%."
    assert part_rel("Hybrid microgrids serving remote towns rely on diesel.") == "Hybrid microgrids that serve remote towns rely on diesel."
    assert that_toggle("The mayor said that the budget was fine.") == "The mayor said the budget was fine."
    assert coord_swap("Output was low during autumn and winter.") == "Output was low during winter and autumn."

def test_split_with_bank_connectives():
    import numpy as np
    from paraphraser.routing.split_plus import split_sentence
    from paraphraser.resources import Resources
    r = Resources(load_heavy=False, load_files=False)
    for i, (t, f) in enumerate([("Also", "transition"), ("However", "counterexample"), ("As a result", "transition"), ("That said", "concession")]):
        r.templates.add({"id": f"t{i}", "text": t, "function": f, "register": "essay", "docs": 9})
    rng = np.random.default_rng(1)
    a = split_sentence("The council approved the new budget on Monday after a long debate, and the mayor signed the order on Friday morning.", r, rng, "essay", set())
    assert a and a[2] == "split_add" and a[1].startswith("Also, the mayor signed") and a[0].endswith("debate.")
    b = split_sentence("The panels were installed in the village last year, which reduces the need for diesel during the long summer months.", r, rng, "essay", set())
    assert b and b[1].startswith("This reduces the need for diesel")
    c = split_sentence("Output falls sharply during the dark winter months in the north, leading to a heavy reliance on diesel generators there.", r, rng, "essay", set())
    assert c and c[1].startswith("This leads to a heavy reliance")
    d = split_sentence("The town relies mostly on diesel generators for its power, although the new solar panels supply a small share of the demand.", r, rng, "essay", set())
    assert d and d[1].startswith("That said, the new solar panels supply")

def test_no_subordinate_fragment_after_split():
    from paraphraser.routing.length import split_options
    s = "In the United States twenty states have banned handheld phones, while another forty eight states have banned texting for all drivers."
    assert all(not b.startswith("While") for a, b in split_options(s))

def test_connective_phrase_not_moved():
    from paraphraser.routing.extra import move_pp
    assert move_pp("On the other hand, many reported cases indicate talking on the phone while driving is risky.") is None

def test_complex_rewrites():
    from paraphraser.routing.complex import because_split, verb_to_noun, appositive, merge_this
    assert because_split("Drivers slow down on wet roads because the surfaces are slippery and grip is reduced.").endswith("That is why drivers slow down on wet roads.")
    assert verb_to_noun("Researchers analyze the data every day.") == "Researchers carry out the analysis of the data every day."
    assert verb_to_noun("The panels reduce diesel use in winter by a large share.") is None
    assert appositive("Crundall, a researcher at the university, found that drivers miss signals.") == "A researcher at the university, Crundall found that drivers miss signals."
    assert merge_this("The grid depends on diesel in winter months for most households.", "This means that solar power is lost in the dark season.") == "The grid depends on diesel in winter months for most households, which means that solar power is lost in the dark season."
    assert because_split("The town grew quickly because the mill opened.") is None   # too short after the split

def test_front_to_back_and_glue_gate():
    from paraphraser import glue
    from paraphraser.routing.complex import front_to_back, because_split
    glue.set_glue({"that is why", "which", "carry out", "the", "of", "this", "by", "be", "is", "that"})
    s = "With verse, prose, etc., the whole point is the virtuoso placing of words in sentences."
    assert front_to_back(s) == "The whole point is the virtuoso placing of words in sentences, with verse, prose, etc."
    glue.set_glue(set())                                                      # fail closed
    assert because_split("Drivers slow down on wet roads because the surfaces are slippery and grip is reduced.") is None
    glue.set_glue(["which", "which means", "this means", "that is why", "this is why", "that is", "this is", "this", "that", "the", "a", "an", "of", "carry out", "is", "are", "was", "were", "be", "been", "by"])
