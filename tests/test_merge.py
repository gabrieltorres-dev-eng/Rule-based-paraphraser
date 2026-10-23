from paraphraser import glue
from paraphraser.routing.merge import merge_pair, relcl_pair, front_time_place

def setup_module():
    glue.set_glue(["and", "which", "who", "the", "a"])

def test_merge_shared_subject():
    assert merge_pair("Canada would send a large fighting force to Europe.", "Canada would field many men for the war effort.") == \
        "Canada would send a large fighting force to Europe and would field many men for the war effort."

def test_merge_blocked_for_linked_second_sentence():
    assert merge_pair("Canada sent a large army to Europe.", "However, Canada kept many troops at home.") is None

def test_relative_clause_from_this():
    out = relcl_pair("There was a great deal of strife, such as the Conscription Crisis.", "This would further widen the divide between the two groups.")
    assert out == "There was a great deal of strife, such as the Conscription Crisis, which would further widen the divide between the two groups."

def test_relative_clause_people_use_who():
    assert relcl_pair("The country sent many soldiers.", "The soldiers fought bravely on the front lines.") == \
        "The country sent many soldiers, who fought bravely on the front lines."

def test_front_time_phrase():
    assert front_time_place("The council approved the new plan in August 1914.") == "In August 1914, the council approved the new plan."

def test_copula_not_stranded():
    assert front_time_place("The town was quiet in the summer of 1914.") in (None, "In the summer of 1914, the town was quiet.")

def test_fuse_blocks_repeated_opening():
    import numpy as np
    from paraphraser.routing.length import fuse_option
    rng = np.random.default_rng(0)
    assert fuse_option("QZ1X was a big war.", "QZ2X was a turning point.", [", and "], rng, lambda s: s.replace("QZ1X", "World War I").replace("QZ2X", "World War I")) is None
    assert fuse_option("The war ended in May.", "Prices fell sharply afterwards.", [" and "], rng) == "The war ended in May, and prices fell sharply afterwards."

def test_additive_opener_pulled_back():
    from paraphraser.doc import Document
    from paraphraser.macro.paragraph_ops import attach_linked_openers
    d = Document.from_text("The war began in May. It lasted a year.\n\nAlso, prices rose sharply. Wages fell.", "essay", uid="t")
    attach_linked_openers(d)
    assert len(d.paragraphs) == 1
