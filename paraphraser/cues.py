"""Sentence-opening cue lists (shared by calibration and template mining logic)."""
HEDGE_CUES = ("in some cases", "in many cases", "in most cases", "arguably", "perhaps", "maybe", "probably", "it seems",
              "it appears", "apparently", "presumably", "possibly", "in general", "generally", "typically", "usually",
              "often", "sometimes", "for the most part", "to some extent", "to a degree", "more or less", "i think",
              "i suspect", "i guess", "evidence suggests", "research suggests", "it may be", "it might be")
CONCESSION_CUES = ("admittedly", "that said", "having said that", "granted", "of course", "to be fair", "even so", "still",
                   "then again", "all the same", "that being said", "in fairness", "nonetheless", "nevertheless")

def lead_cue(sentence: str, cues) -> bool:
    s = sentence.lower().lstrip("\"'(")
    return any(s == c or s.startswith(c + " ") or s.startswith(c + ",") for c in cues)
