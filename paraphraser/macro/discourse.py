"""Heuristic discourse tagger (stand-in for an RST parser, same interface): paragraph roles,
'linked' detection (causal/sequential/anaphoric openers), and uniform-arc flagging."""
import re
from ..doc import PH_RE

CONCL_CUES = ("in conclusion", "to conclude", "to sum up", "in summary", "to summarize", "overall", "all in all",
              "in short", "ultimately", "finally")
SOLUTION_CUES = re.compile(r"\b(solution|solve|recommend|should|propose|can be addressed|must|ought|need to)\b", re.I)
CAUSAL = ("because", "therefore", "thus", "consequently", "hence", "as a result", "accordingly", "for this reason",
          "due to", "so", "that is why", "this is why", "which is why")
SEQUENCE = ("first", "firstly", "second", "secondly", "third", "thirdly", "next", "then", "finally", "lastly",
            "afterward", "afterwards", "after that", "meanwhile", "subsequently", "initially", "to begin", "to start",
            "another", "also", "furthermore", "moreover", "in addition", "additionally", "on the other hand",
            "however", "but", "yet", "although", "while", "similarly", "likewise", "in contrast", "for example", "for instance")
ANAPHORIC = {"this", "these", "that", "those", "it", "they", "he", "she", "such", "both", "its", "their", "his", "her",
             "the former", "the latter", "here", "there"}

def first_words(s: str, n=3) -> str:
    return " ".join(re.findall(r"[A-Za-z']+", s)[:n]).lower()

def starts_with(s: str, cues) -> bool:
    fw = first_words(s, 4)
    return any(fw == c or fw.startswith(c + " ") for c in cues)

def is_linked(first_sentence: str) -> bool:
    """True if a paragraph opening depends on what precedes it (causal / sequential / anaphoric)."""
    if starts_with(first_sentence, CAUSAL) or starts_with(first_sentence, SEQUENCE): return True
    w = first_words(first_sentence, 1)
    return w in ANAPHORIC

def is_title(p) -> bool:
    return len(p.sentences) == 1 and not re.search(r"[.!?:\"')]$", p.sentences[0].strip()) and len(p.sentences[0].split()) <= 14

def tag_roles(doc):
    idx = [i for i, p in enumerate(doc.paragraphs) if not is_title(p)]
    for i, p in enumerate(doc.paragraphs):
        p.role = "title" if is_title(p) else "body"
    if not idx: return doc
    doc.paragraphs[idx[0]].role = "intro"
    if len(idx) > 1:
        last = doc.paragraphs[idx[-1]]
        last.role = "conclusion"
    body = [i for i in idx if doc.paragraphs[i].role == "body"]
    for n, i in enumerate(body):
        txt = " ".join(doc.paragraphs[i].sentences)
        if n == 0: doc.paragraphs[i].role = "background"
        elif SOLUTION_CUES.search(txt) and n == len(body) - 1: doc.paragraphs[i].role = "solution"
        else: doc.paragraphs[i].role = "elaboration"
    return doc

def uniform_arc(doc) -> bool:
    """Flag the AI-typical Background -> Elaboration -> Solution arc."""
    roles = [p.role for p in doc.paragraphs if p.role in ("background", "elaboration", "solution")]
    return len(roles) >= 3 and roles[0] == "background" and roles[-1] == "solution" and all(r == "elaboration" for r in roles[1:-1])
