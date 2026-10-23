"""Phase 9: fingerprint & operational hygiene. Salted per-document seeds, cross-document n-gram audit,
idiom-usage rotation across the user base, and structured logging of every transform."""
from __future__ import annotations
import hashlib, json, os, re, time, uuid
from pathlib import Path
from ..profile import ROOT

STATE = ROOT / "data" / ".state"

def _salt() -> str:
    p = STATE / "session_salt.txt"
    STATE.mkdir(parents=True, exist_ok=True)
    if not p.exists(): p.write_text(uuid.uuid4().hex)
    return p.read_text().strip()

def make_seed(doc_uid: str) -> int:
    """Never reused: document UUID + persistent salt + clock + OS entropy."""
    h = hashlib.sha256(f"{doc_uid}|{_salt()}|{time.time_ns()}|".encode() + os.urandom(16)).digest()
    return int.from_bytes(h[:8], "big")

def new_uid() -> str: return uuid.uuid4().hex

def words(text: str): return re.findall(r"[a-z0-9']+", text.lower())

def ngram_hashes(text: str, n: int = 8) -> set:
    w = words(text)
    return {int.from_bytes(hashlib.sha1(" ".join(w[i:i + n]).encode()).digest()[:7], "big") for i in range(len(w) - n + 1)}

class NgramStore:
    """Rolling store of output 8-grams, kept in an append-only binary file (no database / file locking).
    `leakage` counts spans in a candidate that are NOT in its own input but already appeared in another
    document's output (template-bank / dictionary leakage). Record = 8-byte span hash + 8-byte doc hash."""
    def __init__(self, path: Path | None = None, n: int = 8):
        self.n = n
        self.path = Path(path or STATE / "ngrams.bin")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index: dict[int, set] = {}
        if self.path.exists():
            raw = self.path.read_bytes()
            for i in range(0, len(raw) - 15, 16):
                self.index.setdefault(int.from_bytes(raw[i:i + 8], "big"), set()).add(int.from_bytes(raw[i + 8:i + 16], "big"))

    @staticmethod
    def _doc_hash(uid: str) -> int: return int.from_bytes(hashlib.sha1(uid.encode()).digest()[:8], "big")

    def leakage(self, out_text: str, src_text: str, doc_uid: str = "") -> int:
        new = ngram_hashes(out_text, self.n) - ngram_hashes(src_text, self.n)
        me = self._doc_hash(doc_uid)
        return sum(1 for h in new if any(d != me for d in self.index.get(h, ())))

    def add(self, out_text: str, doc_uid: str):
        me = self._doc_hash(doc_uid)
        with open(self.path, "ab") as f:
            for h in ngram_hashes(out_text, self.n):
                self.index.setdefault(h, set()).add(me)
                f.write(h.to_bytes(8, "big") + me.to_bytes(8, "big"))

    def close(self): pass

def load_usage():
    from ..ingest import idioms
    p = STATE / "idiom_usage.json"
    if p.exists():
        idioms.USAGE.clear(); idioms.USAGE.update(json.loads(p.read_text()))

def save_usage():
    from ..ingest import idioms
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "idiom_usage.json").write_text(json.dumps(dict(idioms.USAGE.most_common(5000))))

def log_document(uid: str, register: str, seed: int, report: dict, log: list):
    d = STATE / "logs"; d.mkdir(parents=True, exist_ok=True)
    with open(d / "documents.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"uid": uid, "register": register, "seed": seed, "ts": time.time(),
                            "report": report, "transforms": log}, ensure_ascii=False) + "\n")
