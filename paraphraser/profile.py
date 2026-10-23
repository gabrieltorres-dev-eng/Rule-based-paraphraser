"""RegisterProfile: measured human statistics for one register (output of calibration).
Loads data/profiles/<register>.json; falls back to default.json (plain-English priors)."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = ROOT / "data" / "profiles"
REGISTERS = ("blog", "essay", "forum", "technical", "academic")

DEFAULT = {
    "register": "default", "placeholder": True,
    "sent_len": {"mean": 20.0, "var": 110.0, "bin_width": 5,
                 "hist": [0.04, 0.16, 0.21, 0.19, 0.14, 0.10, 0.06, 0.04, 0.03, 0.03]},
    "para": {"sents_mean": 4.5, "sents_var": 6.0, "words_mean": 95.0, "words_var": 3000.0},
    "rates": {"passive": 0.12, "adverbial_front": 0.12, "early_thesis": 0.30, "contraction": 0.25,
              "fragment": 0.02, "hedge_front": 0.03, "concession_front": 0.01, "para_swap": 0.10, "emdash_per_1k": 1.0, "paren_per_1k": 2.0, "semicolon_per_1k": 0.5},
    "position": {"passive": {"first": 0.10, "middle": 0.12, "last": 0.13},
                 "adverbial_front": {"first": 0.08, "middle": 0.13, "last": 0.12},
                 "fragment": {"first": 0.0, "middle": 0.02, "last": 0.0}},
    "bands": {"ttr": [0.35, 0.65], "stopword_ratio": [0.35, 0.55], "punct_entropy": [1.2, 2.4],
              "sent_len_var": [40.0, 220.0], "perplexity": None},
    "meta": {"words": 0, "docs": 0},
}

class RegisterProfile:
    def __init__(self, data: dict):
        self.d = data

    @classmethod
    def load(cls, register: str) -> "RegisterProfile":
        for name in (register, "default"):
            p = PROFILE_DIR / f"{name}.json"
            if p.exists():
                return cls(json.loads(p.read_text(encoding="utf-8")))
        return cls(json.loads(json.dumps(DEFAULT)))

    def save(self, register: str):
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        (PROFILE_DIR / f"{register}.json").write_text(json.dumps(self.d, indent=1), encoding="utf-8")

    def __getitem__(self, k): return self.d[k]
    def rate(self, name: str) -> float: return float(self.d["rates"][name])

    def pos_rate(self, name: str, bucket: str) -> float:
        """Rate conditioned on position in paragraph (falls back to the overall rate)."""
        t = self.d.get("position", {}).get(name, {})
        v = t.get(bucket)
        return float(v) if v is not None else self.rate(name)

    def band(self, name: str):
        return self.d["bands"].get(name)
