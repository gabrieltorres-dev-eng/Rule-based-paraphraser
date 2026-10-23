"""Progress display + checkpoint/resume helpers shared by all build scripts.

Checkpoints live in data/.state/. Every write is atomic (write temp file, then rename),
so pressing Ctrl+C or losing power never leaves a corrupt checkpoint.
"""
import json, os, pickle, sys, time
from pathlib import Path
from _common import DATA

STATE_DIR = DATA / ".state"

def _atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

class State:
    """Small JSON checkpoint: State('download').get('corpora.blog', {})"""
    def __init__(self, name):
        self.path = STATE_DIR / f"{name}.json"
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}
    def get(self, k, default=None): return self.d.get(k, default)
    def set(self, k, v): self.d[k] = v; self.save()
    def done(self, k): return self.d.get(k) == "done"
    def mark(self, k): self.set(k, "done")
    def save(self): _atomic(self.path, json.dumps(self.d, indent=1).encode())
    def clear(self):
        self.d = {}
        if self.path.exists(): self.path.unlink()

def save_pickle(name, obj): _atomic(STATE_DIR / f"{name}.pkl", pickle.dumps(obj, protocol=4))
def load_pickle(name, default=None):
    p = STATE_DIR / f"{name}.pkl"
    return pickle.loads(p.read_bytes()) if p.exists() else default
def drop_pickle(name):
    p = STATE_DIR / f"{name}.pkl"
    if p.exists(): p.unlink()

class Progress:
    """Progress('templates/blog', total=500, start=120).update(n) -> prints % and ETA."""
    def __init__(self, label, total, start=0, unit=""):
        self.label, self.total, self.n, self.unit = label, max(total or 0, 0), start, unit
        self.t0, self.n0, self.last = time.time(), start, 0.0
        self._print(force=True)
    def update(self, k=1): self.n += k; self._print()
    def set(self, n): self.n = n; self._print()
    def _print(self, force=False):
        now = time.time()
        if not force and now - self.last < 1.0: return
        self.last = now
        if self.total:
            pct = min(100.0, 100.0 * self.n / self.total)
            rate = (self.n - self.n0) / max(now - self.t0, 1e-6)
            eta = (self.total - self.n) / rate if rate > 0 else 0
            msg = f"  [{self.label}] {pct:5.1f}%  {self.n:,}/{self.total:,}{self.unit}  ETA {_fmt(eta)}"
        else:
            msg = f"  [{self.label}] {self.n:,}{self.unit}"
        sys.stdout.write("\r" + msg.ljust(90)); sys.stdout.flush()
    def close(self):
        self._print(force=True); sys.stdout.write("\n"); sys.stdout.flush()

def _fmt(s):
    s = int(s)
    return f"{s//3600}h{s%3600//60:02d}m" if s >= 3600 else f"{s//60}m{s%60:02d}s"
