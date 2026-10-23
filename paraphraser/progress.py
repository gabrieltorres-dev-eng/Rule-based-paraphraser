"""Console progress: model-loading report plus one overall bar (percent, pass k/n, stage, elapsed, ETA)."""
import sys, time

def _fmt(sec: float) -> str:
    sec = int(max(sec, 0))
    return f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s"

class Tracker:
    """Overall progress over `passes` repetitions. Each pass has a (growing) number of units; the bar redraws on every tick."""
    def __init__(self, passes: int = 1, enabled: bool = True, stream=None):
        self.passes, self.enabled, self.out = passes, enabled, stream or sys.stderr
        self.tty = hasattr(self.out, "isatty") and self.out.isatty()
        self.k = 0; self.units = 1; self.done = 0; self.t0 = time.time(); self._last_pct = -1; self._w = 0; self._maxf = 0.0

    def begin_pass(self, k: int, units: int):
        self.k, self.units, self.done = k, max(units, 1), 0

    def extend(self, n: int):
        self.units += n

    def tick(self, label: str, n: int = 1):
        self.done = min(self.done + n, self.units)
        self.show(label)

    def fraction(self) -> float:
        return ((self.k - 1) + self.done / self.units) / self.passes

    def show(self, label: str):
        if not self.enabled: return
        f = max(self.fraction(), self._maxf); self._maxf = f; el = time.time() - self.t0
        eta = el / f - el if f > 0.03 else None
        pct = int(f * 100)
        bar = "#" * int(f * 24) + "-" * (24 - int(f * 24))
        pas = f"pass {self.k}/{self.passes} | " if self.passes > 1 else ""
        line = f"[{bar}] {pct:3d}% | {pas}{label} | {_fmt(el)} elapsed" + (f" | ~{_fmt(eta)} left" if eta is not None else "")
        if self.tty:
            self.out.write("\r" + line.ljust(self._w)); self._w = max(self._w, len(line)); self.out.flush()
        elif pct // 10 != self._last_pct // 10:          # redirected output: one line per 10%
            self._last_pct = pct; self.out.write(line + "\n"); self.out.flush()

    def finish(self, msg: str = "done"):
        if not self.enabled: return
        if self.tty: self.out.write("\r" + " " * self._w + "\r")
        self.out.write(f"[{'#' * 24}] 100% | {msg} | {_fmt(time.time() - self.t0)} total\n"); self.out.flush()

def load_report(res, quiet: bool = False, out=None):
    """Load every model up front, one line per resource with its load time, so a slow start is explained."""
    if quiet: return
    out = out or sys.stderr
    def step(name, fn):
        out.write(f"  loading {name:<26}"); out.flush()
        t = time.time()
        try: info = fn()
        except Exception as e: info = f"FAILED ({type(e).__name__})"
        out.write(f"{time.time() - t:5.1f}s  {info}\n"); out.flush()
    out.write("Loading resources (one-off cost per launch)\n")
    from .doc import get_nlp
    step("spaCy language model", lambda: (get_nlp(), "ok")[1])
    step("WordNet", lambda: (res.wordnet.ensure_loaded(), "ok")[1])
    step("GloVe word vectors", lambda: "ok" if res.vectors is not None else "missing (synonym filter off)")
    step("KenLM perplexity model", lambda: "ok" if res.kenlm else "not installed (perplexity gate off)")
    step("idiom dictionary", lambda: f"{res.idioms.n_entries:,} entries")
    step("template bank", lambda: f"{len(res.templates.rows) if hasattr(res.templates, 'rows') else 'ok'} templates")
    step("Moby thesaurus", lambda: f"{len(res.thesaurus):,} roots" if res.thesaurus else "not built (WordNet only)")
    out.write("\n"); out.flush()
