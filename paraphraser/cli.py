"""Command line: paraphrase text with the rule-based pipeline, calibrate profiles, score a file."""
import argparse, json, sys
from pathlib import Path

REGS = ["blog", "essay", "forum", "technical", "academic"]

def _strength_args(p):
    p.add_argument("--register", default="essay", choices=REGS)
    p.add_argument("--candidates", type=int, default=4, help="candidates per pass (more = better pick, slower)")
    p.add_argument("-n", "--repeat", "--repetitions", dest="repeat", type=int, default=1,
                   help="paraphrase the output again N times in total (default 1)")
    p.add_argument("--seed", type=int)
    p.add_argument("--swap-every", type=int, default=None, help="one idiom swap per N words (default 8; lower = more swaps)")
    p.add_argument("--lexical", choices=["strict", "loose"], default="loose", help="synonym filter strength (loose = many more swaps)")
    p.add_argument("--lexical-rate", type=float, default=None, help="fraction of eligible content words to try swapping")
    p.add_argument("--structure", choices=["human", "aggressive"], default="aggressive",
                   help="aggressive = rearrange nearly everything possible; human = only match human base rates")
    p.add_argument("--min-ted", type=float, default=None, help="structural gate: minimum normalised dependency-tree edit distance (0..1); default is drawn per document from the human-paraphrase distribution")
    p.add_argument("--min-burst", type=float, default=None, help="structural gate: minimum burstiness of sentence lengths (-1..1; applies to texts of 6+ sentences)")
    p.add_argument("--no-hygiene", action="store_true", help="skip the n-gram store / usage rotation / logging")
    p.add_argument("--quiet", action="store_true", help="no loading report or progress bar")

def _apply_strength(a):
    import paraphraser.intensity as _int; _int.LEVEL = 1 if a.structure == "aggressive" else 0
    import paraphraser.lexical as _lex; _lex.LOOSE = (a.lexical == "loose")
    import paraphraser.pipeline as _pl
    if a.lexical_rate is not None: _pl.LEX_RATE = a.lexical_rate
    _pl.MIN_TED, _pl.MIN_BURST = a.min_ted, a.min_burst
    if a.swap_every:
        from .ingest import idioms as _idi; _idi.SWAP_EVERY = a.swap_every

def main(argv=None):
    ap = argparse.ArgumentParser(prog="paraphraser", description="Rule-based, human-sourced paraphraser (no generative model).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="paraphrase a text file")
    r.add_argument("input"); r.add_argument("-o", "--output"); r.add_argument("--report")
    r.add_argument("--save-passes", action="store_true", help="with --repeat, also save every intermediate pass as <output>.passN.txt")
    _strength_args(r)
    b = sub.add_parser("batch", help="paraphrase every .txt in a folder (models load once, so it is much faster than separate runs)")
    b.add_argument("indir"); b.add_argument("outdir")
    _strength_args(b)
    c = sub.add_parser("calibrate", help="re-measure human style profiles from data/corpus")
    c.add_argument("--register"); c.add_argument("--max-docs", type=int, default=0)
    c.add_argument("--perplexity", action="store_true", help="also fit the KenLM perplexity band (needs kenlm)")
    s = sub.add_parser("score", help="report scoring-gate metrics for a text file")
    s.add_argument("input"); s.add_argument("--register", default="essay")
    a = ap.parse_args(argv)

    if a.cmd in ("run", "batch"):
        import warnings; warnings.filterwarnings("ignore")
        from .pipeline import paraphrase_repeated, get_resources
        from .progress import Tracker, load_report
        _apply_strength(a)
        res = get_resources()
        load_report(res, a.quiet)

    if a.cmd == "run":
        ip = Path(a.input)
        if not ip.exists():
            raise SystemExit(f"Input file not found: {ip.resolve()}\nCreate it (paste your text, save as UTF-8) or pass the path to an existing .txt file.")
        text = ip.read_text(encoding="utf-8-sig")
        tr = Tracker(passes=a.repeat, enabled=not a.quiet)
        def save_pass(k, t):
            if a.save_passes and a.output and k < a.repeat:
                o = Path(a.output); (o.parent / f"{o.stem}.pass{k}{o.suffix}").write_text(t, encoding="utf-8")
        res_ = paraphrase_repeated(text, a.register, a.repeat, a.candidates, a.seed, res=res, use_hygiene=not a.no_hygiene,
                                   progress=tr, on_pass=save_pass)
        tr.finish(f"{a.repeat} pass{'es' if a.repeat > 1 else ''} complete, {res_.report['change_ratio']:.0%} of the text changed")
        if a.output: Path(a.output).write_text(res_.text, encoding="utf-8")
        else: print(res_.text)
        rep = json.dumps(res_.report, indent=1)
        if a.report: Path(a.report).write_text(rep, encoding="utf-8")
        elif not a.quiet: print("\n--- report ---\n" + rep, file=sys.stderr)
    elif a.cmd == "batch":
        src, dst = Path(a.indir), Path(a.outdir); dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.txt"))
        if not files: raise SystemExit(f"No .txt files found in {src.resolve()}")
        tr = Tracker(passes=a.repeat * len(files), enabled=not a.quiet)
        import time
        for k, f in enumerate(files, 1):
            t0 = time.time()
            class _Sub:                                   # map this file's passes onto the overall bar
                def __init__(self, base): self.base = base
                def begin_pass(self, j, units): tr.begin_pass(self.base + j, units)
                def extend(self, n): tr.extend(n)
                def tick(self, label): tr.tick(f"{f.name} | {label}")
            r_ = paraphrase_repeated(f.read_text(encoding="utf-8-sig"), a.register, a.repeat, a.candidates, a.seed, res=res,
                                     use_hygiene=not a.no_hygiene, progress=_Sub((k - 1) * a.repeat))
            (dst / f.name).write_text(r_.text, encoding="utf-8")
            (dst / (f.stem + ".report.json")).write_text(json.dumps(r_.report, indent=1), encoding="utf-8")
        tr.finish(f"{len(files)} files written to {dst}")
    elif a.cmd == "calibrate":
        from .calibration.measure import calibrate
        from .profile import REGISTERS, RegisterProfile
        for reg in ([a.register] if a.register else REGISTERS):
            p = calibrate(reg, a.max_docs)
            if p and a.perplexity:
                from .scoring import calibrate_perplexity
                from .pipeline import get_resources
                print(f"  {reg}: perplexity band", calibrate_perplexity(reg, get_resources(), RegisterProfile(p)))
    else:
        from .profile import RegisterProfile
        from .scoring import metrics, violations
        from .pipeline import get_resources
        prof = RegisterProfile.load(a.register)
        m = metrics(Path(a.input).read_text(encoding="utf-8"), prof, get_resources())
        print(json.dumps({"metrics": m, "violations": violations(m, prof)}, indent=1))

if __name__ == "__main__":
    main()
