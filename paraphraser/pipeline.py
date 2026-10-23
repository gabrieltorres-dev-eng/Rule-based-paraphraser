"""Orchestrates Phases 2-8 with N-candidate acceptance sampling (Phase 8) and hygiene (Phase 9)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .doc import Document
from .profile import RegisterProfile
from .resources import Resources
from .ingest import ingest
from .macro import macro
from .routing import route
from .lexical import perturb
from .register import tells
from .healing import heal
from . import scoring, hygiene, intensity

_RES = None
LEX_RATE = None          # CLI override for the lexical swap rate
MIN_TED = None           # CLI override: structural (tree-edit) distance floor, 0..1
MIN_BURST = None         # CLI override: burstiness floor, -1..1
def get_resources() -> Resources:
    global _RES
    if _RES is None: _RES = Resources()
    return _RES

@dataclass
class Result:
    text: str
    report: dict = field(default_factory=dict)

def run_once(base: Document, profile, res, seed: int, lexical_rate=None, lock=True) -> Document:
    rng = np.random.default_rng(seed)
    d = base.clone()
    ingest(d, res, profile, rng, lock=lock)
    macro(d, profile, res, rng)
    route(d, profile, res, rng)
    from .register import deflate as _deflate
    _deflate.apply(d, res, rng)
    from .macro.paragraph_ops import attach_linked_openers
    attach_linked_openers(d)
    perturb(d, profile, res, rng, rate=lexical_rate if lexical_rate is not None else (LEX_RATE if LEX_RATE is not None else (0.6 if intensity.aggressive() else profile["rates"].get("lexical", 0.08))))
    tells(d, profile, res, rng)
    heal(d)
    return d

def paraphrase(text: str, register: str = "essay", candidates: int = 4, seed: int | None = None,
               res: Resources | None = None, use_hygiene: bool = True, log: bool = True, progress=None, pass_no: int = 1,
               anchor: str | None = None) -> Result:
    anchor = anchor or text            # the ORIGINAL input: later passes are judged against it so they cannot walk back earlier changes
    res = res or get_resources()
    from . import glue as _glue
    _glue.set_glue(res.glue)
    if progress: progress.begin_pass(pass_no, candidates + 2)
    tick = (lambda label: progress.tick(label)) if progress else (lambda label: None)
    profile = RegisterProfile.load(register)
    uid = hygiene.new_uid()
    base = Document.from_text(text, register, uid=uid)
    from .routing.util import warm
    warm([x for _, _, x in base.all_sentences()])
    from .ingest.lock import lock_protected
    lock_protected(base)                       # done once; every candidate clones the locked base
    warm([x for _, _, x in base.all_sentences()])
    from . import intensity
    intensity.SEEN.update(base.restore(x) for _, _, x in base.all_sentences())     # forms later passes must not revert to
    tick("analysed text")
    if use_hygiene: hygiene.load_usage()
    store = hygiene.NgramStore() if use_hygiene else None
    master = seed if seed is not None else hygiene.make_seed(uid)
    seeds = [int(x) for x in np.random.default_rng(master).integers(0, 2**63 - 1, size=candidates)]
    relax: dict = {}
    best = None
    gated = intensity.aggressive()
    tg = None
    if gated:
        from . import structure as _st
        tg = _st.draw_targets(np.random.default_rng(master ^ 0x5DEECE66D), register, MIN_TED, MIN_BURST)
        src_info = _st.infos(anchor)
    def evaluate(sd, boost=1.0):
        rec = None
        if gated:
            rec = _st.draw_recipe(np.random.default_rng(sd ^ 0xA5A5), boost)
            intensity.set_recipe(rec)
        try:
            d = run_once(base, profile, res, sd, lock=False)
        finally:
            intensity.set_recipe(None)
        c = finish(d, sd)
        c["recipe"] = rec
        return c
    def finish(d, sd):
        out = d.text()
        m = scoring.metrics(out, profile, res)
        leak = store.leakage(out, text, uid) if store else 0
        _re = __import__("re"); bad = len(_re.findall(r"qz\d+x", out, _re.I)) > len(_re.findall(r"qz\d+x", text, _re.I))     # an unrestored placeholder must never reach the user
        c = {"doc": d, "text": out, "m": m, "leak": leak, "bad": bad, "change": scoring.change_ratio(text, out), "seed": sd,
             "anchor_change": scoring.change_ratio(anchor, out)}
        if gated:
            o_info = _st.infos(out)
            cmp_ = _st.compare(src_info, o_info)
            b = _st.burstiness([i.n for i in o_info])
            c["st"] = {**cmp_, "burstiness": round(b, 4), "sentences": len(o_info)}
        return c
    def grade(c):
        c["v"] = scoring.violations(c["m"], profile, relax)
        c["score"] = scoring.total_score(c["m"], c["v"], c["change"]) + 0.1 * c["leak"] + (1000 if c["bad"] else 0)
    def st_ok(c):
        if c["bad"]: return False
        s_ = c["st"]
        if s_["ted"] < tg["floor"]: return False
        if s_["sentences"] >= 6 and s_["burstiness"] < tg["burst_floor"]: return False
        return True
    def utility(c):                 # structure first; statistical fit and leakage are only tie-breakers
        s_ = c["st"]
        return (4.0 * min(s_["ted"] / max(tg["target"], 1e-6), 1.0) + 0.6 * s_["order"] + 2.0 * min(c["anchor_change"], 0.6)
                - 0.25 * min(c["score"], 8) - 0.1 * c["leak"])
    scored = []
    for i, sd in enumerate(seeds, 1):
        scored.append(evaluate(sd)); tick(f"candidate {i}/{candidates}")
    rnd = 0
    if gated:
        for step in range(3):
            for c in scored: grade(c)
            if not any(st_ok(c) for c in scored):                       # maybe only burstiness is missing: repair the best-TED candidates
                for c in sorted((c for c in scored if not c["bad"] and c["st"]["ted"] >= tg["floor"]), key=lambda c: -c["st"]["ted"])[:2]:
                    if _st.repair_burstiness(c["doc"], res, np.random.default_rng(c["seed"] ^ 0x77), tg["burst_floor"]):
                        heal(c["doc"]); c.update(finish(c["doc"], c["seed"])); grade(c)
            ok = [c for c in scored if st_ok(c)]
            if ok:
                u = np.array([utility(c) for c in ok])
                pr = np.exp((u - u.max()) / 0.6); pr /= pr.sum()          # sample among legitimate structures instead of always taking the argmax
                best = ok[int(np.random.default_rng(master ^ 0xC0FFEE).choice(len(ok), p=pr))]
                break
            rnd = step + 1
            if step == 2: break
            boost = 1.4 if step == 0 else 1.8                           # escalate the transforms; the structural gate itself never relaxes
            seeds = [int(x) for x in np.random.default_rng([master, 1 + step]).integers(0, 2**63 - 1, size=candidates)]
            if progress: progress.extend(candidates)
            for i, sd in enumerate(seeds, 1):
                scored.append(evaluate(sd, boost)); tick(f"extra candidate {i}/{candidates} (structure below target)")
        if best is None:
            pool = [c for c in scored if not c["bad"]] or scored
            best = max(pool, key=lambda c: (round(c["st"]["ted"], 2), c["anchor_change"], -c["score"]))
    else:
        for step in range(3):                       # grade -> relax the strictest gate (re-grading the SAME candidates is free) -> one resample
            for c in scored: grade(c)
            passing = [c for c in scored if all(x == 0 for x in c["v"].values()) and not c["bad"]]
            best = min(passing or scored, key=lambda c: c["score"])
            if passing: break
            rnd = step
            if step == 0 or step == 1:
                worst = max(best["v"], key=best["v"].get)
                relax[worst] = relax.get(worst, 1.0) * 1.5
                if step == 1:                       # still failing after two relaxations: draw a fresh batch once
                    seeds = [int(x) for x in np.random.default_rng([master, 1]).integers(0, 2**63 - 1, size=candidates)]
                    if progress: progress.extend(candidates)
                    for i, sd in enumerate(seeds, 1):
                        scored.append(evaluate(sd)); tick(f"extra candidate {i}/{candidates} (gates not met)")
    d = best["doc"]
    report = {"uid": uid, "register": register, "candidates": candidates, "rounds": rnd + 1, "relaxed": relax,
              "passed_all_gates": all(x == 0 for x in best["v"].values()), "metrics": best["m"],
              "violations": best["v"], "change_ratio": round(best["change"], 3), "change_vs_original": round(best["anchor_change"], 3), "leakage": best["leak"],
              "transforms": dict(__import__("collections").Counter(f"{x['phase']}.{x['transform']}" for x in d.log if "transform" in x)),
              "profile_placeholder": bool(profile["placeholder"] if "placeholder" in profile.d else False)}
    if gated:
        recipe = best.get("recipe") or {}
        report["structure"] = {**best["st"], "target": tg["target"], "floor": tg["floor"], "burst_floor": tg["burst_floor"],
                               "quantile": tg["quantile"], "targets_from": tg["profile"], "passed_structure_gate": st_ok(best),
                               "escalations": rnd, "recipe": {k: round(v, 2) for k, v in recipe.items() if not k.startswith("_")},
                               "recipe_boost": recipe.get("_boost", 1.0)}
    if store: store.add(best["text"], uid); store.close()
    if use_hygiene: hygiene.save_usage()
    if log and use_hygiene: hygiene.log_document(uid, register, best["seed"], report, d.log)
    tick("finished")
    return Result(best["text"], report)

def paraphrase_repeated(text: str, register: str = "essay", repeats: int = 1, candidates: int = 4, seed: int | None = None,
                        res: Resources | None = None, use_hygiene: bool = True, progress=None, on_pass=None) -> Result:
    """Paraphrase, then paraphrase the OUTPUT again, `repeats` times in total. Each pass gets its own seed."""
    from . import intensity
    intensity.SEEN.clear()
    cur, passes = text, []
    for k in range(1, max(repeats, 1) + 1):
        orig_words = {w for w in __import__("re").findall(r"[a-z]{4,}", text.lower())}
        now_words = set(__import__("re").findall(r"[a-z]{4,}", cur.lower()))
        intensity.RETIRED.clear(); intensity.RETIRED.update(orig_words - now_words)        # words earlier passes removed
        r = paraphrase(cur, register, candidates, None if seed is None else seed + k - 1, res=res,
                       use_hygiene=use_hygiene, progress=progress, pass_no=k, anchor=text)
        passes.append({"pass": k, "change_vs_previous": r.report["change_ratio"], "change_vs_original": r.report.get("change_vs_original"), "passed_all_gates": r.report["passed_all_gates"],
                       "transforms": r.report["transforms"], "relaxed": r.report["relaxed"], "structure": r.report.get("structure")})
        if on_pass: on_pass(k, r.text)
        prev_total = scoring.change_ratio(text, cur) if k > 1 else 0.0
        if k > 1 and scoring.change_ratio(text, r.text) < prev_total - 0.005:      # this pass walked earlier changes back: keep the better text
            passes[-1]["discarded"] = True
            continue
        cur = r.text
    report = dict(r.report)
    intensity.RETIRED.clear()
    _C = __import__("collections").Counter()
    for ps in passes: _C.update(ps["transforms"])
    report["transforms"] = dict(_C)                                           # all passes, not only the last
    report["repeats"] = len(passes)
    report["passes"] = passes
    report["final_text_note"] = "passes marked discarded reverted earlier changes and were not used" if any(x.get("discarded") for x in passes) else ""
    report["change_ratio"] = round(scoring.change_ratio(text, cur), 3)        # total change vs the ORIGINAL input
    return Result(cur, report)
