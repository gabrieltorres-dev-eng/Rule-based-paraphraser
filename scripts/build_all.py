"""Run the whole data build in order. Safe to stop (Ctrl+C) and rerun at any time:
finished steps are skipped and unfinished steps resume from their last checkpoint.

    python scripts/build_all.py               # run / resume everything
    python scripts/build_all.py --redo idioms # force one step to run again
    python scripts/build_all.py --restart     # forget all progress (keeps downloaded files)

Steps: download -> ai_pairs (DeepSeek, optional) -> phrases -> idioms -> templates
       -> fragments -> label (DeepSeek, optional) -> apply
"""
import argparse, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _progress import State, STATE_DIR

HERE = Path(__file__).resolve().parent
STEPS = [
    ("download",  ["download_data.py"]),
    ("ai_pairs",  ["gen_ai_pairs_deepseek.py"]),      # needs DEEPSEEK_API_KEY, else skipped
    ("phrases",   ["build_phrase_pairs.py"]),
    ("thesaurus", ["build_thesaurus.py"]),            # needs data/raw/moby_thesaurus.txt (download_data.py --only thesaurus)
    ("idioms",    ["build_idioms.py"]),
    ("templates", ["build_templates.py"]),
    ("fragments", ["build_fragments.py"]),
    ("label",     ["label_fragments_deepseek.py"]),   # needs DEEPSEEK_API_KEY, else stop for hand labels
    ("apply",     ["build_fragments.py", "--apply"]),
]
NEEDS_KEY = {"ai_pairs", "label"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", default="", help="comma list of steps to run again")
    ap.add_argument("--restart", action="store_true", help="clear all checkpoints")
    a = ap.parse_args()
    st = State("build_all")
    if a.restart:
        for p in STATE_DIR.glob("*"): p.unlink()
        st.clear(); print("All progress cleared.")
    for n in filter(None, a.redo.split(",")):
        st.set(n, None)
    has_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
    total = len(STEPS); t_all = time.time()
    for k, (name, cmd) in enumerate(STEPS):
        print(f"\n===== step {k+1}/{total} [{100*k//total}% overall] {name} =====", flush=True)
        if st.done(name):
            print("  already done - skipping"); continue
        if name in NEEDS_KEY and not has_key:
            if name == "ai_pairs":
                print("  no DEEPSEEK_API_KEY: skipping (HC3 + RAID still used)"); continue
            print("  no DEEPSEEK_API_KEY: fill is_fragment in data/profiles/fragment_sample.csv "
                  "by hand, then rerun this script."); return
        t = time.time()
        r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], cwd=HERE.parent)
        if r.returncode:
            sys.exit(f"\nStep '{name}' stopped (exit {r.returncode}). Progress is saved - "
                     f"just rerun: python scripts/build_all.py")
        st.mark(name)
        print(f"----- {name} done in {(time.time()-t)/60:.1f} min")
    print(f"\n===== 100% - all steps done in {(time.time()-t_all)/60:.1f} min =====")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        sys.exit("\nStopped. Progress is saved - rerun python scripts/build_all.py to continue.")
