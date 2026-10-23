import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paraphraser.intensity as _i; _i.LEVEL = 0   # tests exercise the human-rate behaviour

# Provenance gate: tests run with every glue phrase the human corpora attest (see scripts/build_glue.py).
import paraphraser.glue as _glue
_glue.set_glue(["which", "which means", "this means", "that is why", "this is why", "that is", "this is", "this", "that",
                "the", "a", "an", "of", "carry out", "is", "are", "was", "were", "be", "been", "by"])
