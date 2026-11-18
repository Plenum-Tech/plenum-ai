"""Checks plenum-memory's metrics against the Python packages Memorilla scores with.

Writes random predictions and references, scores them with NLTK (BLEU, Porter) and
rouge_score (ROUGE-1/L), and compares with `cargo run --release --example metrics`:

    python plenum-memory/tools/check_metrics.py <out dir>
    cargo run --release -p plenum-memory --example metrics -- <out dir>/cases.json <out dir>/rust.json
    python plenum-memory/tools/check_metrics.py <out dir> --compare <out dir>/rust.json
"""

import json
import random
import sys
from pathlib import Path

from nltk.stem.porter import PorterStemmer
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(7)
VOCAB = ("the cat sat on a mat running runs ran happily happiness relational conditional generalization "
         "agreed feeding motoring hopping filing caresses ponies ties dying skies news proceed exceed "
         "Paris France capital city of is The answer yes no maybe 1969 Mark Twain Mount Everest ocean "
         "carbon dioxide breakfast diabetes eggs avocado blood sugar steady full morning ideas easy").split()


def sentence():
    return " ".join(rng.choice(VOCAB) for _ in range(rng.randint(0, 14))) + rng.choice(["", ".", "!", " ?"])


if "--compare" in sys.argv:
    cases = json.loads((out / "cases.json").read_text(encoding="utf-8"))
    want = json.loads((out / "python.json").read_text(encoding="utf-8"))
    got = json.loads(Path(sys.argv[sys.argv.index("--compare") + 1]).read_text(encoding="utf-8"))
    bad = [w for w, a, b in zip(cases["words"], want["stems"], got["stems"]) if a != b]
    print(f"porter: {len(cases['words']) - len(bad)}/{len(cases['words'])} stems equal" + (f"; differ: {bad[:10]}" if bad else ""))
    worst = 0.0
    for k in ("bleu1", "bleu4", "rouge1", "rougeL"):
        d = abs(want[k] - got[k])
        worst = max(worst, d)
        print(f"{k:7} python {want[k]:.10f}  rust {got[k]:.10f}  diff {d:.2e}")
    sys.exit(0 if not bad and worst < 1e-9 else 1)

preds = [sentence() for _ in range(300)]
refs = [sentence() for _ in range(300)]
words = sorted(set(w.lower() for w in VOCAB) | {w for w in open(__file__, encoding="utf-8").read().lower().replace("_", " ").split() if w.isalpha() and len(w) > 3})
st = PorterStemmer()
sm = SmoothingFunction().method1
hyp = [p.lower().split() for p in preds]
ref = [[r.lower().split()] for r in refs]
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
sc = [scorer.score(r, p.strip()) for p, r in zip(preds, refs)]
python = {
    "stems": [st.stem(w) for w in words],
    "bleu1": corpus_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=sm),
    "bleu4": corpus_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sm),
    "rouge1": sum(s["rouge1"].fmeasure for s in sc) / len(sc),
    "rougeL": sum(s["rougeL"].fmeasure for s in sc) / len(sc),
}
(out / "cases.json").write_text(json.dumps({"preds": preds, "refs": refs, "words": words}), encoding="utf-8")
(out / "python.json").write_text(json.dumps(python), encoding="utf-8")
print(f"{len(preds)} pairs and {len(words)} words written to {out}")
