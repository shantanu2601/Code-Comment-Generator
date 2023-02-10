#!/bin/bash
# Full set of runs on one language. Each summarize run trains, keeps the best
# checkpoint by validation loss, and writes test scores to <run-dir>/scores.test.json
set -euo pipefail

LANG_=${1:-java}
RUNS=${PATH_RUNS:-./runs}/$LANG_

run() { python -m codecg.train "$@" --language "$LANG_"; }

# Positional encodings and copy attention
run summarize --positions absolute --no-copy --run-dir "$RUNS/abs"
run summarize --positions relative --no-copy --run-dir "$RUNS/rel"
run summarize --positions both     --no-copy --run-dir "$RUNS/abs+rel"
run summarize --positions relative           --run-dir "$RUNS/rel+copy"

# Adversarial alignment, then a summarizer initialised from it
run autoencode --positions relative --run-dir "$RUNS/autoencoder"
run align --text-autoencoder "$RUNS/autoencoder/text_autoencoder.pt" --epochs 10 --run-dir "$RUNS/align"
run summarize --init-encoder "$RUNS/align/code_encoder.pt" --run-dir "$RUNS/rel+copy+align-enc"
run summarize --init-encoder "$RUNS/align/code_encoder.pt" \
              --init-decoder "$RUNS/autoencoder/text_autoencoder.pt" --run-dir "$RUNS/rel+copy+align"

python - "$RUNS" <<'EOF'
import json, pathlib, sys
print(f"{'run':<24}{'BLEU':>8}{'METEOR':>8}{'ROUGE-L':>9}")
for p in sorted(pathlib.Path(sys.argv[1]).glob("*/scores.test.json")):
    s = json.loads(p.read_text())
    print(f"{p.parent.name:<24}{s['bleu']:>8.2f}{s.get('meteor', float('nan')):>8.2f}{s['rouge_l']:>9.2f}")
EOF
