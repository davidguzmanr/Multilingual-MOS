#!/bin/bash
# Smoke test: Fish S2-Pro, random voice (no reference), three-step CLI from docs/en/inference.md. Needs a GPU.
#   bash listening-study/environments/smoke_test_fish-speech.sh [out_dir]
set -euo pipefail
REPO=/network/scratch/g/guzmand/Repositories/Multilingual-MOS
OUT=${1:-/network/scratch/g/guzmand/tmp/smoke_fish}; mkdir -p "$OUT"
module load miniconda/3; source ~/.bashrc; conda activate Multilingual-MOS-fish-speech
cd "$REPO/repositories/fish-speech"
python fish_speech/models/text2semantic/inference.py \
    --text "Yusuf yana ƙoƙari sosai, amma bai iya buɗe tagar ba." \
    --checkpoint-path checkpoints/s2-pro --output-dir "$OUT" --seed 0
python fish_speech/models/dac/inference.py -i "$OUT/codes_0.npy" -o "$OUT/fish_s2pro_hausa.wav" \
    --checkpoint-path checkpoints/s2-pro/codec.pth
python - "$OUT/fish_s2pro_hausa.wav" <<'PY'
import sys, soundfile as sf; i = sf.info(sys.argv[1]); print(f"OK {sys.argv[1]}: {i.duration:.1f} s, {i.samplerate} Hz")
PY
