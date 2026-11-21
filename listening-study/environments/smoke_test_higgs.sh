#!/bin/bash
# Smoke test: Higgs Audio v3 TTS via the sglang-omni server (docs/cookbook/higgs_tts.md). Needs a GPU; first start is slow
# (flashinfer JIT, ~15 min) hence the 3600 s startup timeout and the CUDA module for nvcc.
#   bash listening-study/environments/smoke_test_higgs.sh [out_dir] [port]
set -euo pipefail
REPO=/network/scratch/g/guzmand/Repositories/Multilingual-MOS
OUT=${1:-/network/scratch/g/guzmand/tmp/smoke_higgs}; PORT=${2:-8000}; mkdir -p "$OUT"
module load miniconda/3 cuda/13.1.0; source ~/.bashrc; conda activate Multilingual-MOS-higgs
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
cd "$REPO/repositories/sglang-omni"
sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port "$PORT" \
    --allowed-local-media-path "$REPO/audios" > "$OUT/server.log" 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 360); do
    curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1 && break
    kill -0 $SERVER 2>/dev/null || { echo "server died, see $OUT/server.log"; tail -30 "$OUT/server.log"; exit 1; }
    sleep 10
done
echo "server up after ~$((i*10)) s"
curl -sf -X POST "http://localhost:$PORT/v1/audio/speech" -H "Content-Type: application/json" \
    -d '{"model": "bosonai/higgs-audio-v3-tts-4b", "voice": "default", "input": "Yusuf yana ƙoƙari sosai, amma bai iya buɗe tagar ba."}' \
    --output "$OUT/higgs_hausa.wav"
python - "$OUT/higgs_hausa.wav" <<'PY'
import sys, soundfile as sf; i = sf.info(sys.argv[1]); print(f"OK {sys.argv[1]}: {i.duration:.1f} s, {i.samplerate} Hz")
PY
