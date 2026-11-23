#!/bin/bash
#SBATCH --job-name=tts_Hausa
#SBATCH --partition=unkillable
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=jobs/tts_Hausa-%j.out
#SBATCH --mail-type=END,FAIL
# Synthesise every TTS system assigned to Hausa for the ground-truth sentences in audios/Hausa/ground-truth/metadata.csv,
# then rebuild the listening page. 8 systems: GT + Gemini, ElevenLabs, OmniVoice, VITS (MMS hau), EveryVoice, Higgs TTS 3, Sunbird Orpheus-3B (no VoxCPM2 / Fish for Hausa).
# Each system script keeps clips that already exist with the same model, voice and text, so the job can be resubmitted to
# fill gaps; pass EXTRA="--overwrite" to regenerate everything:  EXTRA="--overwrite" sbatch jobs/synthesize_Hausa.sh
# Gemini (gemini-3.8-flash-tts): Tier 1 allows 100 requests per model per day (resets midnight Pacific); the script stops cleanly at the quota.
set -uo pipefail
LANGUAGE="Hausa"
EXTRA="${EXTRA:-}"                                     # extra arguments passed to every system script
REPO=/network/scratch/g/guzmand/Repositories/Multilingual-MOS
cd "$REPO"
module load miniconda/3 cuda/13.1.0                     # cuda: nvcc for the Higgs / sglang-omni kernels
source ~/.bashrc
for k in GEMINI_API_KEY ELEVENLABS_API_KEY; do eval "$(grep -m1 "^\s*export $k=" ~/.bashrc)"; export $k; done   # ~/.bashrc does not export them to batch shells
echo "start $(date) on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

run_system() {                                          # run_system <conda env> <script in src/synthesis, without .py>
    local env=$1 script=$2
    echo; echo "################ $LANGUAGE / $script  ($(date +%H:%M))"
    conda activate "$env"
    python "src/synthesis/$script.py" --language "$LANGUAGE" $EXTRA 2>&1 | grep -v "Warning\|warn\|deprecated\|it/s\]\|Licensed under" | tail -8 \
        || echo "!!!! $script failed"
}

conda activate Multilingual-MOS
[ -f "audios/$LANGUAGE/reference-voices/metadata.csv" ] || python src/synthesis/references.py --language "$LANGUAGE"   # cloning references

run_system Multilingual-MOS             gemini_tts
run_system Multilingual-MOS             elevenlabs_tts
run_system Multilingual-MOS             omnivoice_tts
run_system Multilingual-MOS             vits
run_system Multilingual-MOS-everyvoice  everyvoice_tts
run_system Multilingual-MOS-higgs       higgs_tts3
run_system Multilingual-MOS             sunbird_orpheus

echo; echo "################ summary"
for d in "audios/$LANGUAGE"/*/; do [ -f "$d/metadata.csv" ] && printf "%-22s %s clips\n" "$(basename "$d")" "$(($(wc -l < "$d/metadata.csv") - 1))"; done
conda activate Multilingual-MOS
python src/synthesis/make_listening_page.py --language "$LANGUAGE"
echo "end $(date)"
