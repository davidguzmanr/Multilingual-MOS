#!/bin/bash
# Synthesise every system assigned to one language, each script in its own conda env. GPU node required.
#   bash src/synthesis/run_language.sh Yoruba [extra args passed to every script, e.g. --limit 2 --overwrite]
# Systems per language follow listening-study/README.md ("Candidate TTS systems per language").
set -uo pipefail
LANG_NAME=$1; shift; EXTRA=("$@")
REPO=/network/scratch/g/guzmand/Repositories/Multilingual-MOS; cd "$REPO"
module load miniconda/3 cuda/13.1.0 2>/dev/null; source ~/.bashrc

declare -A SYSTEMS   # language -> space-separated script names (without .py)
SYSTEMS[Yoruba]="gemini_tts elevenlabs_tts omnivoice_tts vits everyvoice_tts voxcpm2 fish_s2pro sunbird_orpheus"
SYSTEMS[Hausa]="gemini_tts elevenlabs_tts omnivoice_tts vits everyvoice_tts higgs_tts3 sunbird_orpheus"
SYSTEMS[Swahili]="gemini_tts elevenlabs_tts omnivoice_tts vits everyvoice_tts voxcpm2 fish_s2pro higgs_tts3"
declare -A ENV=( [gemini_tts]=Multilingual-MOS [elevenlabs_tts]=Multilingual-MOS [omnivoice_tts]=Multilingual-MOS [vits]=Multilingual-MOS
                 [voxcpm2]=Multilingual-MOS [sunbird_orpheus]=Multilingual-MOS [everyvoice_tts]=Multilingual-MOS-everyvoice
                 [fish_s2pro]=Multilingual-MOS-fish-speech [higgs_tts3]=Multilingual-MOS-higgs )

[ -n "${SYSTEMS[$LANG_NAME]:-}" ] || { echo "no system list for $LANG_NAME"; exit 1; }
[ -f "audios/$LANG_NAME/reference-voices/metadata.csv" ] || { conda activate Multilingual-MOS; python src/synthesis/references.py --language "$LANG_NAME"; }
for s in ${SYSTEMS[$LANG_NAME]}; do
    echo; echo "################ $LANG_NAME / $s  ($(date +%H:%M))"
    conda activate "${ENV[$s]}"
    python "src/synthesis/$s.py" --language "$LANG_NAME" "${EXTRA[@]}" 2>&1 | grep -v "Warning\|warn\|deprecated\|it/s\]" | tail -8 \
        || echo "!!!! $s failed"
done
echo; echo "################ summary"; for d in "audios/$LANG_NAME"/*/; do [ -f "$d/metadata.csv" ] && printf "%-22s %s clips\n" "$(basename "$d")" "$(($(wc -l < "$d/metadata.csv") - 1))"; done
