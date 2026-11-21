#!/bin/bash
# Rebuild the Fish Speech env (S2-Pro). The clone lives in repositories/fish-speech (gitignored) and is installed editable,
# so deleting the clone breaks the env: re-run this script to restore it. Weights (~11 GB) go to the clone's checkpoints/.
set -euo pipefail
REPO=/network/scratch/g/guzmand/Repositories/Multilingual-MOS
module load miniconda/3; source ~/.bashrc
conda env list | grep -q "^Multilingual-MOS-fish-speech " || conda create -y -n Multilingual-MOS-fish-speech --override-channels -c conda-forge python=3.10 pip
conda activate Multilingual-MOS-fish-speech
[ -d "$REPO/repositories/fish-speech" ] || git clone https://github.com/fishaudio/fish-speech.git "$REPO/repositories/fish-speech"
cd "$REPO/repositories/fish-speech"
pip install -e .                       # add --no-deps when only the clone was lost and the env is otherwise intact
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro     # gated: accept the Fish Audio Research License on HF first
python -c "import fish_speech.models.text2semantic.inference; print('fish-speech OK')"
