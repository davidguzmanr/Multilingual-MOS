#!/bin/bash
# Rebuild the Higgs Audio v3 TTS env (served through sglang-omni). The clone lives in repositories/sglang-omni (gitignored)
# and is installed editable; weights (~9 GB) go to the Hugging Face cache ($HF_HOME). Needs cuda/13.1.0 for flashinfer JIT.
set -euo pipefail
REPO=/network/scratch/g/guzmand/Repositories/Multilingual-MOS
module load miniconda/3 cuda/13.1.0; source ~/.bashrc
conda env list | grep -q "^Multilingual-MOS-higgs " || conda create -y -n Multilingual-MOS-higgs --override-channels -c conda-forge python=3.12 pip
conda activate Multilingual-MOS-higgs
[ -d "$REPO/repositories/sglang-omni" ] || git clone https://github.com/sgl-project/sglang-omni.git "$REPO/repositories/sglang-omni"
cd "$REPO/repositories/sglang-omni"
pip install -e .                       # add --no-deps when only the clone was lost and the env is otherwise intact
hf download bosonai/higgs-audio-v3-tts-4b
python -c "import sglang_omni; print('sglang-omni OK')"; which sgl-omni
