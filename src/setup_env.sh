#!/usr/bin/env bash
# Build the shared `UTMOS` conda environment for BOTH UTMOS22 and UTMOSv2.
#
# The two repos have conflicting pins -- UTMOS22 asks for torch 1.11 /
# pytorch-lightning 1.5 / transformers ^4.15, UTMOSv2 requires torch>=2.1.1 and
# transformers>=4.42.4.  The versions below are the narrow window where both
# actually run:
#
#   torch 2.1.2          oldest torch satisfying UTMOSv2; also the LAST torchaudio
#                        that still ships sox bindings, which WavAugment needs.
#   transformers 4.44.2  satisfies UTMOSv2's >=4.42.4 and still exports `AdamW`,
#                        which UTMOS22's lightning_module.py imports at module load.
#   numpy <2             torch 2.1.2 predates the numpy 2 ABI break.
#   pytorch-lightning    last 1.x. UTMOS22 uses validation_epoch_end /
#     1.9.5              test_epoch_end / `gpus=`, all removed in PL 2.0.
#   setuptools 69.5.1    setuptools >=81 dropped pkg_resources, which
#                        lightning_fabric 1.9.5 still calls.
#   hydra-core 1.1.1     fairseq pins hydra-core>=1.0.7,<=1.1.1.
#
# Usage:  bash src/setup_env.sh
set -euo pipefail

module load miniconda/3
conda activate UTMOS

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${TMPDIR:-/tmp}/utmos-build"
mkdir -p "$SCRATCH"

echo "==> 1/7  torch 2.1.2 + cu121 (L40S is sm_89, needs cu118+)"
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

echo "==> 2/7  UTMOSv2 (editable) + numpy pin"
pip install "numpy<2"
pip install -e "${REPO_ROOT}/repositories/UTMOSv2[train,optional]"
# UTMOSv2's floor pulls transformers 5.x, which disables itself on torch<2.5
pip install "transformers==4.44.2"

echo "==> 3/7  UTMOS22 python deps"
pip install "setuptools==69.5.1" wheel
pip install "pytorch-lightning==1.9.5" "hydra-core==1.1.1" "omegaconf==2.1.2" \
            phonemizer soundfile python-Levenshtein wandb scikit-learn \
            matplotlib seaborn pandas gpytorch lightgbm gdown \
            tensorboardX tensorboard datasets

echo "==> 4/7  fairseq (sarulab-speech fork, branch for_utmos)"
rm -rf "${SCRATCH}/fairseq"
git clone --depth 1 -b for_utmos https://github.com/sarulab-speech/fairseq.git "${SCRATCH}/fairseq"
pip install cython cffi "sacrebleu>=1.4.12" bitarray regex
# numpy>=1.24 removed np.float / np.int; upstream fairseq 0.12.2 made the same fix
grep -rl 'np\.float\b\|np\.int\b\|np\.object\b\|np\.bool\b' --include=*.py "${SCRATCH}/fairseq/fairseq/" \
  | xargs -r sed -i 's/\bnp\.float\b/float/g; s/\bnp\.int\b/int/g; s/\bnp\.object\b/object/g; s/\bnp\.bool\b/bool/g'
pip install --no-build-isolation "${SCRATCH}/fairseq"

echo "==> 5/7  WavAugment (needs libsox at runtime; none on the host, so use conda's)"
conda install -y -c conda-forge sox
pip install git+https://github.com/facebookresearch/WavAugment.git

echo "==> 6/7  SSL checkpoint for the UTMOS22 strong learner"
cd "${REPO_ROOT}/repositories/UTMOS22/fairseq_checkpoints"
[ -f wav2vec_small.pt ] || wget -q --show-progress \
  https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -O wav2vec_small.pt

echo "==> 7/7  verify"
python "${REPO_ROOT}/src/verify_env.py"
