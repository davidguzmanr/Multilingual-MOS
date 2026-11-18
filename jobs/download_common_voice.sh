#!/usr/bin/env bash
#SBATCH --job-name=cv_download
#SBATCH --partition=long-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Download the Common Voice Scripted Speech 27.0 releases for the 14 target locales from the Mozilla Data Collective.
# - Dataset IDs: data/Common-Voice/datasets.json; downloads go to data/Common-Voice/<locale>/ (gitignored) (resolved from the public API by locale).
# - Terms must already be accepted in the MDC web UI for every dataset (done 2026-11-16).
# - MDC_API_KEY comes from ~/.bashrc (sbatch exports the submitting shell's environment; sourced again below to be safe).
# - The API allows 7 dataset downloads per organisation per day (resets at midnight UTC): the 8th and later calls will
#   fail with 429. Re-submitting this job the next day is enough: finished archives are skipped, partial ones resume.
# - 2 minutes between languages to stay clear of the request rate limit.

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST"

##################################################################
# Activate the environment by loading Python and required packages
##################################################################
module load miniconda/3
source ~/.bashrc
export PYTHONUNBUFFERED=1

conda activate Multilingual-MOS-moss-tts      # datacollective SDK >=0.7 needs Python 3.11+

if [ -z "${MDC_API_KEY:-}" ]; then echo "MDC_API_KEY is not set"; exit 1; fi

cd /home/mila/g/guzmand/scratch/Repositories/Multilingual-MOS

##################################################################
# Run: smallest first so the daily quota covers the ones that matter most for the domain question
##################################################################
LOCALES=(ko yo ha vi hi ht id tr ar ja sw fr de es)

for loc in "${LOCALES[@]}"; do
    echo "=== $(date) :: $loc ==="
    python listening-study/tools/download_common_voice.py "$loc"
    echo "=== sleeping 120 s before the next language ==="
    sleep 120
done

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
