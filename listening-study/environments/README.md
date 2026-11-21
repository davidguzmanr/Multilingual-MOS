# TTS environments on the cluster

Conda envs under `/network/scratch/g/guzmand/.conda/envs`. Activate with `module load miniconda/3 && conda activate <env>`.
GPU work needs a GPU allocation (e.g. `sbatch --partition=unkillable --gpus=l40s:1 --cpus-per-task=4 --mem=32G`).

| Env | Python | Models | Notes |
|---|---|---|---|
| Multilingual-MOS | 3.10 | Gemini API, ElevenLabs API, OmniVoice, VoxCPM2, MMS-TTS (VITS), Sunbird Orpheus, Coqui | shared env; also the Jupyter kernel for `audios/dev.ipynb` |
| Multilingual-MOS-fish-speech | 3.10 | Fish S2-Pro | editable install of `repositories/fish-speech`; weights in its `checkpoints/s2-pro` (11 GB) |
| Multilingual-MOS-higgs | 3.12 | Higgs Audio v3 TTS | editable install of `repositories/sglang-omni`; served with `sgl-omni serve`; needs `module load cuda/13.1.0` |
| Multilingual-MOS-moss-tts | 3.12 | MOSS-TTS, `datacollective` SDK (Common Voice downloads) | |
| Multilingual-MOS-qwen3-tts | 3.10 | Qwen3-TTS | transformers 4.57.3 |
| Multilingual-MOS-indic | 3.10 | Indic Parler-TTS, IndicF5 | protobuf<5, sentencepiece 0.2.0 |

## Rebuilding after the clones were lost (2026-11-20)

The folder that held the clones and scripts was named `env/`, which the Python template in `.gitignore` ignores, so it was
never versioned and is gone. Clones now live in `repositories/` (also gitignored, listed explicitly). The two envs that
install their package in editable mode broke with the clones and were restored with `setup_env_fish-speech.sh` and
`setup_env_higgs.sh` (`pip install -e . --no-deps` since the dependencies were intact), plus a fresh weight download.

## Smoke tests

| Script | What it does | Last result |
|---|---|---|
| `smoke_test_fish-speech.sh` | S2-Pro, random voice, 3-step CLI (text2semantic -> codes -> codec) | 2026-11-20: Hausa sentence, 4.7 s, 44.1 kHz, 15 GB GPU |
| `smoke_test_higgs.sh` | starts `sgl-omni serve`, POSTs to `/v1/audio/speech`, saves wav | 2026-11-20: server up in 440 s, Hausa sentence, 5.0 s, 24 kHz; `nixl` import error in the log is harmless |

Fish CLI facts: `--prompt-tokens`/`--prompt-text` are optional (random voice without them); step 2 writes `codes_N.npy`
to `--output-dir`; `fish_speech/models/dac/inference.py -i codes_0.npy -o out.wav` decodes. Higgs facts: zero-shot with
`"voice": "default"`, cloning via `"references": [{"audio_path", "text"}]`; reference paths must be under
`--allowed-local-media-path`; output 24 kHz.
