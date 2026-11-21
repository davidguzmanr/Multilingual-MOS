"""VoxCPM2 (openbmb/VoxCPM2, Apache-2.0) with zero-shot cloning of the 4 reference voices. For Yoruba the default is the
community full fine-tune LyngualLabs/VoxCPM2-Yoruba (system name "VoxCPM2-Yoruba"). Env: Multilingual-MOS.
    python src/synthesis/voxcpm2.py --language Yoruba
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *

DEFAULT = {"Yoruba": ("LyngualLabs/VoxCPM2-Yoruba", "VoxCPM2-Yoruba")}
SEED, CFG, STEPS = 0, 2.0, 22
args = parse_args(__doc__.split("\n")[0], lambda p: (p.add_argument("--model-id"), p.add_argument("--system")))
model_id, system = DEFAULT.get(args.language, ("openbmb/VoxCPM2", "VoxCPM2"))
model_id, system = args.model_id or model_id, args.system or system
voices = load_references(args.language)

from huggingface_hub import snapshot_download
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained(snapshot_download(model_id), load_denoiser=False)    # local dir, as the fine-tune card does

def synth(text, voice):
    torch.manual_seed(SEED)
    wav = model.generate(text=text, prompt_wav_path=voice["audio_path"], prompt_text=voice["text"], cfg_value=CFG,
                         inference_timesteps=STEPS, normalize=False, denoise=False)
    return wav, model.tts_model.sample_rate

run(system, model_id, args.language, synth, voices, dict(seed=SEED, cfg_value=CFG, inference_timesteps=STEPS, mode="voice_clone"),
    limit=args.limit, overwrite=args.overwrite, dry_run=args.dry_run)
