"""OmniVoice (k2-fsa/OmniVoice, zero-shot voice cloning, 600+ languages). Clones the 4 reference voices from
audios/<Language>/reference-voices/ (see references.py), one per sentence in rotation. Env: Multilingual-MOS.
    python src/synthesis/omnivoice_tts.py --language Yoruba
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *

MODEL_ID, SEED = "k2-fsa/OmniVoice", 0
args = parse_args(__doc__.split("\n")[0])
voices = load_references(args.language)

from omnivoice import OmniVoice
model = OmniVoice.from_pretrained(MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")
prompts = {v["voice_id"]: model.create_voice_clone_prompt(ref_audio=v["audio_path"], ref_text=v["text"]) for v in voices}

def synth(text, voice):
    torch.manual_seed(SEED)
    wav = model.generate(text=text, language=args.language, voice_clone_prompt=prompts[voice["voice_id"]])[0]
    return wav, model.sampling_rate

run("OmniVoice", MODEL_ID, args.language, synth, voices, dict(seed=SEED, language=args.language, mode="voice_clone"),
    limit=args.limit, overwrite=args.overwrite, dry_run=args.dry_run)
