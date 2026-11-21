"""VITS: Meta MMS-TTS checkpoints (facebook/mms-tts-<iso3>, single speaker, 16 kHz, CC-BY-NC-4.0), or an own VITS
checkpoint for Japanese (no MMS Japanese model) via --model-id. Env: Multilingual-MOS.
    python src/synthesis/vits.py --language Yoruba
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *

MMS = {"Arabic": "ara", "French": "fra", "German": "deu", "Haitian Creole": "hat", "Hausa": "hau", "Hindi": "hin", "Indonesian": "ind",
       "Korean": "kor", "Spanish": "spa", "Swahili": "swh", "Turkish": "tur", "Vietnamese": "vie", "Yoruba": "yor"}   # no Japanese
SEED = 0

args = parse_args(__doc__.split("\n")[0], lambda p: p.add_argument("--model-id", help="HF id or path; default facebook/mms-tts-<iso3>"))
model_id = args.model_id or (f"facebook/mms-tts-{MMS[args.language]}" if args.language in MMS else sys.exit("no MMS model for this language; pass --model-id"))

from transformers import VitsModel, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(model_id); model = VitsModel.from_pretrained(model_id).to(device).eval()
if getattr(tok, "is_uroman", False): print("NOTE: this MMS tokenizer expects romanised (uroman) input")

def synth(text, voice):
    ids = tok(text, return_tensors="pt")
    n_unk = int((ids["input_ids"] == tok.unk_token_id).sum()) if tok.unk_token_id is not None else 0
    if n_unk: print(f"   WARNING: {n_unk} unknown character(s) in: {text[:60]}")
    torch.manual_seed(SEED)
    with torch.no_grad(): wav = model(**{k: v.to(device) for k, v in ids.items()}).waveform[0].cpu().numpy()
    return wav, model.config.sampling_rate

run("VITS", model_id, args.language, synth, [dict(voice_id=model_id.split("/")[-1], gender="unknown")],
    dict(seed=SEED, noise_scale=model.config.noise_scale, speaking_rate=model.config.speaking_rate),
    limit=args.limit, overwrite=args.overwrite, dry_run=args.dry_run)
