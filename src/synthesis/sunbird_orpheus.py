"""Sunbird Orpheus-3B multilingual (Sunbird/orpheus-3b-tts-multilingual, Apache-2.0): Llama-3B emitting SNAC codes,
decoded with hubertsiuzdak/snac_24khz. The speaker id carries the language (waxal_yor_*, waxal_hau_*); all ids of the
language are rotated over the sentences (gender not documented). Plain transformers, bf16, ~14 GB GPU. Env: Multilingual-MOS.
    python src/synthesis/sunbird_orpheus.py --language Yoruba
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *

MODEL_ID, SNAC_ID = "Sunbird/orpheus-3b-tts-multilingual", "hubertsiuzdak/snac_24khz"
SPEAKERS = {"Yoruba": ["waxal_yor_0002", "waxal_yor_0006", "waxal_yor_0008"],
            "Hausa": ["waxal_hau_0004", "waxal_hau_0006", "waxal_hau_0007", "waxal_hau_0008"]}
SOH, EOH, EOT, SOS, EOS, AUDIO_LO, CODEBOOK = 128259, 128260, 128009, 128257, 128258, 128266, 4096
GEN = dict(max_new_tokens=1200, do_sample=True, temperature=0.6, top_p=0.95, repetition_penalty=1.1)
SEED = 0

args = parse_args(__doc__.split("\n")[0])
if args.language not in SPEAKERS and not args.voices: sys.exit("no Sunbird speakers for this language")
voices = [dict(voice_id=s, gender="unknown") for s in (args.voices.split(",") if args.voices else SPEAKERS[args.language])]

from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
snac = SNAC.from_pretrained(SNAC_ID).eval()        # CPU decode is fast enough

def decode(tokens):
    """Orpheus layout: 7 codes per frame -> SNAC layers [1, 2, 4] codes per frame."""
    t = tokens.tolist()
    if SOS in t: t = t[len(t) - 1 - t[::-1].index(SOS) + 1:]           # after the last start-of-speech
    t = [x for x in t if AUDIO_LO <= x < AUDIO_LO + 7 * CODEBOOK]
    t = t[:len(t) // 7 * 7]
    frames = [[t[i + k] - AUDIO_LO - k * CODEBOOK for k in range(7)] for i in range(0, len(t), 7)]
    frames = [[min(max(c, 0), CODEBOOK - 1) for c in f] for f in frames]
    l1 = [f[0] for f in frames]; l2 = [c for f in frames for c in (f[1], f[4])]; l3 = [c for f in frames for c in (f[2], f[3], f[5], f[6])]
    codes = [torch.tensor(l, dtype=torch.long).unsqueeze(0) for l in (l1, l2, l3)]
    with torch.no_grad(): return snac.decode(codes).squeeze().cpu().numpy()

def synth(text, voice):
    ids = tok(f"{voice['voice_id']}: {text}", return_tensors="pt").input_ids
    prompt = torch.cat([torch.tensor([[SOH]]), ids, torch.tensor([[EOT, EOH]])], dim=1).to("cuda")
    torch.manual_seed(SEED)
    with torch.no_grad(): out = model.generate(prompt, attention_mask=torch.ones_like(prompt), eos_token_id=EOS, pad_token_id=128263, **GEN)
    return decode(out[0].cpu()), 24000

run("Sunbird-Orpheus-3B", MODEL_ID, args.language, synth, voices, dict(seed=SEED, snac=SNAC_ID, **GEN),
    limit=args.limit, overwrite=args.overwrite, dry_run=args.dry_run)
