"""ElevenLabs eleven_v3 through the official SDK, 4 premade voices (f, m, f, m) in rotation, 24 kHz PCM. Needs
ELEVENLABS_API_KEY (text_to_speech permission only; models/voices listing is not required). The language is
auto-detected: eleven_v3 rejects language_code for e.g. "yo". Env: Multilingual-MOS.
    python src/synthesis/elevenlabs_tts.py --language Yoruba
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *

MODEL_ID, SR, SEED = "eleven_v3", 24000, 0
PREMADE = [dict(voice_id="EXAVITQu4vr4xnSDxMaL", name="Sarah", gender="female"), dict(voice_id="JBFqnCBsd6RMkjVDRZzb", name="George", gender="male"),
           dict(voice_id="FGY2WhTYpPnrIDTdsKH5", name="Laura", gender="female"), dict(voice_id="TX3LPaxmHKxFdv7VOQHJ", name="Liam", gender="male")]
args = parse_args(__doc__.split("\n")[0])
if not os.environ.get("ELEVENLABS_API_KEY"): sys.exit("ELEVENLABS_API_KEY is not set")
voices = [dict(voice_id=v, gender="unknown") for v in args.voices.split(",")] if args.voices else PREMADE

from elevenlabs import ElevenLabs
client = ElevenLabs()

def synth(text, voice):
    for attempt in range(5):
        try:
            pcm = b"".join(client.text_to_speech.convert(voice_id=voice["voice_id"], text=text, model_id=MODEL_ID,
                                                        output_format=f"pcm_{SR}", seed=SEED))
            return np.frombuffer(pcm, dtype="<i2"), SR
        except Exception as e:
            if attempt == 4: raise
            print(f"   retry {attempt+1}: {str(e)[-120:]}"); time.sleep(5 * (attempt + 1))

run("ElevenLabs", MODEL_ID, args.language, synth, voices, dict(seed=SEED, output_format=f"pcm_{SR}", language_code=None, voice_settings="default"),
    limit=args.limit, overwrite=args.overwrite, dry_run=args.dry_run)
