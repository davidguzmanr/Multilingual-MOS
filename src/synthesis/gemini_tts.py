"""Gemini TTS (gemini-3.8-flash-tts) through google-genai, 4 prebuilt voices (f, m, f, m) in rotation, 24 kHz wav.
Needs GEMINI_API_KEY (or GEMINI_KEY, as in ~/.bashrc). Language is auto-detected by the model (Yoruba is not an officially listed language).
Env: Multilingual-MOS (google-genai >= 2.25).
    python src/synthesis/gemini_tts.py --language Yoruba
"""
import base64, io, os, sys, time
import soundfile as sf
sys.path.insert(0, os.path.dirname(__file__)); from common import *

MODEL_ID = "gemini-3.8-flash-tts"
PREBUILT = [dict(voice_id="Kore", gender="female"), dict(voice_id="Puck", gender="male"),
            dict(voice_id="Aoede", gender="female"), dict(voice_id="Charon", gender="male")]
args = parse_args(__doc__.split("\n")[0])
if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_KEY"): os.environ["GEMINI_API_KEY"] = os.environ["GEMINI_KEY"]   # ~/.bashrc name
if not os.environ.get("GEMINI_API_KEY"): sys.exit("GEMINI_API_KEY (or GEMINI_KEY) is not set")
voices = [dict(voice_id=v, gender="unknown") for v in args.voices.split(",")] if args.voices else PREBUILT

from google import genai
client = genai.Client(http_options={"api_version": "v1beta"})

def tts_bytes(text, voice):
    """Interactions API per https://aistudio.google.com/docs/speech-generation (unary response is a RIFF wav, 24 kHz)."""
    inter = client.interactions.create(model=MODEL_ID, input=text, response_format={"type": "audio"},
                                       generation_config={"speech_config": [{"voice": voice}]})
    return base64.b64decode(inter.output_audio.data)

def synth(text, voice):
    for attempt in range(5):
        try:
            wav, sr = sf.read(io.BytesIO(tts_bytes(text, voice["voice_id"])), dtype="int16"); return wav, sr
        except Exception as e:
            if attempt == 4: raise
            print(f"   retry {attempt+1}: {str(e)[-120:]}"); time.sleep(10 * (attempt + 1))

run("Gemini", MODEL_ID, args.language, synth, voices, dict(api="interactions", response_format="audio/wav"),
    limit=args.limit, overwrite=args.overwrite, dry_run=args.dry_run)
