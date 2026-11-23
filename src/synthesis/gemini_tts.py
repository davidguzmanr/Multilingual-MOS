"""Gemini TTS through google-genai: gemini-3.8-flash-tts by default (current model, 100 requests per model per day on Tier 1,
no empty responses in ~100 requests), via generate_content with response_modalities=["AUDIO"], prebuilt voices, 24 kHz 16-bit
PCM. The OpenBibleTTS pilot used gemini-2.5-pro-preview-tts (same call; 50 requests/day, frequent "no audio" refusals):
pass --model-id gemini-2.5-pro-preview-tts to reproduce it. 4 prebuilt voices (f, m, f, m) in rotation. Needs GEMINI_API_KEY.
When the daily quota is exhausted the API may hang instead of returning 429, so every call has a hard deadline and a cheap
REST probe decides whether to stop. Env: Multilingual-MOS.
    python src/synthesis/gemini_tts.py --language Yoruba
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *

MODEL_ID, SR = "gemini-3.8-flash-tts", 24000
PREBUILT = [dict(voice_id="Kore", gender="female"), dict(voice_id="Puck", gender="male"),
            dict(voice_id="Aoede", gender="female"), dict(voice_id="Charon", gender="male")]
MAX_RETRIES, BACKOFF_BASE, HARD_TIMEOUT_S = 12, 1.6, 90      # "no audio (finish_reason=OTHER)" is random per request: the same
                                                             # sentence failed 6 times in a row in one run and succeeded first try after
args = parse_args(__doc__.split("\n")[0], lambda p: p.add_argument("--model-id", default=MODEL_ID))
MODEL_ID = args.model_id
if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_KEY"): os.environ["GEMINI_API_KEY"] = os.environ["GEMINI_KEY"]
if not os.environ.get("GEMINI_API_KEY"): sys.exit("GEMINI_API_KEY (or GEMINI_KEY) is not set")
voices = [dict(voice_id=v, gender="unknown") for v in args.voices.split(",")] if args.voices else PREBUILT

from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
client = genai.Client(http_options={"timeout": 60_000})
pool = ThreadPoolExecutor(max_workers=4)


def call_gemini(text, voice):
    """Pilot call. Raises when the model answers 200 with no audio (finish_reason SAFETY / RECITATION / OTHER / LANGUAGE),
    which the 2.5 Pro preview does on very short inputs; the caller retries."""
    cfg = types.GenerateContentConfig(response_modalities=["AUDIO"],
                                      speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                                          prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    resp = client.models.generate_content(model=MODEL_ID, contents=text, config=cfg)
    cand = resp.candidates[0]
    if cand.content is None or not cand.content.parts or cand.content.parts[0].inline_data is None:
        raise RuntimeError(f"Gemini returned no audio (finish_reason={getattr(cand, 'finish_reason', 'UNKNOWN')})")
    return cand.content.parts[0].inline_data.data                      # audio/wav (3.8 Flash) or raw PCM s16le 24 kHz (2.5 Pro preview)


def to_audio(data):
    """3.8 Flash returns a RIFF/WAV container that ends with a C2PA content-credentials chunk (~7.6 KB): read it as WAV so the
    header and that chunk never reach the output (interpreted as PCM they were a click at the start and ~0.16 s of noise at the
    end). The 2.5 Pro preview returns headerless s16le PCM at 24 kHz."""
    if data[:4] == b"RIFF":
        import io, soundfile as sf
        y, sr = sf.read(io.BytesIO(data), dtype="int16"); return y, sr
    return np.frombuffer(data, dtype="<i2"), SR


def quota_or_exit():
    """One plain REST request to read the quota state. Daily quota gone -> stop (the clips so far are kept). Per-minute limit
    (Tier 1 TTS allows only a handful of requests per minute) -> wait the delay the API asks for and carry on."""
    import re, requests
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent",
                      headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=30,
                      json={"contents": [{"parts": [{"text": "Hello there, how are you today?"}]}], "generationConfig": {"responseModalities": ["AUDIO"]}})
    if r.status_code != 429: return
    msg = r.json().get("error", {}).get("message", "")
    if "per_day" in msg or "per day" in msg:
        sys.exit(f"Gemini daily quota exhausted, stopping (finished clips are kept; rerun later): {msg[-140:]}")
    wait = re.search(r"retry in ([0-9.]+)s", msg); wait = float(wait.group(1)) + 5 if wait else 60
    print(f"   per-minute limit hit, waiting {wait:.0f} s", flush=True); time.sleep(wait)


MIN_INTERVAL_S = 10.0                                            # pacing: 6 requests per minute, safely under the Tier 1 per-minute limit
_last_call = [0.0]

def synth(text, voice):
    global client
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(max(0.0, _last_call[0] + MIN_INTERVAL_S - time.time())); _last_call[0] = time.time()
            data = pool.submit(call_gemini, text, voice["voice_id"]).result(timeout=HARD_TIMEOUT_S)
            return to_audio(data)
        except FutTimeout:
            quota_or_exit(); last = "timeout"
            print(f"   retry {attempt+1}: no response after {HARD_TIMEOUT_S} s, reconnecting", flush=True)
            client = genai.Client(http_options={"timeout": 60_000})
        except Exception as e:
            last = e; msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or " 429" in msg: quota_or_exit(); continue      # waited (per-minute) or exited (per-day)
            print(f"   retry {attempt+1}: {msg[-120:]}", flush=True); time.sleep(BACKOFF_BASE ** attempt)
    print(f"   FAILED after {MAX_RETRIES} attempts ({last}); clip skipped, rerun the script later to fill it", flush=True)
    return None                                                  # run() records the clip as missing instead of aborting the language


run("Gemini", MODEL_ID, args.language, synth, voices, dict(api="generate_content", pcm="s16le", sample_rate=SR, retries=MAX_RETRIES),
    limit=args.limit, overwrite=args.overwrite, dry_run=args.dry_run)
