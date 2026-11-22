"""Gemini TTS, same model and call as the OpenBibleTTS pilot (open-bible-surveys/utils/inference-Gemini.py):
gemini-2.5-pro-preview-tts via generate_content with response_modalities=["AUDIO"], prebuilt voices, 24 kHz 16-bit PCM.
4 prebuilt voices (f, m, f, m) in rotation. Needs GEMINI_API_KEY (or GEMINI_KEY). Env: Multilingual-MOS.
Quota: Tier 1 allows 100 generate requests per model per day (resets midnight Pacific); when it is exhausted the API may
hang instead of returning 429, so every call has a hard deadline and a cheap REST probe decides whether to stop.
    python src/synthesis/gemini_tts.py --language Yoruba
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *

MODEL_ID, SR = "gemini-2.5-pro-preview-tts", 24000
PREBUILT = [dict(voice_id="Kore", gender="female"), dict(voice_id="Puck", gender="male"),
            dict(voice_id="Aoede", gender="female"), dict(voice_id="Charon", gender="male")]
MAX_RETRIES, BACKOFF_BASE, HARD_TIMEOUT_S = 6, 2.0, 90
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
    return cand.content.parts[0].inline_data.data                      # raw little-endian 16-bit PCM at 24 kHz


def quota_or_exit():
    """One plain REST request: when the daily quota is gone it returns the 429 (with the reset time) immediately."""
    import requests
    r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent",
                      headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=30,
                      json={"contents": [{"parts": [{"text": "Hello there, how are you today?"}]}], "generationConfig": {"responseModalities": ["AUDIO"]}})
    if r.status_code == 429:
        sys.exit(f"Gemini quota exhausted, stopping (finished clips are kept; rerun later): {r.json().get('error', {}).get('message', '')[-140:]}")


def synth(text, voice):
    global client
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            pcm = pool.submit(call_gemini, text, voice["voice_id"]).result(timeout=HARD_TIMEOUT_S)
            return np.frombuffer(pcm, dtype="<i2"), SR
        except FutTimeout:
            quota_or_exit(); last = "timeout"
            print(f"   retry {attempt+1}: no response after {HARD_TIMEOUT_S} s, reconnecting", flush=True)
            client = genai.Client(http_options={"timeout": 60_000})
        except Exception as e:
            last = e; msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or " 429" in msg: quota_or_exit()
            print(f"   retry {attempt+1}: {msg[-120:]}", flush=True); time.sleep(BACKOFF_BASE ** attempt)
    raise RuntimeError(f"Gemini TTS failed after {MAX_RETRIES} attempts: {last}")


run("Gemini", MODEL_ID, args.language, synth, voices, dict(api="generate_content", pcm="s16le", sample_rate=SR, retries=MAX_RETRIES),
    limit=args.limit, overwrite=args.overwrite, dry_run=args.dry_run)
