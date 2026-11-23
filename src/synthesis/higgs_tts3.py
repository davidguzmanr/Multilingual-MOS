"""Higgs Audio v3 TTS (bosonai/higgs-audio-v3-tts-4b, 24 kHz) served by sglang-omni, cloning the 4 reference voices
(reference audio path + transcript per request). Starts `sgl-omni serve` from repositories/sglang-omni, waits for it
(first start compiles kernels, ~8-15 min), synthesises, and stops it. Env: Multilingual-MOS-higgs with cuda/13.1.0 loaded.
    python src/synthesis/higgs_tts3.py --language Hausa
"""
import atexit, io, os, subprocess, sys, time
import requests, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__)); from common import *

SGL = f"{REPO}/repositories/sglang-omni"; MODEL_ID = "bosonai/higgs-audio-v3-tts-4b"
PORT = int(os.environ.get("HIGGS_PORT", 8000))
GEN = dict(temperature=0.8, top_k=50, max_new_tokens=1024)        # values from the sglang-omni Higgs cookbook

args = parse_args(__doc__.split("\n")[0])
voices = load_references(args.language)
if not args.dry_run:
    env = {**os.environ, "SGLANG_OMNI_STARTUP_TIMEOUT": os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "3600")}
    log = open(f"/network/scratch/g/guzmand/tmp/higgs_server_{PORT}.log", "w")
    srv = subprocess.Popen(["sgl-omni", "serve", "--model-path", MODEL_ID, "--port", str(PORT), "--allowed-local-media-path", f"{REPO}/audios"],
                           cwd=SGL, stdout=log, stderr=subprocess.STDOUT, env=env)
    atexit.register(srv.terminate)
    for _ in range(360):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok: break
        except requests.RequestException: pass
        if srv.poll() is not None: sys.exit(f"higgs server died, see {log.name}")
        time.sleep(10)
    else: sys.exit("higgs server did not come up within an hour")

def synth(text, voice):
    # cap the generation at ~2x the expected duration (0.1 s per character + 2 s, at 25 tokens/s) so a runaway stops early
    # instead of at the 1024-token / 40.7 s ceiling observed on 4 of 72 Hausa+Swahili clips
    expected_s = 0.1 * len(text) + 2.0
    gen = {**GEN, "max_new_tokens": int(min(GEN["max_new_tokens"], 25 * min(2.0 * expected_s, expected_s + 10.0)))}   # at most +10 s over the estimate
    body = dict(model=MODEL_ID, voice="default", input=text, references=[dict(audio_path=voice["audio_path"], text=voice["text"])], **gen)
    for attempt in range(3):
        r = requests.post(f"http://127.0.0.1:{PORT}/v1/audio/speech", json=body, timeout=600)
        if r.ok: wav, sr = sf.read(io.BytesIO(r.content), dtype="int16"); return wav, sr
        print(f"   retry {attempt+1}: HTTP {r.status_code} {r.text[:120]}"); time.sleep(5)
    raise RuntimeError(f"higgs server error: {r.status_code} {r.text[:200]}")

run("Higgs-TTS-3", MODEL_ID, args.language, synth, voices, dict(mode="voice_clone", **GEN),
    limit=args.limit, overwrite=args.overwrite, dry_run=args.dry_run)
