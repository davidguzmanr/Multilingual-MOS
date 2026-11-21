"""Fish Audio S2-Pro (fishaudio/s2-pro, research licence) via the fish-speech API server, cloning the 4 reference voices
(reference audio + transcript sent with each request). Starts tools/api_server.py from repositories/fish-speech, waits for
it, synthesises, and stops it. Env: Multilingual-MOS-fish-speech (run from any cwd).
    python src/synthesis/fish_s2pro.py --language Yoruba
"""
import atexit, base64, io, os, subprocess, sys, time
import requests, soundfile as sf
sys.path.insert(0, os.path.dirname(__file__)); from common import *

FISH = f"{REPO}/repositories/fish-speech"; CKPT = "checkpoints/s2-pro"; MODEL_ID = "fishaudio/s2-pro"
PORT, SEED = int(os.environ.get("FISH_PORT", 8081)), 0
GEN = dict(temperature=0.7, top_p=0.7, chunk_length=200, normalize=False)      # Fish SDK defaults; no en/zh normaliser

args = parse_args(__doc__.split("\n")[0])
voices = load_references(args.language)
if not args.dry_run:
    log = open(f"/network/scratch/g/guzmand/tmp/fish_server_{PORT}.log", "w")
    srv = subprocess.Popen([sys.executable, "tools/api_server.py", "--listen", f"127.0.0.1:{PORT}", "--llama-checkpoint-path", CKPT,
                            "--decoder-checkpoint-path", f"{CKPT}/codec.pth", "--decoder-config-name", "modded_dac_vq"],
                           cwd=FISH, stdout=log, stderr=subprocess.STDOUT)
    atexit.register(srv.terminate)
    for _ in range(120):
        try: requests.get(f"http://127.0.0.1:{PORT}/v1/tts/defaults", timeout=2); break
        except requests.RequestException: pass
        if srv.poll() is not None: sys.exit(f"fish server died, see {log.name}")
        time.sleep(5)
    else: sys.exit("fish server did not come up")
refs = {v["voice_id"]: base64.b64encode(open(v["audio_path"], "rb").read()).decode() for v in voices}

def synth(text, voice):
    body = dict(text=text, references=[dict(audio=refs[voice["voice_id"]], text=voice["text"])], format="wav", seed=SEED, **GEN)
    for attempt in range(3):
        r = requests.post(f"http://127.0.0.1:{PORT}/v1/tts", json=body, timeout=600)
        if r.ok: wav, sr = sf.read(io.BytesIO(r.content), dtype="int16"); return wav, sr
        print(f"   retry {attempt+1}: HTTP {r.status_code} {r.text[:120]}"); time.sleep(5)
    raise RuntimeError(f"fish server error: {r.status_code} {r.text[:200]}")

run("Fish-S2-Pro", MODEL_ID, args.language, synth, voices, dict(seed=SEED, mode="voice_clone", **GEN),
    limit=args.limit, overwrite=args.overwrite, dry_run=args.dry_run)
