"""Download the newest Common Voice Scripted Speech release for each target locale from the Mozilla Data Collective.

Needs: the `datacollective` SDK (>=0.7, Python 3.11+; installed in Multilingual-MOS-moss-tts), MDC_API_KEY exported,
and the dataset terms accepted once in the MDC web UI for each dataset (the API cannot accept them).
    module load miniconda/3 && conda activate Multilingual-MOS-moss-tts
    export MDC_API_KEY=...
    python listening-study/tools/download_common_voice.py [locale ...]      # default: all in datasets.json
Dataset IDs come from data/Common-Voice/datasets.json (resolved from the public API by locale).
Downloads resume automatically; the API rate limit is ~30 downloads per day.
"""
import json, os, sys, time
from datacollective import download_dataset
ROOT = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS/data/Common-Voice"
meta = json.load(open(f"{ROOT}/datasets.json"))
locales = sys.argv[1:] or list(meta)
if not os.environ.get("MDC_API_KEY"): sys.exit("MDC_API_KEY is not set")
for loc in locales:
    m = meta[loc]; out = f"{ROOT}/{loc}"; os.makedirs(out, exist_ok=True); t = time.time()
    print(f"== {m['language']} ({loc}): {m['name']} [{m['id']}] {m['size_gb']} GB -> {out}", flush=True)
    try:
        path = download_dataset(m["id"], download_directory=out, extract=True, show_progress=False)
        print(f"   done in {time.time()-t:.0f}s -> {path}", flush=True)
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
