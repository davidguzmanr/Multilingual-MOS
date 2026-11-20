"""Download the Open Bible **test** split (parquet with embedded audio) for the target languages from
davidguzmanr/open-bible-resources into data/Open-Bible/<Language>/. Train is left out (used for EveryVoice/VITS).
    python listening-study/tools/download_open_bible.py ["Language" ...]
"""
import os, sys, time
from huggingface_hub import snapshot_download
ROOT = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS/data/Open-Bible"
LANGS = ["Arabic Standard", "Haitian Creole", "Hausa", "Hindi", "Swahili", "Turkish", "Vietnamese", "Yoruba"]
for lang in sys.argv[1:] or LANGS:
    t0 = time.time()
    snapshot_download("davidguzmanr/open-bible-resources", repo_type="dataset", local_dir=ROOT, allow_patterns=[f"{lang}/test*"])
    files = os.listdir(f"{ROOT}/{lang}")
    print(f"== {lang}: {files} in {time.time()-t0:.0f}s", flush=True)
