"""Download the FLEURS test split (TSV + 16 kHz wavs) for the target languages
TSV columns (no header): id, filename, raw_transcription, transcription, chars, num_samples, gender. from google/fleurs into data/FLEURS/<config>/.
    python listening-study/tools/download_fleurs.py [config ...]      # default: all 13 (Haitian Creole has no FLEURS)
"""
import os, shutil, sys, tarfile, time
from huggingface_hub import hf_hub_download
ROOT = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS/data/FLEURS"
CONFIGS = ["ja_jp", "ko_kr", "hi_in", "vi_vn", "id_id", "yo_ng", "sw_ke", "ha_ng", "fr_fr", "es_419", "de_de", "ar_eg", "tr_tr"]
for cfg in sys.argv[1:] or CONFIGS:
    out = f"{ROOT}/{cfg}"; os.makedirs(out, exist_ok=True)
    if os.path.isdir(f"{out}/test") and os.path.exists(f"{out}/test.tsv"):
        print(f"== {cfg}: already extracted, skipping"); continue
    t0 = time.time()
    tsv = hf_hub_download("google/fleurs", f"data/{cfg}/test.tsv", repo_type="dataset", local_dir=out)
    tar = hf_hub_download("google/fleurs", f"data/{cfg}/audio/test.tar.gz", repo_type="dataset", local_dir=out)
    with tarfile.open(tar) as tf: tf.extractall(out)      # creates <out>/test/*.wav
    os.replace(tsv, f"{out}/test.tsv"); os.remove(tar)
    for d in ("data", ".cache"): shutil.rmtree(f"{out}/{d}", ignore_errors=True)   # hf_hub_download leftovers
    print(f"== {cfg}: {len(os.listdir(f'{out}/test')):,} wavs in {time.time()-t0:.0f}s")
