"""
Prepare UTMOSv2 finetuning data for Hindi.

Steps:
  1. Parse Hindi.csv: extract (dataset, model, filename, mos_score).
  2. Average the 3 annotator scores per unique audio file (dataset × model × filename).
  3. Hold out 10 unique *utterances* (dataset × filename pairs) for testing —
     the same utterance is excluded from training across ALL models, preventing
     any cross-model leakage of the same underlying sentence.
  4. Write per-(dataset, model) mos_list_train.txt / mos_list_test.txt files.
  5. Write data_config_train.json (all train splits) for `python train.py`.
  6. Write test_samples.csv (averaged MOS for the held-out utterances).
"""

import json
import os
import random
import urllib.parse

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIOS_DIR = os.path.join(REPO_ROOT, "data", "audios")
ANNO_CSV   = os.path.join(REPO_ROOT, "data", "annotations-anonymized", "Hindi.csv")
OUT_DIR    = os.path.join(REPO_ROOT, "data", "UTMOSv2-finetuning", "Hindi")
LANGUAGE   = "Hindi"

N_TEST_UTTERANCES = 10
RANDOM_SEED       = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load & parse ─────────────────────────────────────────────────────────
df = pd.read_csv(ANNO_CSV)

# Extract model from audio_url: .../audios/{dataset}/{model}/{language}/{filename}
url_parts   = df["audio_url"].str.split("/audios/").str[1].str.split("/")
df["model"] = url_parts.str[1]

# MOS: "5 — Excellent" → 5
df["mos_score"] = df["mos"].str.split().str[0].astype(int)

df = df[["dataset", "model", "filename", "mos_score"]]

# Sanity: every sample should have exactly 3 annotations
counts = df.groupby(["dataset", "model", "filename"]).size()
assert (counts == 3).all(), f"Expected 3 annotations per sample; got:\n{counts[counts != 3]}"

# ── 2. Average per audio file ────────────────────────────────────────────────
avg = (
    df.groupby(["dataset", "model", "filename"], as_index=False)["mos_score"]
    .mean()
    .rename(columns={"mos_score": "avg_mos"})
)
print(f"Total unique audio files: {len(avg)}")

# ── 3. Train / test split on unique utterances (dataset × filename) ──────────
# Treat each (dataset, filename) as one utterance; exclude it from ALL models.
random.seed(RANDOM_SEED)
utterances = avg[["dataset", "filename"]].drop_duplicates()
test_utterances = utterances.sample(n=N_TEST_UTTERANCES, random_state=RANDOM_SEED)
test_key = set(zip(test_utterances["dataset"], test_utterances["filename"]))

is_test = avg.apply(lambda r: (r["dataset"], r["filename"]) in test_key, axis=1)
train_df = avg[~is_test].copy()
test_df  = avg[is_test].copy()

print(f"Train audio files : {len(train_df)}")
print(f"Test  audio files : {len(test_df)}  (from {len(test_utterances)} unique utterances)")

# ── 4. Write per-(dataset, model) mos_list files ─────────────────────────────
data_config_entries = []

for (dataset, model), group in train_df.groupby(["dataset", "model"]):
    local_dir = os.path.join(AUDIOS_DIR, dataset, model, LANGUAGE)
    if not os.path.isdir(local_dir):
        print(f"  [SKIP] directory not found: {local_dir}")
        continue

    name      = f"{dataset}-{model}-{LANGUAGE}"
    list_name = f"{dataset}-{model}_train.txt"
    list_path = os.path.join(OUT_DIR, list_name)

    lines = [f"{row.filename},{row.avg_mos:.4f}" for row in group.itertuples()]
    with open(list_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [train] {name}: {len(lines)} samples → {list_name}")
    data_config_entries.append({"name": name, "dir": local_dir, "mos_list": list_path})

# Also write test mos_list files (one per dataset×model) for evaluation
test_config_entries = []
for (dataset, model), group in test_df.groupby(["dataset", "model"]):
    local_dir = os.path.join(AUDIOS_DIR, dataset, model, LANGUAGE)
    if not os.path.isdir(local_dir):
        continue

    name      = f"{dataset}-{model}-{LANGUAGE}"
    list_name = f"{dataset}-{model}_test.txt"
    list_path = os.path.join(OUT_DIR, list_name)

    lines = [f"{row.filename},{row.avg_mos:.4f}" for row in group.itertuples()]
    with open(list_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  [test]  {name}: {len(lines)} samples → {list_name}")
    test_config_entries.append({"name": name, "dir": local_dir, "mos_list": list_path})

# ── 5. Write data_config_train.json ──────────────────────────────────────────
train_cfg_path = os.path.join(OUT_DIR, "data_config_train.json")
with open(train_cfg_path, "w") as f:
    json.dump({"data": data_config_entries}, f, indent=2)
print(f"\nWrote: {train_cfg_path}")

# ── 6. Write test_samples.csv ────────────────────────────────────────────────
test_csv_path = os.path.join(OUT_DIR, "test_samples.csv")
test_df_out = test_df.copy()
test_df_out["audio_path"] = test_df_out.apply(
    lambda r: os.path.join(AUDIOS_DIR, r["dataset"], r["model"], LANGUAGE, r["filename"]),
    axis=1,
)
test_df_out.to_csv(test_csv_path, index=False)
print(f"Wrote: {test_csv_path}")

# Summary
print(f"\n{'─'*60}")
print(f"Language     : {LANGUAGE}")
print(f"Train entries: {len(data_config_entries)} (dataset × model) splits")
print(f"Test entries : {len(test_config_entries)} (dataset × model) splits")
print(f"Held-out utterances ({N_TEST_UTTERANCES} unique filenames):")
for _, row in test_utterances.iterrows():
    print(f"  {row['dataset']}/{row['filename']}")
print(f"\nTo finetune, run from UTMOSv2/:")
print(f"  python train.py --config spec_only --data_config {train_cfg_path} --weight spec_only")
