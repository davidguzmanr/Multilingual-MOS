"""Pick the reference voices used by the zero-shot cloning models (OmniVoice, VoxCPM2, Fish S2-Pro, Higgs TTS 3):
by default 2 female + 2 male speakers from the language's Common Voice release who are NOT among the ground-truth
speakers, one clip each (closest to --target-s, with its transcript), copied as 16-bit wav to
audios/<Language>/reference-voices/ with metadata.csv. The same four voices are reused by every cloning model.
    python src/synthesis/references.py --language Yoruba
Fallbacks: --allow-gt-speakers (tiny locales such as Haitian Creole) or --from-fleurs GENDER (no speaker ids, gender only).
"""
import argparse, glob, hashlib, os
import pandas as pd, soundfile as sf

REPO = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS"
DATA = f"{REPO}/data"
LANG_CODES = {"Arabic": "ar", "French": "fr", "German": "de", "Haitian Creole": "ht", "Hausa": "ha", "Hindi": "hi",
              "Indonesian": "id", "Japanese": "ja", "Korean": "ko", "Spanish": "es", "Swahili": "sw", "Turkish": "tr",
              "Vietnamese": "vi", "Yoruba": "yo"}
COLUMNS = ["ref_id", "language", "lang_code", "dataset", "source", "speaker_id", "gender", "age", "text", "duration_s",
           "sample_rate", "orig_id", "audio_path"]

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--language", required=True, choices=sorted(LANG_CODES))
ap.add_argument("--per-gender", type=int, default=2)
ap.add_argument("--min-s", type=float, default=5.0); ap.add_argument("--max-s", type=float, default=15.0)
ap.add_argument("--target-s", type=float, default=8.0, help="pick each speaker's clip closest to this duration")
ap.add_argument("--allow-gt-speakers", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
lang, code = args.language, LANG_CODES[args.language]
out = f"{REPO}/audios/{lang}/reference-voices"; os.makedirs(out, exist_ok=True)

# candidates: validated Common Voice clips with a gender tag, good votes, in the duration window
d = glob.glob(f"{DATA}/Common-Voice/{code}/*/cv-corpus-*/{code}")[0]
v = pd.read_csv(f"{d}/validated.tsv", sep="\t", quoting=3, dtype=str, on_bad_lines="skip")
s = pd.read_csv(f"{d}/validated_sentences.tsv", sep="\t", quoting=3, dtype=str, on_bad_lines="skip", usecols=["sentence_id", "source"])
dur = pd.read_csv(f"{d}/clip_durations.tsv", sep="\t", dtype=str, names=["path", "ms"], header=0)
v = v.merge(s, on="sentence_id", how="left").merge(dur, on="path", how="left")
v["duration_s"] = v.ms.astype(float) / 1000
v["gender"] = v.gender.map({"female_feminine": "female", "male_masculine": "male"})
v["speaker_id"] = v.client_id.str[:12]
v = v[v.gender.notna() & v.duration_s.between(args.min_s, args.max_s) & (v.up_votes.astype(int) >= 2) & (v.down_votes.astype(int) == 0)]
gt = pd.read_csv(f"{REPO}/audios/{lang}/ground-truth/metadata.csv", dtype=str)
if not args.allow_gt_speakers:
    v = v[~v.speaker_id.isin(set(gt.speaker_id))]
v = v[~v.sentence.isin(set(gt.text))]                                    # never a ground-truth sentence

rows = []
for gender in ["female", "male"]:
    pool = v[v.gender == gender]
    counts = pool.speaker_id.value_counts()
    spk = counts[counts >= 3].index.tolist()[:args.per_gender]           # most-recorded speakers of that gender
    if len(spk) < args.per_gender: print(f"WARNING: only {len(spk)} {gender} speaker(s) with >=3 clips for {lang}")
    for k, sp in enumerate(spk, 1):
        c = pool[pool.speaker_id == sp].copy()
        c["_key"] = (c.duration_s - args.target_s).abs()                  # closest to the target duration, deterministic
        r = c.sort_values(["_key", "path"]).iloc[0]
        ref_id = f"{code}-ref-{gender[0]}{k}"
        y, sr = sf.read(f"{d}/clips/{r.path}")
        path = f"{out}/{ref_id}.wav"; sf.write(path, y, sr, subtype="PCM_16")
        rows.append(dict(ref_id=ref_id, language=lang, lang_code=code, dataset="Common Voice", source=f"Common Voice: {r.source}",
                         speaker_id=sp, gender=gender, age=r.age if isinstance(r.age, str) else "", text=r.sentence.strip(),
                         duration_s=round(r.duration_s, 2), sample_rate=sr, orig_id=r.path, audio_path=os.path.relpath(path, REPO)))
meta = pd.DataFrame(rows, columns=COLUMNS); meta.to_csv(f"{out}/metadata.csv", index=False)
print(meta[["ref_id", "speaker_id", "gender", "age", "duration_s", "sample_rate", "text"]].to_string(index=False))
print(f"\n{len(meta)} reference voices -> {out}/metadata.csv")
