"""Pick the reference voices for the zero-shot cloning models (OmniVoice, VoxCPM2, Fish S2-Pro, Higgs TTS 3): 2 female + 2 male
Common Voice clips from distinct speakers under gates tighter than the ground truth (a reference sets the quality ceiling of
four systems) plus an ASR check of the transcript that is fed to the cloner. Run BEFORE the language's sampler: the sampler
excludes the reference speakers and sentences from its own Common Voice picks. No silent relaxation: if a gender cannot be
filled the script says so and stops at fewer references. Metrics go to audios/<Language>/reference-voices/metadata.csv
(shared quality cache with the sampler: data/quality-cache/<code>.csv). WaxalNLP options exist but are off by default.
    python src/synthesis/references.py --language Yoruba
"""
import argparse, glob, hashlib, io, os, subprocess, sys, tempfile
import pandas as pd, soundfile as sf

REPO = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS"; DATA = f"{REPO}/data"
sys.path.insert(0, f"{REPO}/src"); from audio_quality import measure, passes, METRIC_COLUMNS, DNSMOS_COLUMNS   # noqa: E402
from dnsmos import DNSMOS   # noqa: E402
LANG_CODES = {"Arabic": "ar", "French": "fr", "German": "de", "Haitian Creole": "ht", "Hausa": "ha", "Hindi": "hi", "Indonesian": "id",
              "Japanese": "ja", "Korean": "ko", "Spanish": "es", "Swahili": "sw", "Turkish": "tr", "Vietnamese": "vi", "Yoruba": "yo"}
WAXAL = {"Yoruba": "yor", "Hausa": "hau", "Swahili": "swa"}
ASR_PYTHON = "/home/mila/g/guzmand/scratch/.conda/envs/Multilingual-MOS-asr/bin/python"
# Calibrated on the Hausa/Yoruba pools: Common Voice clips carry 0.6-1.0 s edges from the recording tool and even clean
# crowd speakers sit at DNSMOS OVRL ~3.2, so 0.6 s / OVRL 3.4 left no candidates; level, SNR and background stay strict.
STRICT = dict(min_loudness_lufs=-24.0, min_peak_dbfs=-6.0, min_snr_db=40.0, max_noise_floor_dbfs=-60.0, max_edge_silence_s=1.0,
              max_total_edge_s=1.5, min_speech_ratio=0.5, max_clip_ratio=0.001, max_band_floor_db=-45.0, min_dnsmos_bak=4.0, min_dnsmos_ovrl=3.2)
RELAX = [dict(), dict(max_edge_silence_s=1.5, max_total_edge_s=2.5), dict(max_edge_silence_s=1.5, max_total_edge_s=2.5, min_loudness_lufs=-28.0, min_peak_dbfs=-14.0),
         dict(max_edge_silence_s=1.5, max_total_edge_s=2.5, min_loudness_lufs=-28.0, min_peak_dbfs=-14.0, min_dnsmos_bak=3.8, min_dnsmos_ovrl=3.0, max_band_floor_db=-35.0),
         dict(max_edge_silence_s=2.5, max_total_edge_s=4.0, min_loudness_lufs=-28.0, min_peak_dbfs=-14.0, min_dnsmos_bak=3.8, min_dnsmos_ovrl=3.0, max_band_floor_db=-35.0,
              min_speech_ratio=0.3)]                                                   # last resort: studio clips padded with silence (WaxalNLP)
MAX_CER, MIN_CHARS, CANDIDATES_PER_GENDER = 0.10, 40, 8
NO_DIGITS = lambda s: not any(ch.isdigit() for ch in str(s))      # reference transcripts without numbers, like the ground truth
NO_ACRONYMS = lambda s: not __import__('re').search(r"\b[A-Z]{2,}\b|\b(?:[A-Z]\.){2,}", str(s))
ALLOWED_PUNCT = set(".,;:!?'’‘\"“”«»-–—()")
NO_SYMBOLS = lambda s: all(__import__('unicodedata').category(ch)[0] in "LM" or ch.isspace() or ch in ALLOWED_PUNCT for ch in str(s))
COLUMNS = ["ref_id", "language", "lang_code", "dataset", "source", "speaker_id", "gender", "age", "text", "duration_s", "sample_rate", "orig_id",
           "audio_path", "asr_hyp", "cer", "cer_plain", "relaxed", "trimmed"] + METRIC_COLUMNS + DNSMOS_COLUMNS


def trim_silence(audio, pad_s=0.2, drop_db=35.0):
    """Cut leading/trailing silence (frames more than drop_db below the 90th-percentile level), keep pad_s on each side.
    Used for *reference* clips only: WaxalNLP studio files are padded with several seconds of silence."""
    import numpy as np
    y, sr = (sf.read(audio, dtype="float32") if isinstance(audio, str) else sf.read(io.BytesIO(audio), dtype="float32"))
    mono = y.mean(axis=1) if y.ndim > 1 else y; frame, hop = int(0.025 * sr), int(0.010 * sr)
    idx = np.arange(0, len(mono) - frame + 1, hop); rms = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(mono, frame)[idx] ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-9)); active = np.where(db > np.percentile(db, 90) - drop_db)[0]
    a = max(0, int(idx[active[0]]) - int(pad_s * sr)); b = min(len(mono), int(idx[active[-1]]) + frame + int(pad_s * sr))
    out = io.BytesIO(); sf.write(out, y[a:b], sr, format="WAV", subtype="PCM_16"); return out.getvalue(), round(len(mono) / sr, 2), round((b - a) / sr, 2)

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--language", required=True, choices=sorted(LANG_CODES)); ap.add_argument("--per-gender", type=int, default=2)
ap.add_argument("--min-s", type=float, default=5.0); ap.add_argument("--max-s", type=float, default=10.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--relax", action="store_true", help="allow the stepwise relaxation ladder when a gender cannot be filled (off: stop at fewer references)")
ap.add_argument("--exclude-speakers", default="", help="comma-separated speaker ids kept out of the references (e.g. the only large female speaker, reserved for the ground truth)")
ap.add_argument("--waxal-per-gender", type=int, default=0, help="fill this many slots per gender from WaxalNLP studio speakers first (rest from Common Voice)")
ap.add_argument("--waxal-allow-gt", action="store_true", help="let WaxalNLP references come from speakers who also narrate ground-truth clips (never the same sentences)")
ap.add_argument("--waxal-trimmed-extra", choices=["female", "male"], help="append one extra WaxalNLP reference of this gender with leading/trailing silence TRIMMED "
                "(reference audio only, never ground truth); the silence rules are then applied to the trimmed clip")
args = ap.parse_args(); lang, code = args.language, LANG_CODES[args.language]
out = f"{REPO}/audios/{lang}/reference-voices"; os.makedirs(out, exist_ok=True)
gt_path = f"{REPO}/audios/{lang}/ground-truth/metadata.csv"                 # optional: only used by the WaxalNLP options
gt = pd.read_csv(gt_path, dtype=str) if os.path.exists(gt_path) else pd.DataFrame(columns=["speaker_id", "text"])
gt_speakers, gt_texts = set(gt.speaker_id), set(gt.text)
dn = DNSMOS(); CACHE = f"{DATA}/quality-cache/{code}.csv"
_cache = pd.read_csv(CACHE, dtype={"key": str}).set_index("key").to_dict("index") if os.path.exists(CACHE) else {}


def metrics(dataset, orig_id, audio):
    key = f"{dataset}|{orig_id}"
    if key not in _cache or any(c not in _cache[key] for c in METRIC_COLUMNS + DNSMOS_COLUMNS):
        m = {**measure(audio), **dn.score(audio)}; _cache[key] = {c: m[c] for c in METRIC_COLUMNS + DNSMOS_COLUMNS}
        os.makedirs(os.path.dirname(CACHE), exist_ok=True); pd.DataFrame.from_dict(_cache, orient="index").rename_axis("key").to_csv(CACHE)
    return dict(_cache[key])


# ---- candidate pool: Common Voice (speakers with gender, not in GT) + WaxalNLP test speakers not in GT
cands = []
d = glob.glob(f"{DATA}/Common-Voice/{code}/*/cv-corpus-*/{code}")
if d:
    d = d[0]
    v = pd.read_csv(f"{d}/validated.tsv", sep="\t", quoting=3, dtype=str, on_bad_lines="skip")
    s = pd.read_csv(f"{d}/validated_sentences.tsv", sep="\t", quoting=3, dtype=str, on_bad_lines="skip", usecols=["sentence_id", "source"])
    dur = pd.read_csv(f"{d}/clip_durations.tsv", sep="\t", dtype=str, names=["path", "ms"], header=0)
    v = v.merge(s, on="sentence_id", how="left").merge(dur, on="path", how="left")
    v["duration_s"] = v.ms.astype(float) / 1000; v["gender"] = v.gender.map({"female_feminine": "female", "male_masculine": "male"}); v["speaker_id"] = v.client_id.str[:12]
    v = v[v.gender.notna() & v.duration_s.between(args.min_s, args.max_s) & (v.up_votes.astype(int) >= 2) & (v.down_votes.astype(int) == 0)
          & (v.sentence.str.strip().str.len() >= MIN_CHARS) & v.sentence.map(NO_DIGITS) & v.sentence.map(NO_ACRONYMS) & v.sentence.map(NO_SYMBOLS) & ~v.speaker_id.isin(set(filter(None, args.exclude_speakers.split(","))))]   # references come first; the sampler avoids them
    for r in v.itertuples():
        cands.append(dict(dataset="Common Voice", source=f"Common Voice: {r.source}", speaker_id=r.speaker_id, gender=r.gender, age=r.age if isinstance(r.age, str) else "",
                          text=r.sentence.strip(), duration_s=r.duration_s, orig_id=r.path, audio=f"{d}/clips/{r.path}"))
if lang in WAXAL:
    t = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/WaxalNLP/data/TTS/{WAXAL[lang]}/*test*.parquet"))], ignore_index=True)
    t["speaker_id"] = "waxal-" + t.speaker_id.astype(str); t["duration_s"] = [sf.info(io.BytesIO(a["bytes"])).duration for a in t.audio]
    t = t[(~t.speaker_id.isin(gt_speakers) | args.waxal_allow_gt) & ~t.text.isin(gt_texts) & t.duration_s.between(args.min_s, args.max_s) & (t.text.str.strip().str.len() >= MIN_CHARS)]
    for r in t.itertuples():
        cands.append(dict(dataset="WaxalNLP", source=f"WaxalNLP {WAXAL[lang]}_tts test", speaker_id=r.speaker_id, gender=r.gender.lower(), age="",
                          text=r.text.strip(), duration_s=r.duration_s, orig_id=r.audio["path"], audio=r.audio["bytes"]))
cands = pd.DataFrame(cands)
cands["_key"] = cands.orig_id.map(lambda x: hashlib.md5(f"{args.seed}:{x}".encode()).hexdigest()); cands = cands.sort_values("_key")
print(f"{lang}: {len(cands)} candidate clips from {cands.speaker_id.nunique()} speakers "
      f"({cands.groupby('dataset').speaker_id.nunique().to_dict()} speakers; gender {cands.gender.value_counts().to_dict()})")


def asr(rows):
    """CER (diacritics stripped) of the transcript vs the sentence, in the ASR env; returns {orig_id: (hyp, cer, cer_plain)}."""
    tmp = tempfile.mkdtemp(prefix="refasr_", dir="/network/scratch/g/guzmand/tmp"); recs = []
    for i, r in enumerate(rows):
        path = r["audio"] if isinstance(r["audio"], str) else f"{tmp}/{i}.wav"
        if not isinstance(r["audio"], str): open(path, "wb").write(r["audio"])
        recs.append(dict(orig_id=r["orig_id"], path=path, text=r["text"]))
    pd.DataFrame(recs).to_csv(f"{tmp}/in.csv", index=False)
    subprocess.run([ASR_PYTHON, f"{REPO}/src/asr_check.py", "--language", lang, "--in", f"{tmp}/in.csv", "--out", f"{tmp}/out.csv"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = {r.orig_id: (r.asr_hyp, float(r.cer), float(r.cer_plain)) for r in pd.read_csv(f"{tmp}/out.csv", dtype=str).itertuples()}
    import shutil; shutil.rmtree(tmp, ignore_errors=True); return res


def pick(pool, n, chosen, label):
    """Fill up to n slots from pool (distinct speakers, cleanest clip per speaker, ASR-verified), relaxing the gates step by step."""
    for level, relax in enumerate(RELAX if args.relax else RELAX[:1]):
        if len(chosen) >= n: break
        gates = {**STRICT, **relax}; good = []                      # measure lazily until enough candidates pass, best clip per speaker
        taken = {c["speaker_id"] for c in chosen}                    # one clip per speaker: never re-pick a speaker chosen at a stricter level
        for r in pool[~pool.speaker_id.isin(taken)].itertuples():
            m = metrics(r.dataset, r.orig_id, r.audio)
            if passes(m, gates)[0]: good.append({**r._asdict(), **m})
            if len({g["speaker_id"] for g in good}) >= CANDIDATES_PER_GENDER: break
        if not good: continue
        best = pd.DataFrame(good).sort_values("dnsmos_ovrl", ascending=False).drop_duplicates("speaker_id")   # cleanest clip of each speaker
        res = asr(best.to_dict("records"))
        for r in best.itertuples():
            hyp, c, cp = res[r.orig_id]
            if cp <= MAX_CER and len(chosen) < n: chosen.append({**r._asdict(), "asr_hyp": hyp, "cer": c, "cer_plain": cp, "relaxed": "; ".join(f"{k}={v}" for k, v in relax.items())})
        if len(chosen) < n: print(f"   {label}: only {len(chosen)} speaker(s) pass at relaxation level {level}, relaxing further")
    return chosen


picked = []
trimmed_extra = []
if args.waxal_trimmed_extra:                                        # extra studio reference, silence trimmed, then gated and ASR-checked
    g = args.waxal_trimmed_extra; pool = cands[(cands.gender == g) & (cands.dataset == "WaxalNLP")]
    good = []
    for r in pool.itertuples():
        trimmed_bytes, before, after = trim_silence(r.audio)
        if not args.min_s <= after <= args.max_s: continue
        m = {**measure(trimmed_bytes), **dn.score(trimmed_bytes)}
        for level, relax in enumerate(RELAX):
            if passes(m, {**STRICT, **relax})[0]: good.append({**r._asdict(), **m, "audio": trimmed_bytes, "trimmed": f"{before} s -> {after} s", "relaxed": "; ".join(f"{k}={v}" for k, v in relax.items())}); break
    if good:
        best = pd.DataFrame(good).sort_values("dnsmos_ovrl", ascending=False).drop_duplicates("speaker_id")
        res = asr(best.to_dict("records"))
        for r in best.itertuples():
            hyp, c, cp = res[r.orig_id]
            if cp <= MAX_CER: trimmed_extra.append({**r._asdict(), "asr_hyp": hyp, "cer": c, "cer_plain": cp}); break
    if not trimmed_extra: print(f"   WARNING: no trimmed WaxalNLP {g} reference passed the gates")
for gender in ["female", "male"]:
    pool = cands[cands.gender == gender]; chosen = []
    if args.waxal_per_gender:                                        # studio references first (for the Common Voice vs WaxalNLP comparison)
        pick(pool[pool.dataset == "WaxalNLP"], args.waxal_per_gender, chosen, f"{gender} WaxalNLP")
    pick(pool[pool.dataset == "Common Voice"], args.per_gender, chosen, f"{gender} Common Voice")
    picked += chosen
    if len(chosen) < args.per_gender: print(f"   WARNING: only {len(chosen)} {gender} reference(s) found for {lang}")

rows = []
for gender in ["female", "male"]:
    for k, r in enumerate([p for p in picked if p["gender"] == gender], 1):
        ref_id = f"{code}-ref-{gender[0]}{k}"; path = f"{out}/{ref_id}.wav"; r.setdefault("trimmed", "")
        y, sr = (sf.read(r["audio"]) if isinstance(r["audio"], str) else sf.read(io.BytesIO(r["audio"]))); sf.write(path, y, sr, subtype="PCM_16")
        rows.append({**{c: r.get(c, "") for c in COLUMNS}, "ref_id": ref_id, "language": lang, "lang_code": code, "duration_s": round(r["duration_s"], 2),
                     "sample_rate": sr, "audio_path": os.path.relpath(path, REPO)})
for r in trimmed_extra:                                             # the extra trimmed studio reference goes last in the rotation
    k = sum(1 for p in picked if p["gender"] == r["gender"]) + 1; ref_id = f"{code}-ref-{r['gender'][0]}{k}-trimmed"; path = f"{out}/{ref_id}.wav"
    open(path, "wb").write(r["audio"]); sr = sf.info(path).samplerate
    rows.append({**{c: r.get(c, "") for c in COLUMNS}, "ref_id": ref_id, "language": lang, "lang_code": code, "duration_s": sf.info(path).duration, "sample_rate": sr, "audio_path": os.path.relpath(path, REPO)})
for f in glob.glob(f"{out}/*.wav"):                                 # drop stale clips from earlier picks
    if os.path.basename(f) not in {f"{r['ref_id']}.wav" for r in rows}: os.remove(f)
meta = pd.DataFrame(rows, columns=COLUMNS); meta.to_csv(f"{out}/metadata.csv", index=False)
pd.set_option("display.width", 250)
print(meta[["ref_id", "dataset", "speaker_id", "gender", "duration_s", "loudness_lufs", "snr_db", "dnsmos_bak", "dnsmos_ovrl", "lead_silence_s", "trail_silence_s", "cer_plain", "relaxed", "trimmed"]].to_string(index=False))
print(f"\n{len(meta)} reference voices -> {out}/metadata.csv")
