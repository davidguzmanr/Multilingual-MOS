"""Shared helpers for the per-model synthesis scripts in src/synthesis/.

Every model script: reads the sentences from audios/<Language>/ground-truth/metadata.csv, assigns one voice per sentence
with `assign_voices` (same rule for every model, so multi-voice systems share the same gender split), synthesises, and
writes audios/<Language>/<System>/<dataset>/<clip_id>.wav (same dataset subfolders as the ground truth) plus a metadata.csv
per dataset folder and a merged one at the system root, with the common schema below.
"""
import argparse, datetime, json, os, sys, time
import numpy as np, pandas as pd, soundfile as sf

REPO = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS"
LANG_CODES = {"Arabic": "ar", "French": "fr", "German": "de", "Haitian Creole": "ht", "Hausa": "ha", "Hindi": "hi",
              "Indonesian": "id", "Japanese": "ja", "Korean": "ko", "Spanish": "es", "Swahili": "sw", "Turkish": "tr",
              "Vietnamese": "vi", "Yoruba": "yo"}
COLUMNS = ["clip_id", "language", "lang_code", "dataset", "system", "model_id", "text", "voice_id", "voice_gender", "reference_clip",
           "duration_s", "sample_rate", "params", "audio_path", "generated_at"]


def parse_args(description, add=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--language", required=True, choices=sorted(LANG_CODES), help="language name as used in audios/<Language>/")
    p.add_argument("--voices", help="comma-separated voice ids to use instead of the script's default set")
    p.add_argument("--limit", type=int, help="only synthesise the first N sentences (smoke test)")
    p.add_argument("--overwrite", action="store_true", help="regenerate clips that already exist")
    p.add_argument("--dry-run", action="store_true", help="print the sentence/voice assignment and exit")
    if add: add(p)
    return p.parse_args()


def load_sentences(language, limit=None):
    """Ground-truth sentences (one per clip_id), in clip_id order; the text is passed verbatim to the TTS."""
    df = pd.read_csv(f"{REPO}/audios/{language}/ground-truth/metadata.csv", dtype=str).sort_values("clip_id").reset_index(drop=True)
    return df.head(limit) if limit else df


def load_references(language):
    """Reference voices for cloning models: audios/<Language>/reference-voices/metadata.csv (from references.py)."""
    p = f"{REPO}/audios/{language}/reference-voices/metadata.csv"
    if not os.path.exists(p): sys.exit(f"no reference voices for {language}: run src/synthesis/references.py --language '{language}'")
    df = pd.read_csv(p, dtype=str)
    refs = [dict(voice_id=r.ref_id, gender=r.gender, reference_clip=r.orig_id, audio_path=f"{REPO}/{r.audio_path}", text=r.text)   # reference_clip =
            for r in df.itertuples()]                                                                                            # source clip id
    by_gender = [[r for r in refs if r["gender"] == g] for g in ("female", "male")]
    inter = [r for pair in zip(*by_gender) for r in pair]                 # f1, m1, f2, m2 ... so the rotation alternates gender
    return inter + [r for r in refs if r not in inter]


def assign_voices(sentences, voices):
    """Sentence i (in clip_id order) gets voices[i % len(voices)]. Give voices as female, male, female, male so a 50/50
    split falls out for any even count; a single-voice model passes a one-element list."""
    return [voices[i % len(voices)] for i in range(len(sentences))]


def out_dir(language, system):
    d = f"{REPO}/audios/{language}/{system}"; os.makedirs(d, exist_ok=True); return d


def save_wav(path, audio, sample_rate):
    """audio: float array in [-1, 1] (any shape with time last or first) or int16; written as 16-bit PCM wav."""
    a = np.asarray(audio)
    if a.ndim > 1: a = a.squeeze()
    if a.ndim > 1: a = a.T if a.shape[0] < a.shape[1] else a
    if a.dtype.kind == "f": a = np.clip(a, -1, 1)
    sf.write(path, a, sample_rate, subtype="PCM_16")


def run(system, model_id, language, synth, voices, params, limit=None, overwrite=False, dry_run=False):
    """Generic loop. `synth(text, voice) -> (audio, sample_rate)`; `voices` = list of dicts with at least voice_id and gender."""
    sents = load_sentences(language, limit); assigned = assign_voices(sents, voices)
    if dry_run:
        for r, v in zip(sents.itertuples(), assigned): print(f"{r.clip_id}  {v['gender']:7s} {v['voice_id']:24s} {r.text[:70]}")
        print(f"\n{len(sents)} sentences, voices: {[v['voice_id'] for v in voices]}"); return
    d = out_dir(language, system); rows, missing, t0 = [], [], time.time()
    prev = {}                                                   # clip_id -> (model_id, voice_id, reference_clip, text) of clips on disk
    if os.path.exists(f"{d}/metadata.csv"):
        prev = {r.clip_id: (r.model_id, r.voice_id, str(r.reference_clip), r.text) for r in pd.read_csv(f"{d}/metadata.csv", dtype=str, keep_default_na=False).itertuples()}
    for i, (r, v) in enumerate(zip(sents.itertuples(), assigned), 1):
        os.makedirs(f"{d}/{r.dataset}", exist_ok=True); path = f"{d}/{r.dataset}/{r.clip_id}.wav"
        same = prev.get(r.clip_id) == (model_id, v["voice_id"], str(v.get("reference_clip", "")), r.text)   # keep only if model, voice, reference audio and text are unchanged
        if os.path.exists(path) and not overwrite and same:
            info = sf.info(path); dur, sr = info.duration, info.samplerate; status = "kept"
        else:
            out = synth(r.text, v)
            if out is None:                                        # the model script gave up on this clip: record it as missing
                missing.append(r.clip_id); print(f"[{i:3d}/{len(sents)}] {r.clip_id} MISSING", flush=True); continue
            audio, sr = out; save_wav(path, audio, sr); dur = sf.info(path).duration; status = "new"
        rows.append(dict(clip_id=r.clip_id, language=language, lang_code=LANG_CODES[language], dataset=r.dataset, system=system, model_id=model_id,
                         text=r.text, voice_id=v["voice_id"], voice_gender=v.get("gender", ""), reference_clip=v.get("reference_clip", ""),
                         duration_s=round(dur, 2), sample_rate=sr, params=json.dumps(params, ensure_ascii=False),
                         audio_path=os.path.relpath(path, REPO), generated_at=datetime.date.today().isoformat()))
        print(f"[{i:3d}/{len(sents)}] {r.clip_id} {status:4s} {dur:5.1f} s  voice={v['voice_id']}", flush=True)
        if status == "new": pd.DataFrame(rows, columns=COLUMNS).to_csv(f"{d}/metadata.csv", index=False)   # keep metadata in step with the
    meta = pd.DataFrame(rows, columns=COLUMNS); meta.to_csv(f"{d}/metadata.csv", index=False)               # clips if the run stops early (quota)
    for ds, g in meta.groupby("dataset"): g.to_csv(f"{d}/{ds}/metadata.csv", index=False)                  # per-dataset metadata, like the ground truth
    print(f"\n{system} / {language}: {len(meta)} clips, {meta.duration_s.mean():.1f} s mean, {time.time()-t0:.0f} s wall "
          f"| voices {meta.voice_id.value_counts().to_dict()} -> {d}/metadata.csv")
    if missing: print(f"MISSING {len(missing)} clip(s), rerun to fill them: {', '.join(missing)}")
    return meta
