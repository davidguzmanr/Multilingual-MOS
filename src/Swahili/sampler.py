"""Ground-truth sampler for Swahili (sw): picks the listening-test sentences with their human recordings from each
dataset, copies the audio **unchanged** (original format) to audios/Swahili/ground-truth/<dataset>/ and writes one
metadata.csv per dataset plus a merged one. Deterministic (SEED); list orig_ids or texts in
audios/Swahili/ground-truth/exclude.txt to swap out bad clips without reshuffling the rest.
    python "src/Swahili/sampler.py"
"""
import glob, hashlib, os, shutil, sys
import pandas as pd, soundfile as sf

LANG, CODE = "Swahili", "sw"
REPO = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS"
OUT = f"{REPO}/audios/{LANG}/ground-truth"
SEED = 0
MIN_S, MAX_S = 5.0, 20.0                      # clip duration in seconds
QUALITY = dict(min_loudness_lufs=-38.0, min_peak_dbfs=-20.0,     # skip (never modify) clips that are too quiet,
               min_snr_db=22.0, max_noise_floor_dbfs=-48.0,      # noisy, or padded with silence; calibrated
               max_edge_silence_s=1.0, max_total_edge_s=1.5,     # 2026-11-21 on the ha/yo/sw pools, see
               min_speech_ratio=0.45, max_clip_ratio=0.001)      # src/audio_quality.py
N_OPEN_BIBLE, N_FLEURS, N_COMMON_VOICE = 12, 11, 11   # 34 sentences for 9 systems
OPEN_BIBLE_NAME, OB_SPEAKERS = "Swahili", 4           # 7 narrators in test -> the 4 with most verses
FLEURS_CONFIG = "sw_ke"                                  # both genders present -> balanced
CV_DOMAIN = {"batch-1-19-11-2021-contributors-set": "news",      # East African news (6 of the 11 CV clips)
             "batch-3-17-08-2022-contributors-set": "everyday"}  # short everyday sentences (5); dialect sets excluded
CV_SPEAKERS_PER_GENDER = {"batch-1-19-11-2021-contributors-set": 2,   # no speaker records both sources, so keep the
                          "batch-3-17-08-2022-contributors-set": 1}   # voice count down: 4 news + 2 everyday speakers

def text_ok(text):
    return True                                          # no text filter for Swahili

COLUMNS = ["clip_id", "language", "lang_code", "dataset", "domain", "source", "text", "speaker_id", "gender", "age",
           "duration_s", "sample_rate", "orig_id", "licence", "audio_path"]
LICENCE = {"Open Bible": "CC BY-SA 4.0 (Biblica via open.bible)", "FLEURS": "CC BY 4.0", "Common Voice": "CC0 (MDC terms: no re-hosting)"}
DATA = f"{REPO}/data"
sys.path.insert(0, f"{REPO}/src"); from audio_quality import measure, passes, METRIC_COLUMNS   # noqa: E402
CACHE = f"{DATA}/quality-cache/{CODE}.csv"                        # measured clips (keyed by dataset + orig_id), reused across runs
_cache = pd.read_csv(CACHE, dtype={"key": str}).set_index("key").to_dict("index") if os.path.exists(CACHE) else {}
_rejected = []


def quality(row, audio_of, dataset):
    """Metrics for one candidate (cached). Returns the metrics dict plus _ok / _why."""
    key = f"{dataset}|{row.orig_id}"
    if key not in _cache:
        m = measure(audio_of(row)); ok, why = passes(m, QUALITY)
        _cache[key] = {**{c: m[c] for c in METRIC_COLUMNS}, "ok": ok, "why": "; ".join(why)}
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        pd.DataFrame.from_dict(_cache, orient="index").rename_axis("key").to_csv(CACHE)
    m = _cache[key]
    if not m["ok"]: _rejected.append((dataset, row.orig_id, m["why"]))
    return m


def excluded():
    """orig_ids or exact texts listed in exclude.txt (one per line) are skipped; re-running then fills the gap."""
    p = f"{OUT}/exclude.txt"
    return set(l.strip() for l in open(p) if l.strip() and not l.startswith("#")) if os.path.exists(p) else set()


def shuffled(df):
    """Deterministic order from hash(SEED, orig_id): removing a clip (exclude.txt) leaves every other clip's rank unchanged."""
    key = df.orig_id.map(lambda x: hashlib.md5(f"{SEED}:{x}".encode()).hexdigest())
    return df.assign(_key=key).sort_values("_key").drop(columns="_key")


def spread(df, n, by, k, audio_of, dataset, quiet=False):
    """Take n rows spread over up to k values of `by` (the ones with most rows), round-robin, in `shuffled` order.
    Each candidate is measured lazily; clips failing QUALITY are skipped and the next one of that speaker is tried."""
    if len(df) == 0: return pd.DataFrame(columns=list(df.columns) + METRIC_COLUMNS)
    df = shuffled(df)
    groups = [g for _, g in sorted(df.groupby(by), key=lambda kv: (-len(kv[1]), kv[0]))][:k]
    picked, i = [], 0
    while len(picked) < n and any(len(g) for g in groups):
        g = groups[i % len(groups)]
        if len(g):
            row = g.iloc[0]; groups[i % len(groups)] = g.iloc[1:]
            m = quality(row, audio_of, dataset)
            if m["ok"]: picked.append(pd.concat([row, pd.Series({c: m[c] for c in METRIC_COLUMNS})]))
            else: continue                                   # same speaker gets the next turn: i is not advanced
        i += 1
    if len(picked) < n and not quiet: print(f"   WARNING: only {len(picked)} of {n} available")
    return pd.DataFrame(picked, columns=list(df.columns) + METRIC_COLUMNS)


def gender_balanced(df, n, by, k, audio_of, dataset):
    """Half female / half male when both genders have enough clips; whatever is missing is filled from the rest of the pool."""
    f = spread(df[df.gender == "female"], n // 2, by, k, audio_of, dataset, quiet=True)
    m = spread(df[df.gender == "male"], n - n // 2, by, k, audio_of, dataset, quiet=True)
    picked = pd.concat([x for x in (f, m) if len(x)], ignore_index=True) if len(f) + len(m) else f
    if len(picked) < n:                                                        # e.g. FLEURS ha (female only), yo (male only)
        rest = df[~df.orig_id.isin(picked.orig_id)] if len(picked) else df
        picked = pd.concat([picked, spread(rest, n - len(picked), by, k, audio_of, dataset)], ignore_index=True)
    return picked


def write(dataset, picks, audio_of):
    """picks: DataFrame with the COLUMNS except clip_id/audio_path/sample_rate; audio_of(row) -> bytes or source path."""
    d = f"{OUT}/{dataset}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    tag = {"Open Bible": "ob", "FLEURS": "fl", "Common Voice": "cv"}[dataset]
    rows = []
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        a = audio_of(r)
        ext = os.path.splitext(r.orig_id.split("/")[-1])[1] or ".wav"
        clip_id = f"{CODE}-{tag}-{i:03d}"; path = f"{d}/{clip_id}{ext}"
        if isinstance(a, bytes): open(path, "wb").write(a)
        else: shutil.copyfile(a, path)
        row = {c: r.get(c, "") for c in COLUMNS + METRIC_COLUMNS}
        row.update(clip_id=clip_id, language=LANG, lang_code=CODE, dataset=dataset, licence=LICENCE[dataset],
                   sample_rate=sf.info(path).samplerate, audio_path=os.path.relpath(path, REPO), duration_s=round(float(r.duration_s), 2))
        rows.append(row)
    out = pd.DataFrame(rows, columns=COLUMNS + METRIC_COLUMNS); out.to_csv(f"{d}/metadata.csv", index=False)
    print(f"== {dataset}: {len(out)} clips | speakers {out.speaker_id.value_counts().to_dict()} | gender {out.gender.value_counts().to_dict()} "
          f"| {out.duration_s.mean():.1f} s mean")
    return out


def sample_open_bible(n):
    files = sorted(glob.glob(f"{DATA}/Open-Bible/{OPEN_BIBLE_NAME}/test*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.assign(text=df.text.str.strip(), duration_s=df.duration_seconds.astype(float), speaker_id=df.speaker_id,
                   gender="unknown", age="", domain="scripture", source="Open Bible " + df.testament.astype(str),
                   orig_id=df.book + " " + df.chapter + ":" + df.verse + "/" + df.audio.map(lambda a: a["path"]))
    df = df[df.duration_s.between(MIN_S, MAX_S)].drop_duplicates("text")
    df = df[~df.orig_id.isin(excluded()) & ~df.text.isin(excluded())]
    audio_of = lambda r: r.audio["bytes"]
    picks = spread(df, n, "speaker_id", OB_SPEAKERS, audio_of, "Open Bible")
    return write("Open Bible", picks, audio_of)


def sample_fleurs(n):
    d = f"{DATA}/FLEURS/{FLEURS_CONFIG}"
    df = pd.read_csv(f"{d}/test.tsv", sep="\t", quoting=3, dtype=str, header=None,
                     names=["sentence_id", "path", "raw", "transcription", "chars", "num_samples", "gender"])
    df = df.assign(text=df.raw.str.replace(r'^"(.*)"$', r"\1", regex=True).str.replace('""', '"').str.strip(),
                   duration_s=df.num_samples.astype(int) / 16000, speaker_id="unknown", gender=df.gender.str.lower(), age="",
                   domain="encyclopedic", source="FLORES-101 (Wikipedia)", orig_id=df.sentence_id + "/" + df.path)
    df = df[df.duration_s.between(MIN_S, MAX_S) & df.text.map(text_ok)]
    df = df[~df.orig_id.isin(excluded()) & ~df.text.isin(excluded())]
    df = shuffled(df).drop_duplicates("sentence_id")                             # one reading per sentence
    audio_of = lambda r: f"{d}/test/{r.path}"
    picks = gender_balanced(df, n, "gender", 2, audio_of, "FLEURS")                 # no speaker ids: balance gender only
    return write("FLEURS", picks, audio_of)


def sample_common_voice(n):
    d = glob.glob(f"{DATA}/Common-Voice/{CODE}/*/cv-corpus-*/{CODE}")[0]
    v = pd.read_csv(f"{d}/validated.tsv", sep="\t", quoting=3, dtype=str, on_bad_lines="skip")
    s = pd.read_csv(f"{d}/validated_sentences.tsv", sep="\t", quoting=3, dtype=str, on_bad_lines="skip", usecols=["sentence_id", "source"])
    dur = pd.read_csv(f"{d}/clip_durations.tsv", sep="\t", dtype=str, names=["path", "ms"], header=0)
    df = v.merge(s, on="sentence_id", how="left").merge(dur, on="path", how="left")
    df = df.assign(text=df.sentence.str.strip(), duration_s=df.ms.astype(float) / 1000, speaker_id=df.client_id.str[:12],
                   gender=df.gender.map({"female_feminine": "female", "male_masculine": "male"}).fillna("unknown"),
                   age=df.age.fillna(""), domain=df.source.map(CV_DOMAIN), source="Common Voice: " + df.source.fillna(""), orig_id=df.path)
    df = df[df.source.isin("Common Voice: " + pd.Series(list(CV_DOMAIN))) & df.duration_s.between(MIN_S, MAX_S)
            & (df.up_votes.astype(int) >= 2) & (df.down_votes.astype(int) == 0) & df.text.map(text_ok)]
    df = df[~df.orig_id.isin(excluded()) & ~df.text.isin(excluded())]
    df = shuffled(df).drop_duplicates("sentence_id")                             # one clip per sentence
    audio_of = lambda r: f"{d}/clips/{r.path}"
    picks = []                                                                     # n split evenly over the sources
    for i, src in enumerate(CV_DOMAIN):
        n_src = n // len(CV_DOMAIN) + (i < n % len(CV_DOMAIN))
        k = CV_SPEAKERS_PER_GENDER[src] if isinstance(CV_SPEAKERS_PER_GENDER, dict) else CV_SPEAKERS_PER_GENDER
        picks.append(gender_balanced(df[df.source == "Common Voice: " + src], n_src, "speaker_id", k, audio_of, "Common Voice"))
    picks = pd.concat(picks, ignore_index=True)
    return write("Common Voice", picks, audio_of)


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = [sample_open_bible(N_OPEN_BIBLE)] if N_OPEN_BIBLE else []
    parts += [sample_fleurs(N_FLEURS)] if N_FLEURS else []
    parts += [sample_common_voice(N_COMMON_VOICE)] if N_COMMON_VOICE else []
    all_ = pd.concat(parts, ignore_index=True); all_.to_csv(f"{OUT}/metadata.csv", index=False)
    if _rejected:
        rej = pd.DataFrame(_rejected, columns=["dataset", "orig_id", "why"]).drop_duplicates()
        print(f"\nskipped {len(rej)} candidate clips on quality: " + ", ".join(f"{k} {v}" for k, v in rej.dataset.value_counts().items()))
        rej.to_csv(f"{OUT}/quality-rejected.csv", index=False)
    print(f"\n{LANG}: {len(all_)} ground-truth clips -> {OUT}/metadata.csv")
    print(all_.groupby("dataset").agg(clips=("clip_id", "size"), speakers=("speaker_id", "nunique"), mean_s=("duration_s", "mean")).round(1).to_string())


if __name__ == "__main__":
    main()
