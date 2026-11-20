"""Ground-truth sampler for Hausa (ha): picks the listening-test sentences with their human recordings from each
dataset, copies the audio **unchanged** (original format) to audios/Hausa/ground-truth/<dataset>/ and writes one
metadata.csv per dataset plus a merged one. Deterministic (SEED); list orig_ids or texts in
audios/Hausa/ground-truth/exclude.txt to swap out bad clips without reshuffling the rest.
    python "src/Hausa/sampler.py"
"""
import glob, hashlib, os, shutil
import pandas as pd, soundfile as sf

LANG, CODE = "Hausa", "ha"
REPO = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS"
OUT = f"{REPO}/audios/{LANG}/ground-truth"
SEED = 0
MIN_S, MAX_S = 5.0, 20.0                      # only filter: clip duration in seconds
N_OPEN_BIBLE, N_FLEURS, N_COMMON_VOICE = 13, 13, 12   # 38 sentences for 8 systems
OPEN_BIBLE_NAME, OB_SPEAKERS = "Hausa", 2             # test split has 2 narrators -> use both
FLEURS_CONFIG = "ha_ng"                                  # FLEURS test is female-only (1 male clip), no speaker ids
CV_DOMAIN = {"sentence-collector": "conversational"}     # Common Voice sources to use -> domain label
CV_SPEAKERS_PER_GENDER = 3                               # speakers per gender; 2 female / 7 male pass the filters

def text_ok(text):
    return True                                          # no text filter for Hausa

COLUMNS = ["clip_id", "language", "lang_code", "dataset", "domain", "source", "text", "speaker_id", "gender", "age",
           "duration_s", "sample_rate", "orig_id", "licence", "audio_path"]
LICENCE = {"Open Bible": "CC BY-SA 4.0 (Biblica via open.bible)", "FLEURS": "CC BY 4.0", "Common Voice": "CC0 (MDC terms: no re-hosting)"}
DATA = f"{REPO}/data"


def excluded():
    """orig_ids or exact texts listed in exclude.txt (one per line) are skipped; re-running then fills the gap."""
    p = f"{OUT}/exclude.txt"
    return set(l.strip() for l in open(p) if l.strip() and not l.startswith("#")) if os.path.exists(p) else set()


def shuffled(df):
    """Deterministic order from hash(SEED, orig_id): removing a clip (exclude.txt) leaves every other clip's rank unchanged."""
    key = df.orig_id.map(lambda x: hashlib.md5(f"{SEED}:{x}".encode()).hexdigest())
    return df.assign(_key=key).sort_values("_key").drop(columns="_key")


def spread(df, n, by, k, quiet=False):
    """Take n rows spread over up to k values of `by` (the ones with most rows), round-robin, in `shuffled` order."""
    if len(df) == 0: return pd.DataFrame(columns=df.columns)
    df = shuffled(df)
    groups = [g for _, g in sorted(df.groupby(by), key=lambda kv: (-len(kv[1]), kv[0]))][:k]
    picked, i = [], 0
    while len(picked) < n and any(len(g) for g in groups):
        g = groups[i % len(groups)]
        if len(g): picked.append(g.iloc[0]); groups[i % len(groups)] = g.iloc[1:]
        i += 1
    if len(picked) < n and not quiet: print(f"   WARNING: only {len(picked)} of {n} available")
    return pd.DataFrame(picked, columns=df.columns)


def gender_balanced(df, n, by, k):
    """Half female / half male when both genders have enough clips; whatever is missing is filled from the rest of the pool."""
    f = spread(df[df.gender == "female"], n // 2, by, k, quiet=True)
    m = spread(df[df.gender == "male"], n - n // 2, by, k, quiet=True)
    picked = pd.concat([x for x in (f, m) if len(x)], ignore_index=True) if len(f) + len(m) else f
    if len(picked) < n:                                                        # e.g. FLEURS ha (female only), yo (male only)
        rest = df[~df.orig_id.isin(picked.orig_id)] if len(picked) else df
        picked = pd.concat([picked, spread(rest, n - len(picked), by, k)], ignore_index=True)
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
        row = {c: r.get(c, "") for c in COLUMNS}
        row.update(clip_id=clip_id, language=LANG, lang_code=CODE, dataset=dataset, licence=LICENCE[dataset],
                   sample_rate=sf.info(path).samplerate, audio_path=os.path.relpath(path, REPO), duration_s=round(float(r.duration_s), 2))
        rows.append(row)
    out = pd.DataFrame(rows, columns=COLUMNS); out.to_csv(f"{d}/metadata.csv", index=False)
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
    picks = spread(df, n, "speaker_id", OB_SPEAKERS)
    return write("Open Bible", picks, lambda r: r.audio["bytes"])


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
    picks = gender_balanced(df, n, "gender", 2)                                    # no speaker ids: balance gender only
    return write("FLEURS", picks, lambda r: f"{d}/test/{r.path}")


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
    picks = []                                                                     # n split evenly over the sources
    for i, src in enumerate(CV_DOMAIN):
        n_src = n // len(CV_DOMAIN) + (i < n % len(CV_DOMAIN))
        k = CV_SPEAKERS_PER_GENDER[src] if isinstance(CV_SPEAKERS_PER_GENDER, dict) else CV_SPEAKERS_PER_GENDER
        picks.append(gender_balanced(df[df.source == "Common Voice: " + src], n_src, "speaker_id", k))
    picks = pd.concat(picks, ignore_index=True)
    return write("Common Voice", picks, lambda r: f"{d}/clips/{r.path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = [sample_open_bible(N_OPEN_BIBLE)] if N_OPEN_BIBLE else []
    parts += [sample_fleurs(N_FLEURS)] if N_FLEURS else []
    parts += [sample_common_voice(N_COMMON_VOICE)] if N_COMMON_VOICE else []
    all_ = pd.concat(parts, ignore_index=True); all_.to_csv(f"{OUT}/metadata.csv", index=False)
    print(f"\n{LANG}: {len(all_)} ground-truth clips -> {OUT}/metadata.csv")
    print(all_.groupby("dataset").agg(clips=("clip_id", "size"), speakers=("speaker_id", "nunique"), mean_s=("duration_s", "mean")).round(1).to_string())


if __name__ == "__main__":
    main()
