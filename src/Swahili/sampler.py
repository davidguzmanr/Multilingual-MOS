"""Ground-truth sampler for Swahili (sw): picks the listening-test sentences with their human recordings from each
dataset (WaxalNLP TTS test split, FLEURS test, Common Voice), copies the audio **unchanged** (original format) to
audios/Swahili/ground-truth/<dataset>/ and writes one metadata.csv per dataset plus a merged one. Deterministic (SEED); list
orig_ids or texts in audios/Swahili/ground-truth/exclude.txt to swap out bad clips without reshuffling the rest.
Quality gates (skip, never modify): level, silence, stationary noise, DNSMOS P.835, ASR text match; thresholds per language.
    python "src/Swahili/sampler.py"
"""
import glob, hashlib, io, os, shutil, subprocess, sys, tempfile, unicodedata
import pandas as pd, soundfile as sf

LANG, CODE = "Swahili", "sw"
REPO = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS"
OUT = f"{REPO}/audios/{LANG}/ground-truth"
SEED = 0
MIN_S, MAX_S = 5.0, 20.0                      # clip duration in seconds
QUALITY = dict(min_loudness_lufs=-32.0, min_peak_dbfs=-14.0,     # level: FLEURS sw is very quiet (median -41 LUFS); -28 leaves none
               min_snr_db=22.0, max_noise_floor_dbfs=-48.0,      # frame-RMS noise floor
               max_edge_silence_s=1.0, max_total_edge_s=1.5,     # leading / trailing silence
               min_speech_ratio=0.45, max_clip_ratio=0.001,
               max_band_floor_db=-35.0,                          # stationary background per band vs speech (src/audio_quality.py)
               min_dnsmos_bak=3.8, min_dnsmos_ovrl=3.0,          # DNSMOS P.835 (src/dnsmos.py); None = check disabled
               max_cer_plain=0.30)                               # ASR text match, omniASR_LLM_1B_v2, CER with diacritics stripped
ASR_PYTHON = "/home/mila/g/guzmand/scratch/.conda/envs/Multilingual-MOS-asr/bin/python"
QUALITY_OVERRIDES = {"FLEURS": dict(min_loudness_lufs=-34.0, min_peak_dbfs=-16.0, max_band_floor_db=-30.0)}   # FLEURS sw is quiet
# (median -41 LUFS) with a raised low-band floor; with the language gates only 9 clips pass, and listening found them clean, so
# the level and stationary-noise rules are loosened for this dataset only (37 candidates); DNSMOS, silence and ASR gates unchanged
N_WAXAL, N_FLEURS, N_COMMON_VOICE = 12, 11, 11   # 34 sentences for 9 systems
WAXAL_LANG, WAXAL_SPEAKERS_PER_GENDER, WAXAL_MIN_CHARS = "swa", 2, 25   # test: 199 clips, 7 studio speakers (4 f / 3 m), 103 in 5-20 s
FLEURS_CONFIG = "sw_ke"                                  # both genders present -> balanced
CV_DOMAIN = {"batch-1-19-11-2021-contributors-set": "news"}      # East African news only: the everyday set (batch-3) has 68
CV_SPEAKERS_PER_GENDER = 3                                       # gender-tagged clips in 5-20 s and none pass the DNSMOS gate

def text_ok(text):
    if any(ch.isdigit() for ch in text): return False          # no digits: every TTS reads numbers differently
    return True                                          # no text filter for Swahili
COLUMNS = ["clip_id", "language", "lang_code", "dataset", "domain", "source", "text", "speaker_id", "gender", "age",
           "duration_s", "sample_rate", "orig_id", "licence", "audio_path"]
LICENCE = {"WaxalNLP": "CC BY-SA 4.0 (google/WaxalNLP)", "FLEURS": "CC BY 4.0", "Common Voice": "CC0 (MDC terms: no re-hosting)"}
DATA = f"{REPO}/data"
sys.path.insert(0, f"{REPO}/src"); from audio_quality import measure, passes, METRIC_COLUMNS, DNSMOS_COLUMNS   # noqa: E402
from dnsmos import DNSMOS   # noqa: E402
ASR_COLUMNS = ["asr_hyp", "wer", "cer", "cer_plain"]
_dnsmos = DNSMOS()
CACHE = f"{DATA}/quality-cache/{CODE}.csv"                        # measured clips (keyed by dataset + orig_id), reused across runs
_cache = pd.read_csv(CACHE, dtype={"key": str}).set_index("key").to_dict("index") if os.path.exists(CACHE) else {}
_rejected = []


def quality(row, audio_of, dataset):
    """Metrics for one candidate (cached). Returns the metrics dict plus _ok / _why."""
    key = f"{dataset}|{row.orig_id}"
    if key not in _cache or any(c not in _cache[key] for c in METRIC_COLUMNS + DNSMOS_COLUMNS):
        src = audio_of(row); m = {**measure(src), **_dnsmos.score(src)}
        _cache[key] = {c: m[c] for c in METRIC_COLUMNS + DNSMOS_COLUMNS}
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        pd.DataFrame.from_dict(_cache, orient="index").rename_axis("key").to_csv(CACHE)
    m = dict(_cache[key]); ok, why = passes(m, {**QUALITY, **QUALITY_OVERRIDES.get(dataset, {})})   # thresholds re-applied every run, only metrics cached
    m["ok"], m["why"] = ok, "; ".join(why)
    if not ok: _rejected.append((dataset, row.orig_id, m["why"]))
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
    if len(df) == 0: return pd.DataFrame(columns=list(df.columns) + METRIC_COLUMNS + DNSMOS_COLUMNS)
    df = shuffled(df)
    queue = [g for _, g in sorted(df.groupby(by), key=lambda kv: (-len(kv[1]), kv[0]))]
    active = queue[:k]; queue = queue[k:]                     # k speakers in rotation; a speaker whose clips all fail the
    picked, i = [], 0                                         # gates is replaced by the next most-recorded speaker
    while len(picked) < n and active:
        j = i % len(active); g = active[j]
        if len(g) == 0:
            active.pop(j)
            if queue: active.insert(j, queue.pop(0))
            continue
        row = g.iloc[0]; active[j] = g.iloc[1:]
        m = quality(row, audio_of, dataset)
        if m["ok"]: picked.append(pd.concat([row, pd.Series({c: m[c] for c in METRIC_COLUMNS + DNSMOS_COLUMNS})])); i += 1
        # a rejected clip gives the same speaker the next turn: i is not advanced
    if len(picked) < n and not quiet: print(f"   WARNING: only {len(picked)} of {n} available")
    return pd.DataFrame(picked, columns=list(df.columns) + METRIC_COLUMNS + DNSMOS_COLUMNS)


def gender_balanced(df, n, by, k, audio_of, dataset):
    """Half female / half male when both genders have enough clips; whatever is missing is filled from the rest of the pool."""
    f = spread(df[df.gender == "female"], n // 2, by, k, audio_of, dataset, quiet=True)
    m = spread(df[df.gender == "male"], n - n // 2, by, k, audio_of, dataset, quiet=True)
    picked = pd.concat([x for x in (f, m) if len(x)], ignore_index=True) if len(f) + len(m) else f
    if len(picked) < n:                                                        # e.g. FLEURS ha (female only), yo (male only)
        rest = df[~df.orig_id.isin(picked.orig_id)] if len(picked) else df
        picked = pd.concat([picked, spread(rest, n - len(picked), by, k, audio_of, dataset)], ignore_index=True)
    return picked


ASR_CACHE = f"{DATA}/quality-cache/{CODE}-asr.csv"
_asr = pd.read_csv(ASR_CACHE, dtype={"key": str}).set_index("key").to_dict("index") if os.path.exists(ASR_CACHE) else {}
_asr_bad = set()                                                   # orig_ids rejected by the text match in this run


def asr_verify(picks, audio_of, dataset):
    """Transcribe the picked clips (batched, in the Multilingual-MOS-asr env) and cache hyp / wer / cer / cer_plain."""
    todo = [r for r in picks.itertuples() if f"{dataset}|{r.orig_id}" not in _asr]
    if todo:
        tmp = tempfile.mkdtemp(prefix="asr_", dir="/network/scratch/g/guzmand/tmp"); rows = []
        for i, r in enumerate(todo):
            src = audio_of(r)
            if isinstance(src, bytes): path = f"{tmp}/{i}.wav"; open(path, "wb").write(src)
            else: path = src
            rows.append(dict(key=f"{dataset}|{r.orig_id}", path=path, text=r.text))
        pd.DataFrame(rows).to_csv(f"{tmp}/in.csv", index=False)
        subprocess.run([ASR_PYTHON, f"{REPO}/src/asr_check.py", "--language", LANG, "--in", f"{tmp}/in.csv", "--out", f"{tmp}/out.csv"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for r in pd.read_csv(f"{tmp}/out.csv", dtype=str).itertuples():
            _asr[r.key] = dict(asr_hyp=r.asr_hyp, wer=float(r.wer), cer=float(r.cer), cer_plain=float(r.cer_plain))
        pd.DataFrame.from_dict(_asr, orient="index").rename_axis("key").to_csv(ASR_CACHE); shutil.rmtree(tmp, ignore_errors=True)
    return {r.orig_id: _asr[f"{dataset}|{r.orig_id}"] for r in picks.itertuples()}


def with_asr(select, audio_of, dataset, rounds=6):
    """select() -> picks passing the signal/DNSMOS gates. Verify the text with ASR; clips above max_cer_plain are
    rejected and select() is re-run (it excludes _asr_bad) until every pick is verified."""
    for _ in range(rounds):
        picks = select()
        if QUALITY.get("max_cer_plain") is None or len(picks) == 0: break
        res = asr_verify(picks, audio_of, dataset)
        bad = [o for o in picks.orig_id if res[o]["cer_plain"] > QUALITY["max_cer_plain"]]
        for o in bad: _asr_bad.add(o); _rejected.append((dataset, o, f"text mismatch: CER {res[o]['cer_plain']:.2f} ({res[o]['asr_hyp'][:50]})"))
        if not bad: break
    else: print(f"   WARNING: ASR verification did not converge for {dataset}")
    if QUALITY.get("max_cer_plain") is not None and len(picks):
        res = asr_verify(picks, audio_of, dataset)
        for c in ASR_COLUMNS: picks[c] = [res[o][c] for o in picks.orig_id]
    return picks


def write(dataset, picks, audio_of):
    """picks: DataFrame with the COLUMNS except clip_id/audio_path/sample_rate; audio_of(row) -> bytes or source path."""
    d = f"{OUT}/{dataset}"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    tag = {"WaxalNLP": "wx", "FLEURS": "fl", "Common Voice": "cv"}[dataset]
    rows = []
    for i, (_, r) in enumerate(picks.iterrows(), 1):
        a = audio_of(r)
        ext = os.path.splitext(r.orig_id.split("/")[-1])[1] or ".wav"
        clip_id = f"{CODE}-{tag}-{i:03d}"; path = f"{d}/{clip_id}{ext}"
        if isinstance(a, bytes): open(path, "wb").write(a)
        else: shutil.copyfile(a, path)
        row = {c: r.get(c, "") for c in COLUMNS + METRIC_COLUMNS + DNSMOS_COLUMNS + ASR_COLUMNS}
        row.update(clip_id=clip_id, language=LANG, lang_code=CODE, dataset=dataset, licence=LICENCE[dataset],
                   sample_rate=sf.info(path).samplerate, audio_path=os.path.relpath(path, REPO), duration_s=round(float(r.duration_s), 2))
        rows.append(row)
    out = pd.DataFrame(rows, columns=COLUMNS + METRIC_COLUMNS + DNSMOS_COLUMNS + ASR_COLUMNS); out.to_csv(f"{d}/metadata.csv", index=False)
    print(f"== {dataset}: {len(out)} clips | speakers {out.speaker_id.value_counts().to_dict()} | gender {out.gender.value_counts().to_dict()} "
          f"| {out.duration_s.mean():.1f} s mean | {out.loudness_lufs.median():.1f} LUFS median | BAK median {out.dnsmos_bak.median():.2f}"
          + (f" | CER(plain) max {pd.to_numeric(out.cer_plain).max():.2f}" if "cer_plain" in out and out.cer_plain.notna().any() else ""))
    return out


def sample_waxal(n):
    """google/WaxalNLP <lang>_tts **test** split (studio read speech, phonetically balanced scripts, 48 kHz mp3, speaker id
    and gender per clip), downloaded with snapshot_download(allow_patterns=["data/TTS/<lang>/*test*"]) to data/WaxalNLP/."""
    files = sorted(glob.glob(f"{DATA}/WaxalNLP/data/TTS/{WAXAL_LANG}/*test*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.assign(text=df.text.str.strip(), duration_s=[sf.info(io.BytesIO(a["bytes"])).duration for a in df.audio],
                   speaker_id="waxal-" + df.speaker_id.astype(str), gender=df.gender.str.lower(), age="", domain="read script",
                   source=f"WaxalNLP {WAXAL_LANG}_tts test", orig_id=df.audio.map(lambda a: a["path"]))
    df = df[df.duration_s.between(MIN_S, MAX_S) & df.text.map(text_ok) & (df.text.str.len() >= WAXAL_MIN_CHARS)].drop_duplicates("text")
    df = df[~df.orig_id.isin(excluded()) & ~df.text.isin(excluded())]
    audio_of = lambda r: r.audio["bytes"]
    picks = with_asr(lambda: gender_balanced(df[~df.orig_id.isin(_asr_bad)], n, "speaker_id", WAXAL_SPEAKERS_PER_GENDER, audio_of, "WaxalNLP"), audio_of, "WaxalNLP")
    return write("WaxalNLP", picks, audio_of)


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
    picks = with_asr(lambda: gender_balanced(df[~df.orig_id.isin(_asr_bad)], n, "gender", 2, audio_of, "FLEURS"), audio_of, "FLEURS")   # no speaker ids: gender only
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
    refs_path = f"{OUT}/../reference-voices/metadata.csv"                       # references are picked first (src/synthesis/references.py):
    if os.path.exists(refs_path):                                                # their speakers and sentences stay out of the ground truth
        refs = pd.read_csv(refs_path, dtype=str); df = df[~df.speaker_id.isin(set(refs.speaker_id)) & ~df.text.isin(set(refs.text))]
    df = shuffled(df).drop_duplicates("sentence_id")                             # one clip per sentence
    audio_of = lambda r: f"{d}/clips/{r.path}"
    def select():                                                                  # n split evenly over the sources
        out = []
        for i, src in enumerate(CV_DOMAIN):
            n_src = n // len(CV_DOMAIN) + (i < n % len(CV_DOMAIN))
            k = CV_SPEAKERS_PER_GENDER[src] if isinstance(CV_SPEAKERS_PER_GENDER, dict) else CV_SPEAKERS_PER_GENDER
            pool = df[(df.source == "Common Voice: " + src) & ~df.orig_id.isin(_asr_bad)]
            out.append(gender_balanced(pool, n_src, "speaker_id", k, audio_of, "Common Voice"))
        return pd.concat(out, ignore_index=True)
    picks = with_asr(select, audio_of, "Common Voice")
    return write("Common Voice", picks, audio_of)


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = [sample_waxal(N_WAXAL)] if N_WAXAL else []
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
