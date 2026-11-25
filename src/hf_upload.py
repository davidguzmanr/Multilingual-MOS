"""Upload the listening-test audio to the Hugging Face dataset repo, keeping the audios/ tree:
    <Language>/<System>/<dataset>/<clip_id>.wav  (+ metadata.csv per folder), ground truth included, reference voices excluded.
Also writes clips.csv (one row per stimulus with its URL, the HumanSignal import file) and README.md (dataset card).
Survey URLs are pinned to the upload commit (resolve/<sha>/...), since clips get regenerated under the same names.

    python src/hf_upload.py --languages Yoruba Hausa Swahili --dry-run      # list what would be uploaded, write clips.csv + card locally
    python src/hf_upload.py --languages Yoruba Hausa Swahili                # upload, print the commit, verify a few URLs
"""
import argparse, glob, os, sys, urllib.parse
import pandas as pd

REPO = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS"; AUDIOS = f"{REPO}/audios"; STAGE = f"{REPO}/data/hf-upload"
HF_REPO = "davidguzmanr/multilingual-mos-listening-test"
EXCLUDE_DIRS = {"reference-voices"}; EXCLUDE_FILES = {"index.html", "exclude.txt", "quality-rejected.csv"}
SOURCES = {"Common Voice": "Mozilla Common Voice 27.0 (Mozilla Data Collective), CC0 1.0; condition: do not attempt to identify the speakers",
           "FLEURS": "FLEURS (Google), CC BY 4.0", "WaxalNLP": "WaxalNLP TTS test split (Google; providers Media Trust for Hausa/Yoruba, Makerere University for Swahili), CC BY-SA 4.0"}


def files_for(language):
    out = []
    for root, dirs, files in os.walk(f"{AUDIOS}/{language}"):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in sorted(files):
            if f in EXCLUDE_FILES or not (f.endswith((".wav", ".mp3", ".flac")) or f == "metadata.csv"): continue
            out.append(os.path.relpath(os.path.join(root, f), AUDIOS))
    return out


def clips_table(languages, sha):
    rows = []
    for L in languages:
        for m in sorted(glob.glob(f"{AUDIOS}/{L}/*/metadata.csv")):
            system = m.split("/")[-2]
            if system in EXCLUDE_DIRS: continue
            d = pd.read_csv(m, dtype=str, keep_default_na=False)
            for r in d.itertuples():
                rel = os.path.relpath(f"{REPO}/{r.audio_path}", AUDIOS)
                rows.append(dict(language=L, system=system, dataset=r.dataset, clip_id=r.clip_id, text=r.text,
                                 voice=getattr(r, "voice_id", getattr(r, "speaker_id", "")), duration_s=r.duration_s, sample_rate=r.sample_rate,
                                 path=rel, url=f"https://huggingface.co/datasets/{HF_REPO}/resolve/{sha}/{urllib.parse.quote(rel)}"))
    return pd.DataFrame(rows)


def card(languages, clips):
    per_lang = clips.groupby("language").agg(sentences=("clip_id", "nunique"), systems=("system", "nunique"), clips=("clip_id", "size"))
    systems = sorted(s for s in clips.system.unique() if s != "ground-truth")
    return f"""---
license: cc-by-nc-4.0
language: [{", ".join(sorted({l[:2].lower() for l in languages}))}]
task_categories: [audio-classification]
tags: [speech, tts, mos, listening-test, naturalness, intelligibility]
pretty_name: Multilingual MOS listening test (audio stimuli)
---

# Multilingual MOS listening test: audio stimuli

Audio used in a multilingual TTS listening study (naturalness and faithfulness-to-text ratings, 1-5) that feeds an open
multilingual MOS corpus. Layout mirrors the project tree: `<Language>/<System>/<dataset>/<clip_id>.wav`, with a
`metadata.csv` in every folder and `clips.csv` at the root (one row per stimulus with its pinned URL).
`<Language>/ground-truth/` holds the human recordings each sentence was taken from; every synthesized clip reads the same
sentence, so systems are fully crossed with sentences.

| Language | Sentences | Systems (incl. ground truth) | Clips |
|---|---|---|---|
""" + "\n".join(f"| {L} | {r.sentences} | {r.systems} | {r.clips} |" for L, r in per_lang.iterrows()) + f"""

Systems: {", ".join(systems)}. Voice-cloning systems (OmniVoice, VoxCPM2, Fish S2-Pro, Higgs TTS 3) share the same Common Voice
reference voices within a language (2 female + 2 male; Hausa 1 + 2; not included here); Gemini and ElevenLabs use four prebuilt voices.

## Ground-truth sources and licences

""" + "\n".join(f"- **{k}**: {v}" for k, v in SOURCES.items()) + """

Ground-truth clips are copied unchanged from their source (original format and sample rate). Sentences were sampled with
quality gates (level, silence, background noise, DNSMOS, ASR text match) and contain no digits, acronyms or symbols.

## Synthesized clips

Generated with the model versions recorded in each `metadata.csv` (`model_id`, `params`). Synthesized audio is released
under CC BY-NC 4.0; the underlying models carry their own licences (e.g. MMS-TTS CC BY-NC 4.0, Fish Audio research licence).

## Citation

Project: Multilingual MOS (Mila / McGill). Citation to be added with the paper.
"""


def main():
    global HF_REPO
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--languages", nargs="+", required=True); ap.add_argument("--repo", default=HF_REPO); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--message", default="Add listening-test audio"); a = ap.parse_args()
    HF_REPO = a.repo
    files = [f for L in a.languages for f in files_for(L)]
    size = sum(os.path.getsize(f"{AUDIOS}/{f}") for f in files)
    print(f"{len(files)} files, {size/1e6:.0f} MB from {a.languages}")
    for L in a.languages:
        fl = [f for f in files if f.startswith(L + "/")]
        print(f"  {L}: {len(fl)} files; folders: {sorted({f.split('/')[1] for f in fl})}")
    os.makedirs(STAGE, exist_ok=True)
    clips = clips_table(a.languages, "main"); clips.to_csv(f"{STAGE}/clips.csv", index=False); open(f"{STAGE}/README.md", "w").write(card(a.languages, clips))
    print(f"clips.csv: {len(clips)} stimuli; staged clips.csv and README.md in {STAGE}")
    if a.dry_run: print("dry run: nothing uploaded"); return
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(); api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True, private=False)
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=f"{AUDIOS}/{f}") for f in files]
    ops.append(CommitOperationAdd(path_in_repo="clips.csv", path_or_fileobj=f"{STAGE}/clips.csv"))
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=f"{STAGE}/README.md"))
    info = api.create_commit(repo_id=HF_REPO, repo_type="dataset", operations=ops, commit_message=a.message)
    sha = info.oid; print(f"uploaded: {info.commit_url}")
    clips = clips_table(a.languages, sha); clips.to_csv(f"{STAGE}/clips.csv", index=False)                     # pin the URLs to this commit
    api.upload_file(path_or_fileobj=f"{STAGE}/clips.csv", path_in_repo="clips.csv", repo_id=HF_REPO, repo_type="dataset", commit_message=f"clips.csv pinned to {sha[:8]}")
    clips.to_csv(f"{STAGE}/clips_{sha[:8]}.csv", index=False)
    import requests
    for url in clips.sample(min(4, len(clips)), random_state=0).url:
        r = requests.get(url, headers={"Range": "bytes=0-99"}, timeout=60, allow_redirects=True)
        print(f"  {r.status_code} {r.headers.get('content-type')} accept-ranges={r.headers.get('accept-ranges')} {url.split('/resolve/')[1][:70]}")
    print(f"commit {sha}; URL template: https://huggingface.co/datasets/{HF_REPO}/resolve/{sha}/<Language>/<System>/<dataset>/<clip_id>.wav")


if __name__ == "__main__":
    main()
