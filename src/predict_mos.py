"""
Score a split's clips with UTMOS22 and/or UTMOSv2.

UTMOS22 uses the pretrained main-track strong learner from the official
HuggingFace demo (``checkpoints/UTMOS22/main``).  Note that checkpoint is the
*w/o phoneme encoder* variant -- its feature extractors are the wav2vec2 SSL
model and a domain embedding, nothing else -- so no phoneme transcriptions are
needed.  It is conditioned on ``domain=0`` (BVCC main) and ``judge_id=288`` (the
main-track mean listener), exactly as the demo does.

UTMOSv2 scores already exist as ``tts_utmos`` in
``data/audios/{corpus}/{system}/{language}/results.csv``.  Pass
``--utmosv2 rerun`` to recompute them here instead, or ``--utmosv2 verify`` to
recompute a random subsample and check it against the stored values.

Examples
--------
    python src/predict_mos.py --split test --models utmos22
    python src/predict_mos.py --split test --models utmosv2 --utmosv2 verify --verify-n 60
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_ROOT = REPO_ROOT / "data" / "splits"
AUDIO_DIR = REPO_ROOT / "data" / "audios"
UTMOS22_DIR = REPO_ROOT / "checkpoints" / "UTMOS22" / "main"
UTMOS22_CKPT = "epoch=3-step=7459.ckpt"


@contextlib.contextmanager
def working_dir(path: Path):
    """UTMOS22's demo module hardcodes a relative path to wav2vec_small.pt."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# ── UTMOS22 ──────────────────────────────────────────────────────────────────

def load_utmos22(device: str):
    sys.path.insert(0, str(UTMOS22_DIR))
    with working_dir(UTMOS22_DIR):
        from score import Score

        return Score(ckpt_path=UTMOS22_CKPT, input_sample_rate=16000, device=device)


def predict_utmos22(clips: pd.DataFrame, device: str) -> list[float]:
    scorer = load_utmos22(device)
    preds = []
    for row in tqdm(clips.itertuples(), total=len(clips), desc="UTMOS22"):
        wav, sr = torchaudio.load(row.audio_path)
        if wav.shape[0] > 1:  # force mono
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        with torch.no_grad():
            preds.append(float(scorer.score(wav.squeeze(0).to(device))))
    return preds


# ── UTMOSv2 ──────────────────────────────────────────────────────────────────

def load_stored_utmosv2() -> pd.DataFrame:
    frames = []
    for path in glob.glob(str(AUDIO_DIR / "*" / "*" / "*" / "results.csv")):
        corpus, system, language = path.split(os.sep)[-4:-1]
        df = pd.read_csv(path)
        df["dataset"], df["system"], df["language"] = corpus, system, language
        frames.append(df)
    stored = pd.concat(frames, ignore_index=True)
    return stored[["dataset", "system", "language", "filename", "tts_utmos"]]


def predict_utmosv2(clips: pd.DataFrame) -> list[float]:
    """Score every clip in one batched pass.

    ``predict(input_path=...)`` rebuilds a dataloader per call, so instead we
    stage the clips as a directory of symlinks named by ``utt_id`` (unique per
    clip -- raw filenames collide across languages and systems) and hand
    UTMOSv2 the whole directory at once.
    """
    import tempfile

    import utmosv2

    model = utmosv2.create_model(pretrained=True)
    with tempfile.TemporaryDirectory(prefix="utmosv2-batch-") as tmp:
        staging = Path(tmp)
        for row in clips.itertuples():
            (staging / f"{row.utt_id}.wav").symlink_to(row.audio_path)
        results = model.predict(input_dir=str(staging))

    by_utt = {Path(r["file_path"]).stem: float(r["predicted_mos"]) for r in results}
    missing = set(clips["utt_id"]) - set(by_utt)
    if missing:
        raise RuntimeError(f"UTMOSv2 returned no score for {len(missing)} clips")
    return [by_utt[u] for u in clips["utt_id"]]


# ── Main ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tag", default="system-corpus", help="Split subdirectory.")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--models", nargs="+", default=["utmos22", "utmosv2"])
    p.add_argument(
        "--utmosv2",
        default="stored",
        choices=["stored", "rerun", "verify"],
        help="Use stored tts_utmos, recompute everything, or spot-check a subsample.",
    )
    p.add_argument("--verify-n", type=int, default=60)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="Output CSV (default: <split dir>/predictions.csv)")
    return p.parse_args()


def main():
    args = get_args()
    split_dir = SPLIT_ROOT / args.tag
    clips = pd.read_csv(split_dir / "clips.csv")
    if args.split != "all":
        clips = clips[clips["split"] == args.split].reset_index(drop=True)
    print(f"{len(clips)} clips in split '{args.split}' of '{args.tag}'")

    out_path = Path(args.out) if args.out else split_dir / f"predictions_{args.split}.csv"
    if out_path.is_file():
        # carry forward predictions for models we are NOT recomputing this run
        previous = pd.read_csv(out_path)
        keep = [
            c
            for c in previous.columns
            if c.startswith("pred_") and c.removeprefix("pred_") not in args.models
        ]
        if keep:
            clips = clips.merge(previous[["clip_uid", *keep]], on="clip_uid", how="left")

    if "utmos22" in args.models:
        clips["pred_utmos22"] = predict_utmos22(clips, args.device)

    if "utmosv2" in args.models:
        stored = load_stored_utmosv2()
        clips = clips.merge(
            stored, on=["dataset", "system", "language", "filename"], how="left"
        ).rename(columns={"tts_utmos": "pred_utmosv2"})
        missing = int(clips["pred_utmosv2"].isna().sum())
        if missing:
            print(f"WARNING: {missing} clips lack a stored tts_utmos value")

        if args.utmosv2 == "rerun":
            clips["pred_utmosv2"] = predict_utmosv2(clips)
        elif args.utmosv2 == "verify":
            sample = clips.sample(n=min(args.verify_n, len(clips)), random_state=0)
            fresh = predict_utmosv2(sample)
            delta = (pd.Series(fresh, index=sample.index) - sample["pred_utmosv2"]).abs()
            print(
                f"\nUTMOSv2 stored-vs-fresh on {len(sample)} clips: "
                f"max|d|={delta.max():.4f}  mean|d|={delta.mean():.4f}  "
                f"n(>0.01)={int((delta > 0.01).sum())}"
            )

    clips.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    cols = [c for c in clips.columns if c.startswith("pred_")]
    if cols:
        print(clips[cols + ["mos"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
