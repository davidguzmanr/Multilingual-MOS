"""
Build train/val/test splits for the multilingual MOS annotations.

Reads the per-language CSVs in ``data/annotations-anonymized/`` and emits a tidy
long-format table (one row per individual rating) tagged with a ``split`` column,
plus a per-clip table.  The output is model-agnostic: a downstream converter can
turn it into UTMOS ``TRAINSET`` files, UTMOSv2 ``mos_list`` files, or anything else.

Strategy: unseen system x corpus pairing (Latin-square diagonal)
----------------------------------------------------------------
The data is a fully-crossed design -- per language, 20 sentences x 3 corpora x
{5 TTS systems + ground-truth}, each clip rated by the same 3 annotators.  One
(corpus, system) cell per *system* is held out for test, and the same six cells
are held out in every language.

The held-out system is still seen during training in its other two corpora, and
the held-out corpus is still seen with the other systems -- only the
intersection is novel.  Test therefore covers all 6 systems in all 10 languages
(1200 clips, 20 per language x system), which is what makes system-level
correlations computable, while every test clip still comes from a pairing the
model never saw.

Note that the test sentences do appear in training, rendered by other systems.
That is deliberate: the task is judging the quality of a *rendering*, not of a
sentence, and the VoiceMOS main track has the same property.  It does mean the
phoneme encoder sees test sentences during training, so a sentence-level prior is
learnable in principle.

Examples
--------
    # Default diagonal
    python src/build_splits.py

    # Choose a different diagonal (one cell per system, all 6 systems)
    python src/build_splits.py --test-cells bouquet:F5 bouquet:Gemini fleurs:EveryVoice \
        fleurs:VITS open-bible:OmniVoice open-bible:ground-truth

    # No validation holdout (train gets those clips back)
    python src/build_splits.py --no-val
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNO_DIR = REPO_ROOT / "data" / "annotations-anonymized"
AUDIO_DIR = REPO_ROOT / "data" / "audios"
OUT_ROOT = REPO_ROOT / "data" / "splits"

# One held-out cell per system: a Latin-square "diagonal" over the 3 corpora x
# 6 systems grid.  Every system appears in test exactly once, and every system is
# still trained on in its two remaining corpora -- so only the (corpus, system)
# intersection is novel, never the system itself.
#
# ground-truth must sit in open-bible: that is its only cell present in all 10
# languages (Haitian Creole has no fleurs ground-truth).  With this constraint
# all 10 languages contribute all 6 systems, giving a fully balanced
# 10 x 6 x 20 = 1200-clip test set.
#
# Chosen by enumerating all 30 valid diagonals, keeping those whose pooled test
# mean MOS is within 0.05 of the grand mean (3.444), and taking the one with the
# largest spread between system means -- because per-language system-level SRCC
# is computed over only 6 points, and well-separated systems make that ranking
# far less noise-dominated.
DEFAULT_TEST_CELLS = [
    "bouquet:Gemini",
    "bouquet:OmniVoice",
    "fleurs:F5",
    "fleurs:VITS",
    "open-bible:EveryVoice",
    "open-bible:ground-truth",
]

# Validation is a *sentence* holdout carved from the non-test cells rather than
# more cells: taking a second diagonal would leave ground-truth with a single
# training cell and strip whole systems down to one corpus.
DEFAULT_VAL_SENTENCES = 3

DIFFICULTY_TIERS = [
    (0, "unanimous"),
    (1, "easy"),
    (2, "hard"),
]


# ── Loading ──────────────────────────────────────────────────────────────────

def load_annotations() -> pd.DataFrame:
    """Load every language CSV into one tidy frame, one row per rating."""
    frames = []
    for csv_path in sorted(ANNO_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["language"] = csv_path.stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No annotation CSVs found under {ANNO_DIR}")

    d = pd.concat(frames, ignore_index=True)

    # audio_url: .../audios/{dataset}/{system}/{language}/{filename}
    url_parts = d["audio_url"].str.split("/audios/").str[1].str.split("/")
    d["system"] = url_parts.str[1]

    # "3 — Fair" -> 3
    d["rating"] = d["mos"].str.split().str[0].astype(int)

    d = d[
        [
            "language",
            "dataset",
            "system",
            "filename",
            "annotator",
            "rating",
            "reference_text",
            "agreement",
            "annotation_id",
            "created_at",
        ]
    ]

    # A handful of clips carry a duplicate submission from the same annotator
    # (identical score, seconds apart).  Keep the most recent.
    before = len(d)
    d = (
        d.sort_values("created_at")
        .drop_duplicates(["language", "dataset", "system", "filename", "annotator"], keep="last")
        .reset_index(drop=True)
    )
    n_dropped = before - len(d)
    if n_dropped:
        print(f"Dropped {n_dropped} duplicate (clip, annotator) submissions.")

    d["audio_path"] = [
        str(AUDIO_DIR / row.dataset / row.system / row.language / row.filename)
        for row in d.itertuples()
    ]
    return d


def add_clip_identifiers(d: pd.DataFrame) -> pd.DataFrame:
    """Attach stable per-clip ids.

    ``utt_id`` is unique per *clip*, not per sentence: the same sentence is
    synthesised by every system, and collapsing them would make utterance-level
    ground truth average a good system with a bad one.
    """
    d = d.copy()
    d["clip_uid"] = (
        d["language"] + "/" + d["dataset"] + "/" + d["system"] + "/" + d["filename"]
    )
    d["sentence_uid"] = d["language"] + "/" + d["dataset"] + "/" + d["filename"]

    clips = sorted(d["clip_uid"].unique())
    utt_index = {uid: i for i, uid in enumerate(clips)}
    d["utt_id"] = d["clip_uid"].map(lambda u: f"utt{utt_index[u]:05d}")

    def system_id(row) -> str:
        lang = row.language.replace(" ", "")
        return f"sys{row.dataset.replace('-', '')}{row.system.replace('-', '')}{lang}"

    d["system_id"] = [system_id(row) for row in d.itertuples()]
    return d


def add_difficulty(d: pd.DataFrame) -> pd.DataFrame:
    """Tier each clip by how far apart its annotators were.

    This is a reporting dimension, not a selection criterion: with only 3 raters
    a wide spread is partly annotator noise rather than intrinsic difficulty, so
    biasing the test set toward it would optimise against annotation variance.
    """
    spread = d.groupby("clip_uid")["rating"].agg(lambda s: s.max() - s.min())

    def tier(value: int) -> str:
        for threshold, name in DIFFICULTY_TIERS:
            if value <= threshold:
                return name
        return "very_hard"

    d = d.copy()
    d["rating_spread"] = d["clip_uid"].map(spread)
    d["difficulty"] = d["rating_spread"].map(tier)
    return d


# ── Split assignment ─────────────────────────────────────────────────────────

def parse_cell(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"Cell must be 'CORPUS:SYSTEM', got {spec!r}")
    corpus, system = spec.split(":", 1)
    return corpus, system


def validate_cells(d: pd.DataFrame, cells: list[tuple[str, str]]) -> None:
    """Fail loudly if the test diagonal is malformed or unevenly covered."""
    systems = sorted(d["system"].unique())
    chosen = [s for _, s in cells]
    if sorted(chosen) != systems:
        raise ValueError(
            f"Test cells must cover each system exactly once. "
            f"Got {sorted(chosen)}, data has {systems}."
        )

    for corpus, system in cells:
        mask = d["dataset"].eq(corpus) & d["system"].eq(system)
        if not mask.any():
            raise ValueError(f"Test cell {corpus}:{system} matches no data.")
        per_lang = d[mask].groupby("language")["clip_uid"].nunique()
        missing = sorted(set(d["language"].unique()) - set(per_lang.index))
        if missing:
            print(
                f"WARNING: test cell {corpus}:{system} is absent for {missing}; "
                f"those languages contribute no test clips for that system."
            )

    # every system must survive in training somewhere
    for system in systems:
        held = {c for c, s in cells if s == system}
        available = set(d.loc[d["system"].eq(system), "dataset"].unique())
        if not (available - held):
            raise ValueError(
                f"System {system} would have no training cell left "
                f"(present in {sorted(available)}, held out {sorted(held)})."
            )


def assign_splits(
    d: pd.DataFrame,
    test_cells: list[tuple[str, str]],
    val_sentences: int,
    seed: int,
) -> pd.Series:
    """Label every rating row: test diagonal first, then a val sentence holdout.

    Validation sentences are drawn per (language, corpus) and removed from every
    non-test cell, so val never shares a sentence with train.  Test is defined
    purely by cell, so test sentences do still appear in training under other
    systems -- see the module docstring.
    """
    split = pd.Series("train", index=d.index, dtype=object)

    key = set(test_cells)
    is_test = pd.Series(
        [(row.dataset, row.system) in key for row in d.itertuples()], index=d.index
    )

    if val_sentences > 0:
        val_uids: set[str] = set()
        pool = d[~is_test]
        for (lang, corpus), group in pool.groupby(["language", "dataset"]):
            uids = sorted(group["sentence_uid"].unique())
            rng = np.random.RandomState(abs(hash((lang, corpus, seed))) % (2**31))
            rng.shuffle(uids)
            val_uids.update(uids[:val_sentences])
        split[~is_test & d["sentence_uid"].isin(val_uids)] = "val"

    split[is_test] = "test"
    return split


def build_clip_table(ratings: pd.DataFrame) -> pd.DataFrame:
    """Collapse ratings to one row per clip with the averaged MOS."""
    grouped = ratings.groupby(
        [
            "clip_uid",
            "language",
            "dataset",
            "system",
            "filename",
            "sentence_uid",
            "utt_id",
            "system_id",
            "audio_path",
            "reference_text",
            "difficulty",
            "rating_spread",
            "split",
        ],
        as_index=False,
    ).agg(mos=("rating", "mean"), n_ratings=("rating", "size"))
    return grouped.sort_values(["language", "dataset", "system", "filename"])


# ── Reporting ────────────────────────────────────────────────────────────────

def summarise(clips: pd.DataFrame, ratings: pd.DataFrame) -> str:
    order = [s for s in ["train", "val", "test"] if s in set(clips["split"])]
    lines = []
    lines.append("Clips per split")
    lines.append(
        clips.groupby(["language", "split"]).size().unstack(fill_value=0)[order].to_string()
    )
    lines.append("")
    lines.append("Ratings per split")
    lines.append(
        ratings.groupby(["language", "split"]).size().unstack(fill_value=0)[order].to_string()
    )
    lines.append("")
    lines.append("Systems x corpora in each split (clips)")
    pivot = (
        clips.groupby(["dataset", "system", "split"]).size().unstack(fill_value=0)
    )
    lines.append(pivot.reindex(columns=order, fill_value=0).to_string())
    lines.append("")
    lines.append("Difficulty tiers within each split (clips)")
    lines.append(
        clips.groupby(["split", "difficulty"]).size().unstack(fill_value=0).to_string()
    )
    lines.append("")
    lines.append("MOS per split")
    lines.append(clips.groupby("split")["mos"].agg(["mean", "std", "size"]).to_string())
    return "\n".join(lines)


def check_audio(clips: pd.DataFrame) -> None:
    missing = [p for p in clips["audio_path"].unique() if not os.path.isfile(p)]
    if missing:
        print(f"WARNING: {len(missing)} referenced wav files not found, e.g. {missing[:3]}")
    else:
        print(f"All {clips['audio_path'].nunique()} referenced wav files exist.")


# ── Main ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tag", default="system-corpus", help="Output subdirectory name.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--test-cells",
        nargs="+",
        default=DEFAULT_TEST_CELLS,
        help="Held-out test diagonal, one CORPUS:SYSTEM per system.",
    )
    p.add_argument(
        "--val-sentences",
        type=int,
        default=DEFAULT_VAL_SENTENCES,
        help="Sentences per (language, corpus) held out from non-test cells for validation.",
    )
    p.add_argument("--no-val", action="store_true", help="Do not hold out a validation set.")
    return p.parse_args()


def main():
    args = get_args()

    d = load_annotations()
    d = add_clip_identifiers(d)
    d = add_difficulty(d)
    print(
        f"Loaded {len(d):,} ratings over {d['clip_uid'].nunique():,} clips, "
        f"{d['language'].nunique()} languages, {d['annotator'].nunique()} annotators."
    )

    test_cells = [parse_cell(spec) for spec in args.test_cells]
    validate_cells(d, test_cells)
    val_sentences = 0 if args.no_val else args.val_sentences

    ratings = d.assign(split=assign_splits(d, test_cells, val_sentences, args.seed))
    clips = build_clip_table(ratings)

    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(out_dir / "ratings.csv", index=False)
    clips.to_csv(out_dir / "clips.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "strategy": (
                    "unseen system x corpus pairing; one held-out cell per system "
                    "(Latin-square diagonal), identical across all languages"
                ),
                "test_cells": [f"{c}:{s}" for c, s in test_cells],
                "val_sentences_per_language_corpus": val_sentences,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    report = summarise(clips, ratings)
    (out_dir / "summary.txt").write_text(report)
    print()
    print(report)

    check_audio(clips)
    print(f"\nWrote splits to {out_dir}")


if __name__ == "__main__":
    main()
