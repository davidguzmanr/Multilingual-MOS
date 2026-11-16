"""
Plot the distribution of human MOS ratings per language from the OpenBibleTTS
listening study (data/annotations-anonymized/*.csv).

One figure per language: one panel per system (share of ratings at each score
1-5) plus an "All systems" panel. Writes PNGs to data/figures/mos-distribution/.

    python src/plot_mos_distributions.py
"""

from __future__ import annotations

import glob
import os
import urllib.parse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ANNO = REPO / "data" / "annotations-anonymized"
OUT = REPO / "data" / "figures" / "mos-distribution"

SYSTEM_ORDER = ["ground-truth", "Gemini", "OmniVoice", "F5", "EveryVoice", "VITS"]
SYSTEM_LABEL = {"ground-truth": "Ground truth", "Gemini": "Gemini TTS", "OmniVoice": "OmniVoice",
                "F5": "F5-TTS", "EveryVoice": "EveryVoice", "VITS": "VITS"}
SCORES = [1, 2, 3, 4, 5]
SCORE_LABEL = ["1\nBad", "2\nPoor", "3\nFair", "4\nGood", "5\nExcellent"]

# fixed categorical hue per system (reference palette slots 1-6, documented order)
SYSTEM_COLOR = {"ground-truth": "#4a6fa5", "Gemini": "#c4714b", "OmniVoice": "#4f9a7e",
                "F5": "#b8933a", "EveryVoice": "#b06b8a", "VITS": "#6f8f5a"}
OVERALL = "#7a7975"  # text-secondary gray for the pooled panel
TEXT = "#0b0b0b"; TEXT2 = "#52514e"; GRID = "#e6e5e1"; SURFACE = "#fcfcfb"


def load() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(ANNO / "*.csv")))]
    df = pd.concat(frames, ignore_index=True)
    # keep the most recent rating per (annotator, clip)
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df = (df.sort_values("updated_at", ascending=False)
            .drop_duplicates(subset=["annotator", "audio_url"], keep="first"))
    parts = df["audio_url"].str.split("/audios/").str[1].str.split("/")
    df["system"] = parts.str[1]
    df["language"] = parts.str[2].map(urllib.parse.unquote)
    df["mos"] = df["mos"].str.split().str[0].astype(int)
    return df[["language", "system", "dataset", "filename", "annotator", "mos"]]


def panel(ax, scores: pd.Series, title: str, color: str, ymax: float) -> None:
    counts = scores.value_counts().reindex(SCORES, fill_value=0)
    share = counts / counts.sum() * 100
    bars = ax.bar(SCORES, share, width=0.72, color=color, zorder=3)
    for b, s, c in zip(bars, share, counts):
        if c:
            ax.text(b.get_x() + b.get_width() / 2, s + ymax * 0.015, f"{s:.0f}%",
                    ha="center", va="bottom", fontsize=8.5, color=TEXT2)
    ax.set_title(title, fontsize=11, color=TEXT, loc="left", pad=6, fontweight="medium")
    ax.set_title(f"mean {scores.mean():.2f}  ·  n = {len(scores):,}", fontsize=8.5, color=TEXT2,
                 loc="right", pad=6)
    ax.set_xticks(SCORES, SCORE_LABEL, fontsize=8.5, color=TEXT2)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.tick_params(axis="y", labelsize=8.5, colors=TEXT2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)


def plot_language(df: pd.DataFrame, language: str) -> Path:
    systems = [s for s in SYSTEM_ORDER if s in set(df["system"])]
    n_panels = len(systems) + 1
    ncols = 4 if n_panels > 6 else 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows + 0.7),
                             facecolor=SURFACE, squeeze=False)
    axes = axes.ravel()

    # common y-limit so panels are comparable
    shares = [df[df["system"] == s]["mos"].value_counts(normalize=True).max() for s in systems]
    shares.append(df["mos"].value_counts(normalize=True).max())
    ymax = min(100, np.ceil(max(shares) * 100 / 10) * 10 + 10)

    for ax, s in zip(axes, systems):
        panel(ax, df[df["system"] == s]["mos"], SYSTEM_LABEL[s], SYSTEM_COLOR[s], ymax)
    panel(axes[len(systems)], df["mos"], "All systems", OVERALL, ymax)
    for ax in axes[n_panels:]:
        ax.axis("off")
    for i, ax in enumerate(axes[:n_panels]):
        if i % ncols:
            ax.set_yticklabels([])

    n_clips = df.drop_duplicates(["dataset", "system", "filename"]).shape[0]
    n_ann = df["annotator"].nunique()
    fig.suptitle(f"{language}: distribution of human MOS ratings", x=0.01, ha="left",
                 fontsize=14, color=TEXT, fontweight="semibold")
    fig.text(0.01, 0.955 - 0.012 * nrows, f"OpenBibleTTS listening study. {n_clips:,} clips, "
             f"{len(df):,} ratings, {n_ann} native annotators. Share of ratings at each score.",
             fontsize=9.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.0, w_pad=1.2)
    out = OUT / f"{language.replace(' ', '_')}.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    for lang, sub in df.groupby("language"):
        print(plot_language(sub, lang).relative_to(REPO))


if __name__ == "__main__":
    main()
