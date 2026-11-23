"""Proxy for the listening-test outcome before the survey: distribution of UTMOS (UTMOS22 strong, naturalness proxy) and omniASR WER
(intelligibility proxy) per system for one language, in the layout of src/plot_mos_distributions.py (one panel per system
plus a pooled panel). Reads data/proxy/<Language>_scores.csv (src/proxy_scores.py) and writes
data/figures/proxy/<Language>_utmos.png and <Language>_wer.png, plus a per-system summary table on stdout.

    python src/plot_proxy_distributions.py --language Yoruba
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "figures" / "proxy"
SYSTEM_ORDER = ["ground-truth", "Gemini", "ElevenLabs", "OmniVoice", "VITS", "EveryVoice", "VoxCPM2", "VoxCPM2-Yoruba", "Fish-S2-Pro",
                "Higgs-TTS-3", "Sunbird-Orpheus-3B"]
LABEL = {"ground-truth": "Ground truth", "Gemini": "Gemini TTS", "ElevenLabs": "ElevenLabs", "OmniVoice": "OmniVoice", "VITS": "VITS (MMS)",
         "EveryVoice": "EveryVoice", "VoxCPM2": "VoxCPM2", "VoxCPM2-Yoruba": "VoxCPM2-Yoruba", "Fish-S2-Pro": "Fish S2-Pro",
         "Higgs-TTS-3": "Higgs TTS 3", "Sunbird-Orpheus-3B": "Sunbird Orpheus-3B"}
PALETTE = ["#4a6fa5", "#c4714b", "#4f9a7e", "#b8933a", "#b06b8a", "#6f8f5a", "#8a6fb0", "#c05c5c", "#5c9ab0", "#a0845c", "#6f6f6f"]
OVERALL = "#7a7975"; TEXT = "#0b0b0b"; TEXT2 = "#52514e"; GRID = "#e6e5e1"; SURFACE = "#fcfcfb"

UTMOS_BINS = [1, 2, 3, 4, 5]; UTMOS_LABEL = ["1\nBad", "2\nPoor", "3\nFair", "4\nGood", "5\nExcellent"]          # rounded UTMOS
WER_EDGES = [-0.001, 0.0, 0.10, 0.25, 0.50, 10]; WER_LABEL = ["0%", "0-10%", "10-25%", "25-50%", ">50%"]        # WER bands


def bin_utmos(s): return pd.Series(np.clip(np.rint(s.astype(float)), 1, 5).astype(int), index=s.index)
def bin_wer(s): return pd.cut(s.astype(float).clip(upper=9), WER_EDGES, labels=WER_LABEL)


def panel(ax, binned, cats, labels, title, right, color, ymax):
    counts = binned.value_counts().reindex(cats, fill_value=0); share = counts / max(counts.sum(), 1) * 100
    x = np.arange(len(cats)); bars = ax.bar(x, share.values, width=0.72, color=color, zorder=3)
    for b, s, c in zip(bars, share.values, counts.values):
        if c: ax.text(b.get_x() + b.get_width() / 2, s + ymax * 0.015, f"{s:.0f}%", ha="center", va="bottom", fontsize=8.5, color=TEXT2)
    ax.set_title(title, fontsize=11, color=TEXT, loc="left", pad=6, fontweight="medium")
    ax.set_title(right, fontsize=8.5, color=TEXT2, loc="right", pad=6)
    ax.set_xticks(x, labels, fontsize=8.5, color=TEXT2); ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.tick_params(axis="y", labelsize=8.5, colors=TEXT2, length=0); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.set_facecolor(SURFACE)


def figure(df, language, metric):
    systems = [s for s in SYSTEM_ORDER if s in set(df.system)] + sorted(set(df.system) - set(SYSTEM_ORDER))
    n = len(systems) + 1; ncols = 4 if n > 6 else 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows + 0.8), facecolor=SURFACE, squeeze=False); axes = axes.ravel()
    if metric == "utmos": binf, cats, labels, name = bin_utmos, UTMOS_BINS, UTMOS_LABEL, "UTMOS (rounded)"
    elif metric == "wer": binf, cats, labels, name = bin_wer, WER_LABEL, WER_LABEL, "omniASR word error rate"
    else: binf, cats, labels, name = bin_wer, WER_LABEL, WER_LABEL, "omniASR character error rate (tone marks stripped)"
    shares = [binf(df[df.system == s][metric]).value_counts(normalize=True).max() for s in systems] + [binf(df[metric]).value_counts(normalize=True).max()]
    ymax = min(100, np.ceil(max(shares) * 100 / 10) * 10 + 10)
    for ax, s, color in zip(axes, systems, PALETTE * 3):
        sub = df[df.system == s][metric].astype(float)
        right = f"mean {sub.mean():.2f}  ·  n = {len(sub)}" if metric == "utmos" else f"median {sub.median()*100:.0f}%  ·  n = {len(sub)}"
        panel(ax, binf(sub), cats, labels, LABEL.get(s, s), right, color, ymax)
    allv = df[metric].astype(float)
    right = f"mean {allv.mean():.2f}  ·  n = {len(allv)}" if metric == "utmos" else f"median {allv.median()*100:.0f}%  ·  n = {len(allv)}"
    panel(axes[len(systems)], binf(allv), cats, labels, "All systems", right, OVERALL, ymax)
    for ax in axes[n:]: ax.axis("off")
    for i, ax in enumerate(axes[:n]):
        if i % ncols: ax.set_yticklabels([])
    fig.suptitle(f"{language}: {name} per system (proxy before the listening test)", x=0.01, ha="left", fontsize=14, color=TEXT, fontweight="semibold")
    fig.text(0.01, 0.955 - 0.012 * nrows, f"{len(df):,} clips = {df.clip_id.nunique()} sentences x {len(systems)} systems (ground truth included). "
             + {"utmos": "UTMOS22-strong scores rounded to the 1-5 MOS scale.", "wer": "WER of omniASR_LLM_1B_v2 transcripts vs the sentence (punctuation/case-insensitive; the ASR's own tone-mark errors count).",
                "cer_plain": "CER of omniASR_LLM_1B_v2 transcripts vs the sentence with diacritics stripped (the ASR is unreliable on tone marks)."}[metric],
             fontsize=9.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.0, w_pad=1.2)
    out = OUT / f"{language.replace(' ', '_')}_{metric}.png"; fig.savefig(out, dpi=160, facecolor=SURFACE); plt.close(fig); return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--language", required=True); a = ap.parse_args()
    df = pd.read_csv(REPO / "data" / "proxy" / f"{a.language}_scores.csv"); OUT.mkdir(parents=True, exist_ok=True)
    for c in ["utmos", "wer", "cer_plain"]: df[c] = df[c].astype(float)
    summary = df.groupby("system").agg(clips=("clip_id", "size"), utmos_mean=("utmos", "mean"), utmos_sd=("utmos", "std"), wer_mean=("wer", "mean"),
                                       wer_median=("wer", "median"), cer_plain_mean=("cer_plain", "mean"))
    summary = summary.reindex([s for s in SYSTEM_ORDER if s in summary.index] + sorted(set(summary.index) - set(SYSTEM_ORDER)))
    pd.set_option("display.width", 200); print(summary.round(3).to_string())
    summary.round(4).to_csv(REPO / "data" / "proxy" / f"{a.language}_summary.csv")
    for metric in ["utmos", "wer", "cer_plain"]: print(figure(df, a.language, metric).relative_to(REPO))


if __name__ == "__main__":
    main()
