"""
Evaluate MOS predictions against human ratings on a split.

Reports, for every prediction column in the predictions CSV:

  utterance-level   MSE / LCC / SRCC over individual clips
  system-level      the same, over system mean scores, at two granularities:

    pooled (n=60)   each (language, system) pair is one "system".  This is the
                    headline number -- 60 points is enough for the correlation
                    to mean something.
    per-language    6 systems within one language.  At n=6 the 5% significance
      (n=6)         threshold is |rho| ~ 0.886, so single-language values swing
                    hard on sampling noise; read the spread, not any one row.

MSE is reported but is the weakest of the three here: with 3 raters per clip the
standard error of a clip's mean MOS is ~0.34, putting a noise floor near 0.11 on
utterance MSE even for a perfect model.  Models not calibrated to this scale
(UTMOS22 is trained on BVCC English) will show large MSE regardless of how well
they rank, so prefer SRCC when comparing them.

Example
-------
    python src/evaluate_mos.py --tag system-corpus --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_ROOT = REPO_ROOT / "data" / "splits"


def compute_metrics(true_scores: np.ndarray, pred_scores: np.ndarray) -> dict:
    """MSE, Pearson LCC and Spearman SRCC. NaN when undefined (n < 2)."""
    if len(true_scores) < 2:
        return {"MSE": np.nan, "LCC": np.nan, "SRCC": np.nan, "n": len(true_scores)}
    mse = float(np.mean((true_scores - pred_scores) ** 2))
    lcc = float(scipy.stats.pearsonr(true_scores, pred_scores)[0])
    srcc = float(scipy.stats.spearmanr(true_scores, pred_scores)[0])
    return {"MSE": mse, "LCC": lcc, "SRCC": srcc, "n": len(true_scores)}


def system_means(df: pd.DataFrame, pred_col: str, keys: list[str]) -> tuple:
    """Collapse clips to system means, then correlate. `keys` defines a system."""
    grouped = df.groupby(keys).agg(true=("mos", "mean"), pred=(pred_col, "mean"))
    return grouped["true"].to_numpy(), grouped["pred"].to_numpy()


def evaluate(df: pd.DataFrame, pred_col: str) -> dict:
    sub = df.dropna(subset=["mos", pred_col])
    utt = compute_metrics(sub["mos"].to_numpy(), sub[pred_col].to_numpy())
    t, p = system_means(sub, pred_col, ["language", "system"])
    pooled = compute_metrics(t, p)
    return {"utterance": utt, "system_pooled": pooled}


def per_language_table(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    rows = []
    for lang, group in df.groupby("language"):
        row = {"language": lang, "clips": len(group)}
        for col in pred_cols:
            sub = group.dropna(subset=["mos", col])
            utt = compute_metrics(sub["mos"].to_numpy(), sub[col].to_numpy())
            t, p = system_means(sub, col, ["system"])
            sys_m = compute_metrics(t, p)
            name = col.removeprefix("pred_")
            row[f"{name}_utt_SRCC"] = utt["SRCC"]
            row[f"{name}_sys_SRCC"] = sys_m["SRCC"]
            row[f"{name}_sys_n"] = sys_m["n"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("language")


def per_difficulty_table(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    order = ["unanimous", "easy", "hard", "very_hard"]
    rows = []
    for tier in order:
        group = df[df["difficulty"] == tier]
        if group.empty:
            continue
        row = {"difficulty": tier, "clips": len(group)}
        for col in pred_cols:
            sub = group.dropna(subset=["mos", col])
            row[f"{col.removeprefix('pred_')}_utt_SRCC"] = compute_metrics(
                sub["mos"].to_numpy(), sub[col].to_numpy()
            )["SRCC"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("difficulty")


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tag", default="system-corpus")
    p.add_argument("--split", default="test")
    p.add_argument("--predictions", default=None, help="Override predictions CSV path.")
    return p.parse_args()


def main():
    args = get_args()
    split_dir = SPLIT_ROOT / args.tag
    path = Path(args.predictions) if args.predictions else split_dir / f"predictions_{args.split}.csv"
    df = pd.read_csv(path)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise SystemExit(f"No pred_* columns in {path}")

    print(f"\n{path.relative_to(REPO_ROOT)}")
    print(f"{len(df)} clips | {df['language'].nunique()} languages | "
          f"{df['system'].nunique()} systems | models: {[c.removeprefix('pred_') for c in pred_cols]}")

    print("\n" + "=" * 78)
    print("OVERALL (all languages pooled)")
    print("=" * 78)
    header = f"{'model':10s} {'level':22s} {'n':>5s} {'MSE':>8s} {'LCC':>8s} {'SRCC':>8s}"
    print(header)
    print("-" * 78)
    for col in pred_cols:
        res = evaluate(df, col)
        name = col.removeprefix("pred_")
        for level, key in [("utterance", "utterance"), ("system (lang x sys)", "system_pooled")]:
            m = res[key]
            print(
                f"{name:10s} {level:22s} {m['n']:5d} {m['MSE']:8.3f} "
                f"{m['LCC']:8.3f} {m['SRCC']:8.3f}"
            )
    print("-" * 78)

    print("\n" + "=" * 78)
    print("PER LANGUAGE  (sys_SRCC is over 6 systems -- high variance)")
    print("=" * 78)
    table = per_language_table(df, pred_cols)
    print(table.round(3).to_string())
    srcc_cols = [c for c in table.columns if c.endswith("_SRCC")]
    print("\nmean / std across languages:")
    print(table[srcc_cols].agg(["mean", "std"]).round(3).to_string())

    print("\n" + "=" * 78)
    print("BY ANNOTATOR AGREEMENT (utterance SRCC)")
    print("=" * 78)
    print(per_difficulty_table(df, pred_cols).round(3).to_string())

    out = split_dir / f"metrics_{args.split}.csv"
    table.to_csv(out)
    print(f"\nPer-language table written to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
