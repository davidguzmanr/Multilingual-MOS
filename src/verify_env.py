"""Verify the shared UTMOS conda env can actually run UTMOS22 and UTMOSv2.

Goes past `import x` where it matters: builds a Lightning Trainer from UTMOS22's
real config, runs a wav2vec2 forward pass on GPU, and exercises WavAugment's sox
chain.  Run after src/setup_env.sh, or any time the env is touched.

    module load miniconda/3 && conda activate UTMOS
    python src/verify_env.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UTMOS22_STRONG = REPO_ROOT / "repositories" / "UTMOS22" / "strong"
W2V_CKPT = REPO_ROOT / "repositories" / "UTMOS22" / "fairseq_checkpoints" / "wav2vec_small.pt"

results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            results.append((name, True, fn() or "ok"))
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            detail = f"{type(exc).__name__}: {exc}".splitlines()[0][:160]
            results.append((name, False, detail))
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn

    return deco


@check("core versions")
def _core():
    import numpy
    import torch
    import transformers

    return (
        f"torch {torch.__version__} | numpy {numpy.__version__} "
        f"| transformers {transformers.__version__}"
    )


@check("CUDA visible")
def _cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False (login node?)")
    return torch.cuda.get_device_name(0)


@check("UTMOSv2 import")
def _v2_import():
    import utmosv2  # noqa: F401

    return "ok"


@check("fairseq import + checkpoint_utils")
def _fairseq():
    import fairseq
    from fairseq.checkpoint_utils import load_model_ensemble_and_task  # noqa: F401

    return f"fairseq {fairseq.__version__}"


@check("pytorch-lightning + UTMOS22 trainer_args")
def _pl():
    import yaml
    from pytorch_lightning import Trainer

    cfg = yaml.safe_load((UTMOS22_STRONG / "configs" / "train" / "default.yaml").read_text())
    Trainer(**cfg["trainer_args"])
    import pytorch_lightning as pl

    return f"PL {pl.__version__}, Trainer built from configs/train/default.yaml"


@check("UTMOS22 strong modules")
def _utmos22_modules():
    sys.path.insert(0, str(UTMOS22_STRONG))
    import dataset  # noqa: F401
    import lightning_module  # noqa: F401
    import loss_function  # noqa: F401
    import model  # noqa: F401

    return "model, dataset, loss_function, lightning_module"


@check("WavAugment (sox chain)")
def _augment():
    import augment
    import torch

    info = {
        "channels": 1,
        "length": 16000,
        "precision": 32,
        "rate": 16000.0,
        "bits_per_sample": 32,
    }
    chain = augment.EffectChain().pitch(100).rate(16000).tempo(1.05)
    out = chain.apply(torch.randn(1, 16000), src_info=dict(info), target_info=dict(info))
    return f"output {tuple(out.shape)}"


@check("wav2vec2 forward on GPU")
def _w2v():
    import fairseq
    import torch

    if not W2V_CKPT.is_file():
        raise FileNotFoundError(f"missing {W2V_CKPT}")
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(W2V_CKPT)])
    m = models[0]
    m.remove_pretraining_modules()
    m.eval().cuda()
    with torch.no_grad():
        out = m(torch.randn(1, 16000 * 3).cuda(), mask=False, features_only=True)["x"]
    return f"frame features {tuple(out.shape)}"


def main() -> int:
    width = max(len(n) for n, _, _ in results)
    print()
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:{width}}  {detail}")
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    if n_fail:
        print(f"{n_fail}/{len(results)} checks FAILED (re-run with -v for tracebacks)")
    else:
        print(f"All {len(results)} checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
