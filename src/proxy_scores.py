"""Proxy quality scores for a language's listening-test clips (ground truth + every synthesized system) before the survey:
UTMOS (UTMOS22 strong, as in the pilot) as a naturalness proxy and omniASR WER / CER as an intelligibility proxy.

Two steps, two envs (GPU for both):
  1. ASR (env Multilingual-MOS-asr):  python src/proxy_scores.py --language Yoruba --step list
                                      python src/asr_check.py --language Yoruba --in data/proxy/Yoruba_clips.csv --out data/proxy/Yoruba_asr.csv
  2. UTMOS (env Multilingual-MOS):       python src/proxy_scores.py --language Yoruba --step utmos
Output: data/proxy/<Language>_scores.csv with language, system, dataset, clip_id, text, asr_hyp, wer, cer, cer_plain, utmos.
Then: python src/plot_proxy_distributions.py --language Yoruba
"""
import argparse, glob, os
import pandas as pd

REPO = "/network/scratch/g/guzmand/Repositories/Multilingual-MOS"; OUT = f"{REPO}/data/proxy"


def list_clips(language):
    rows = []
    gt = pd.read_csv(f"{REPO}/audios/{language}/ground-truth/metadata.csv", dtype=str)
    for r in gt.itertuples(): rows.append(dict(language=language, system="ground-truth", dataset=r.dataset, clip_id=r.clip_id, text=r.text, path=f"{REPO}/{r.audio_path}"))
    for m in sorted(glob.glob(f"{REPO}/audios/{language}/*/metadata.csv")):
        s = m.split("/")[-2]
        if s in ("ground-truth", "reference-voices"): continue
        for r in pd.read_csv(m, dtype=str).itertuples(): rows.append(dict(language=language, system=s, dataset=r.dataset, clip_id=r.clip_id, text=r.text, path=f"{REPO}/{r.audio_path}"))
    df = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True); df.to_csv(f"{OUT}/{language}_clips.csv", index=False)
    print(f"{len(df)} clips, {df.system.nunique()} systems -> {OUT}/{language}_clips.csv"); return df


def utmos(language):
    """UTMOS22 strong learner (the UTMOS used in the OpenBibleTTS pilot), through the fairseq-free port tarepan/SpeechMOS
    (same weights, loaded with torch.hub); input resampled to 16 kHz mono."""
    import torch, torchaudio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
    clips = pd.read_csv(f"{OUT}/{language}_clips.csv", dtype=str); scores = []
    for i, p in enumerate(clips.path, 1):
        wav, sr = torchaudio.load(p); wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
        with torch.no_grad(): scores.append(float(predictor(wav.to(device), 16000).item()))
        if i % 50 == 0: print(f"  {i}/{len(clips)}", flush=True)
    clips["utmos"] = scores
    asr_path = f"{OUT}/{language}_asr.csv"
    if os.path.exists(asr_path):
        asr = pd.read_csv(asr_path, dtype=str)[["path", "asr_hyp", "wer", "cer", "cer_plain"]]
        clips = clips.merge(asr, on="path", how="left")
    else: print("NOTE: no ASR file yet; run asr_check.py and rerun --step utmos to merge")
    clips.to_csv(f"{OUT}/{language}_scores.csv", index=False)
    cols = [c for c in ["utmos", "wer", "cer_plain"] if c in clips]
    print(clips.assign(**{c: clips[c].astype(float) for c in cols}).groupby("system")[cols].mean().round(3).to_string())
    print(f"-> {OUT}/{language}_scores.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--language", required=True); ap.add_argument("--step", choices=["list", "utmos"], required=True)
    a = ap.parse_args(); (list_clips if a.step == "list" else utmos)(a.language)
