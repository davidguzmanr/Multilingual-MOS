"""Text-match check for ground-truth candidates: transcribe with omniASR (Meta omnilingual-asr, omniASR_LLM_1B_v2, the model
used for the OpenBibleTTS metrics) and score WER / CER against the sentence after the same normalisation as
open-bible-surveys/utils/evaluate-tts.py (casefold, punctuation and symbols to spaces, Arabic tashkeel stripped).
Runs in the Multilingual-MOS-asr env (torch 2.8 + fairseq2 0.6); GPU recommended.

    python src/asr_check.py --language Yoruba --in candidates.csv --out scored.csv [--path-col path --text-col text]
      in: CSV with an audio path column and a text column; out: same rows + asr_hyp, wer, cer, cer_plain (scored rows are kept).
    from asr_check import ASRCheck; a = ASRCheck("Yoruba"); a.score(paths, texts) -> list of (hyp, wer, cer, cer_plain)
"""
import argparse, os, sys, unicodedata
import pandas as pd

ASR_LANG = {"Arabic": "arb_Arab", "French": "fra_Latn", "German": "deu_Latn", "Haitian Creole": "hat_Latn", "Hausa": "hau_Latn",
            "Hindi": "hin_Deva", "Indonesian": "ind_Latn", "Japanese": "jpn_Jpan", "Korean": "kor_Hang", "Spanish": "spa_Latn",
            "Swahili": "swh_Latn", "Turkish": "tur_Latn", "Vietnamese": "vie_Latn", "Yoruba": "yor_Latn"}
ARABIC_SCRIPT = {"arb_Arab"}
MODEL_CARD = "omniASR_LLM_1B_v2"


def normalise(text, lang):
    text = str(text).casefold()
    if lang in ARABIC_SCRIPT: text = "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))
    out = [" " if unicodedata.category(ch).startswith(("P", "S")) else ch for ch in text]
    return " ".join("".join(out).split())


def cer(ref, hyp):
    """Character error rate (Levenshtein over characters, spaces included) on normalised strings."""
    import jiwer
    return jiwer.cer(ref, hyp) if ref else float("nan")


def plain(text):
    """Tone marks and other combining marks dropped, dotted Yoruba letters folded (ọ->o, ẹ->e, ṣ->s): the ASR is not
    reliable on diacritics, so the content check gates on this variant (cer_plain) while cer keeps the strict value."""
    t = unicodedata.normalize("NFD", str(text)); t = "".join(c for c in t if not unicodedata.combining(c))
    return t.replace("ọ", "o").replace("ẹ", "e").replace("ṣ", "s").replace("Ọ", "O").replace("Ẹ", "E").replace("Ṣ", "S")


class ASRCheck:
    def __init__(self, language, model_card=MODEL_CARD, batch_size=16):
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        self.lang, self.bs = ASR_LANG[language], batch_size
        self.pipe = ASRInferencePipeline(model_card=model_card)

    def transcribe(self, paths):
        hyps = []
        for i in range(0, len(paths), self.bs):
            b = paths[i:i + self.bs]; hyps += self.pipe.transcribe(b, lang=[self.lang] * len(b), batch_size=len(b))
        return hyps

    def score(self, paths, texts):
        import jiwer
        out = []
        for hyp, ref in zip(self.transcribe(paths), texts):
            r, h = normalise(ref, self.lang), normalise(hyp, self.lang)
            out.append((hyp, jiwer.wer(r, h) if r else float("nan"), cer(r, h), cer(normalise(plain(ref), self.lang), normalise(plain(hyp), self.lang))))
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--language", required=True, choices=sorted(ASR_LANG))
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--path-col", default="path"); ap.add_argument("--text-col", default="text"); ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    df = pd.read_csv(a.inp, dtype=str)
    if os.path.exists(a.out):
        done = pd.read_csv(a.out, dtype=str); df = df.merge(done[[a.path_col, "asr_hyp", "wer", "cer", "cer_plain"]], on=a.path_col, how="left")
    else: df["asr_hyp"], df["wer"], df["cer"], df["cer_plain"] = "", float("nan"), float("nan"), float("nan")
    todo = df[df.cer.isna() | (df.cer == "")].index
    todo = [i for i in todo if isinstance(df.at[i, a.path_col], str) and os.path.isfile(df.at[i, a.path_col])]
    print(f"{len(todo)} of {len(df)} rows to transcribe ({a.language})")
    if todo:
        chk = ASRCheck(a.language, batch_size=a.batch_size)
        for s in range(0, len(todo), 64):
            idx = todo[s:s + 64]
            for i, (hyp, w, c, cp) in zip(idx, chk.score([df.at[i, a.path_col] for i in idx], [df.at[i, a.text_col] for i in idx])):
                df.at[i, "asr_hyp"], df.at[i, "wer"], df.at[i, "cer"], df.at[i, "cer_plain"] = hyp, round(w, 4), round(c, 4), round(cp, 4)
            df.to_csv(a.out, index=False); print(f"  {min(s + 64, len(todo))}/{len(todo)}", flush=True)
    df.to_csv(a.out, index=False)
    print(df[["wer", "cer", "cer_plain"]].astype(float).describe().round(3).to_string())
