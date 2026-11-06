"""
Build the English annotator tutorial for the two-axis listening study.

Source: LJSpeech (via davidguzmanr/CSS10-Multilingual-LJSpeech, "English", test split).
Every example starts from a natural recording, then varies ONE axis at a time:

  naturalness ladder  same sentence, words untouched, increasingly synthetic-sounding
                      (Griffin-Lim at 100 and 16 iterations, robotization)
  accuracy ladder     natural audio untouched, mismatch with the text shown
                      (one substituted word, audio cut off at a pause, wrong sentence)
  combined            one example degraded on both axes, for practice

All clips are RMS-matched to their source so loudness is not a cue.

Outputs (next to this script):
  audios/*.wav            22.05 kHz, 16-bit mono
  examples.csv            target ratings and what to say about each clip
  tasks.json              Label Studio tasks (audio served from the dev/tutorial branch)
  labeling_config.xml     two-axis Label Studio interface (also the draft for the real survey)

    module load miniconda/3 && conda activate UTMOS
    python listening-study/tutorial/build_tutorial.py [--audio-base-url URL]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

HERE = Path(__file__).resolve().parent
AUDIO_DIR = HERE / "audios"
SR = 22050
N_FFT, HOP = 1024, 256
# Raw GitHub URLs only work if the repository is public. Override with --audio-base-url,
# e.g. https://huggingface.co/datasets/<user>/<repo>/resolve/main/tutorial
DEFAULT_AUDIO_BASE_URL = (
    "https://raw.githubusercontent.com/davidguzmanr/Multilingual-MOS/"
    "refs/heads/dev/tutorial/listening-study/tutorial/audios"
)
SEED = 0


# ── degradations ────────────────────────────────────────────────────────────

def griffin_lim(x: np.ndarray, n_iter: int) -> np.ndarray:
    """Discard phase and rebuild it with Griffin-Lim. 100 iterations: slightly hollow/phasey.
    16 iterations: clearly metallic and buzzy. Words survive both (checked with Whisper)."""
    mag = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP))
    return librosa.griffinlim(mag, n_iter=n_iter, hop_length=HOP, n_fft=N_FFT,
                              random_state=SEED, length=len(x))


def robotize(x: np.ndarray) -> np.ndarray:
    """Strong: zero all STFT phases. Monotone buzzy robot at SR/HOP Hz, words still audible."""
    mag = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP))
    return librosa.istft(mag.astype(np.complex64), hop_length=HOP, n_fft=N_FFT, length=len(x))


def truncate_at_pause(x: np.ndarray, target_frac: float = 0.6) -> tuple[np.ndarray, float]:
    """Cut the clip at the internal pause closest to target_frac of its length."""
    voiced = librosa.effects.split(x, top_db=35, frame_length=1024, hop_length=256)
    gaps = [(a_end + b_start) // 2 for (_, a_end), (b_start, _) in zip(voiced[:-1], voiced[1:])]
    if not gaps:
        cut = int(len(x) * target_frac)
    else:
        cut = min(gaps, key=lambda g: abs(g - target_frac * len(x)))
    fade = int(0.02 * SR)
    y = x[:cut].copy()
    y[-fade:] *= np.linspace(1, 0, fade)
    return y, cut / SR


def match_rms(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    y = y * (np.sqrt(np.mean(ref ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))
    peak = np.max(np.abs(y))
    return y * (0.95 / peak) if peak > 0.95 else y


# ── examples ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-base-url", default=DEFAULT_AUDIO_BASE_URL)
    base = ap.parse_args().audio_base_url.rstrip("/")

    ds = load_dataset("davidguzmanr/CSS10-Multilingual-LJSpeech", "English", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    by_id = {Path(e["audio"]["path"]).stem: e for e in ds}

    def load(utt: str) -> tuple[np.ndarray, str]:
        e = by_id[utt]
        x, sr = sf.read(io.BytesIO(e["audio"]["bytes"]), dtype="float32")
        assert sr == SR, sr
        return x, e["text"]

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def emit(name, audio, ref, shown_text, nat, acc, what, why_nat, why_acc, source):
        sf.write(AUDIO_DIR / name, match_rms(audio, ref), SR, subtype="PCM_16")
        rows.append({
            "file": name, "source_utt": source, "duration_s": round(len(audio) / SR, 2),
            "reference_text_shown": shown_text, "naturalness": nat, "accuracy": acc,
            "manipulation": what, "why_naturalness": why_nat, "why_accuracy": why_acc,
        })

    # 1-4  naturalness ladder: one sentence, words always correct
    x, text = load("LJ001-0009")
    emit("01_nat5_acc5.wav", x, x, text, 5, 5, "original recording",
         "A real person reading: smooth rhythm, natural intonation, no artifacts.",
         "Every word matches the text.", "LJ001-0009")
    emit("02_nat4_acc5.wav", griffin_lim(x, 100), x, text, 4, 5,
         "Griffin-Lim phase reconstruction, 100 iterations",
         "Still sounds human, but slightly hollow or phasey if you listen closely.",
         "Every word matches the text.", "LJ001-0009")
    emit("03_nat3_acc5.wav", griffin_lim(x, 16), x, text, 3, 5,
         "Griffin-Lim phase reconstruction, 16 iterations",
         "Clearly synthetic: metallic, buzzy texture throughout, but easy to follow.",
         "Every word is still there and understandable, so accuracy stays at 5.", "LJ001-0009")
    emit("04_nat2_acc5.wav", robotize(x), x, text, 2, 5,
         "robotization (all STFT phases set to zero)",
         "Monotone robot voice with constant buzz; tiring to listen to.",
         "Unnatural, but every word can be made out, so accuracy is still 5.", "LJ001-0009")

    # 5-7  accuracy ladder: natural audio, mismatch with the text shown
    rows_utt = _pick(by_id, "the books so ornamented are amongst the most delightful works of art")
    x, text = load(rows_utt)
    shown = text.replace("delightful", "beautiful")
    assert shown != text
    emit("05_nat5_acc3.wav", x, x, shown, 5, 3,
         "natural audio; text shown has one word changed (audio: 'delightful', text: 'beautiful')",
         "Natural recording, so naturalness is 5 even though a word is wrong.",
         "One word differs from the text.", rows_utt)

    utt = _pick(by_id, "He left Newgate utterly corrupted")
    x, text = load(utt)
    y, cut_s = truncate_at_pause(x, 0.6)
    emit("06_nat5_acc2.wav", y, x, text, 5, 2,
         f"natural audio cut off at the pause at {cut_s:.2f} s of {len(x)/SR:.2f} s",
         "The voice itself is natural; a cut-off is an accuracy problem, not a naturalness one.",
         "The end of the sentence is missing (several words).", utt)

    utt_audio = _pick(by_id, "This was the pernicious plan of governing by prisoners")
    utt_text = _pick(by_id, "Again, there was no regular allowance of fuel")
    x, _ = load(utt_audio)
    _, wrong_text = load(utt_text)
    emit("07_nat5_acc1.wav", x, x, wrong_text, 5, 1,
         "natural audio of a different sentence than the one shown",
         "Natural recording, so naturalness is 5.",
         "The audio is a completely different sentence.", f"{utt_audio} (text from {utt_text})")

    # 8  combined: both axes degraded
    utt = _pick(by_id, "a week's allowance at a time, was abolished")
    x, text = load(utt)
    shown = text.replace("daily", "weekly")
    assert shown != text
    emit("08_nat3_acc3.wav", griffin_lim(x, 16), x, shown, 3, 3,
         "Griffin-Lim 16 iterations AND one word changed in the text (audio: 'daily', text: 'weekly')",
         "Metallic and synthetic, same as example 3.",
         "One word differs from the text. Rate each axis on its own.", utt)

    with open(HERE / "examples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tasks = [{"data": {"task_uid": f"tutorial-{i + 1:02d}",
                       "reference_text": r["reference_text_shown"],
                       "audio_url": f"{base}/{r['file']}",
                       "expected_naturalness": r["naturalness"],
                       "expected_accuracy": r["accuracy"]}}
             for i, r in enumerate(rows)]
    (HERE / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "labeling_config.xml").write_text(LABELING_CONFIG, encoding="utf-8")

    for r in rows:
        print(f"{r['file']:20s} N{r['naturalness']} A{r['accuracy']} {r['duration_s']:5.2f}s  {r['manipulation']}")


def _pick(by_id: dict, prefix: str) -> str:
    hits = [u for u, e in by_id.items() if e["text"].startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


LABELING_CONFIG = """\
<View>
  <Header value="Listen to the clip, then answer both questions." />

  <View style="background:#f5f5f5;border-left:4px solid #aaa;padding:10px 16px;margin:12px 0;border-radius:4px;">
    <Header value="Reference text" size="5" />
    <Text name="ref_text" value="$reference_text" />
  </View>

  <Audio name="audio" value="$audio_url" />

  <Header value="1. Naturalness: how natural does the voice sound?" size="4" />
  <Text name="nat_help" value="Judge rhythm, intonation and audio quality. Ignore whether the words match the text." />
  <Choices name="naturalness" toName="audio" choice="single" showInline="true" required="true">
    <Choice value="1 - Completely unnatural" hint="Severe distortion or garbling; barely sounds like speech." />
    <Choice value="2 - Unnatural" hint="Strong artifacts or robotic throughout; tiring to listen to." />
    <Choice value="3 - Somewhat unnatural" hint="Clearly synthetic (metallic, buzzy, flat), but easy to follow." />
    <Choice value="4 - Mostly natural" hint="Small artifacts or slightly odd rhythm, noticeable only when listening closely." />
    <Choice value="5 - Completely natural" hint="Sounds like a real person speaking; no artifacts." />
  </Choices>

  <Header value="2. Accuracy: does the audio say exactly the reference text?" size="4" />
  <Text name="acc_help" value="Judge only which words are spoken. Ignore how natural or synthetic the voice sounds." />
  <Choices name="accuracy" toName="audio" choice="single" showInline="true" required="true">
    <Choice value="1 - Mostly wrong" hint="Most of the text missing or wrong, a different sentence or language, or unintelligible." />
    <Choice value="2 - Several errors" hint="Several words wrong or missing, or part of the sentence cut off." />
    <Choice value="3 - One error" hint="One word wrong, missing, added, or mispronounced so it changes or hides the meaning." />
    <Choice value="4 - Minor slip" hint="A word slightly mispronounced, but the meaning is clear." />
    <Choice value="5 - Every word correct" hint="Nothing missing, added or mispronounced." />
  </Choices>
</View>
"""

if __name__ == "__main__":
    main()
