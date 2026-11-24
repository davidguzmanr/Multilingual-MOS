"""
Build the English annotator tutorial for the two-axis listening study.

Source: LJSpeech (via davidguzmanr/CSS10-Multilingual-LJSpeech, "English", test split) and Coqui TTS
models trained on LJSpeech, so every clip is the same speaker:

  naturalness ladder      same sentence, from the original recording down to a TTS that falls apart
                          (original > Neural-HMM > robotization > VITS with far too much noise); as the
                          voice degrades, words also get harder to make out, so intelligibility drops too
  intelligibility ladder  a repeated word plus a mispronounced one (3), natural audio cut off at a pause (2),
                          a different sentence (1)
  combined                older TTS that skips one word and mispronounces another, rated on both axes

Naturalness 4 and intelligibility 4 have no dedicated clip: annotators place them between the examples.
Mispronunciations are made the way TTS systems make them: the G2P output of a word is replaced
before synthesis (e.g. "wound" /wˈuːnd/ -> /wˈaʊnd/), everything else is the model's own output.
All clips are RMS-matched to the natural recording of the same sentence so loudness is not a cue.

Outputs (next to this script):
  audios/*.wav            22.05 kHz, 16-bit mono
  examples.csv            target ratings and what to say about each clip
  tasks.json              Label Studio tasks (audio served from the dev/tutorial branch)
  labeling_config.xml     two-axis Label Studio interface (also the draft for the real survey)
  index.html              local listening page (open in a browser) to review the clips before the survey

Needs coqui-tts, phonemizer-fork and espeakng-loader (no espeak-ng binary required) and a GPU:

    module load miniconda/3 && conda activate Multilingual-MOS
    python listening-study/tutorial/build_tutorial.py [--audio-base-url URL]
    python listening-study/tutorial/build_tutorial.py --preview-only   # only rewrite index.html
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
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


# ── TTS ─────────────────────────────────────────────────────────────────────

# (correct phonemes, wrong phonemes) applied to the espeak output before synthesis
G2P_OVERRIDES: list[tuple[str, str]] = []


def _use_bundled_espeak() -> None:
    """Coqui's ESpeak phonemizer shells out to an `espeak-ng` binary. Route it through the espeak-ng
    library shipped with espeakng-loader instead, and apply G2P_OVERRIDES to simulate G2P errors."""
    import espeakng_loader
    import TTS.tts.utils.text.phonemizers.espeak_wrapper as ew
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer.separator import Separator

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    backends: dict[str, EspeakBackend] = {}

    def phonemize_espeak(self, text, separator="|", *, tie=False):
        if self._language not in backends:
            backends[self._language] = EspeakBackend(self._language, with_stress=True, tie=tie,
                                                     language_switch="remove-flags")
        ph = backends[self._language].phonemize([text], separator=Separator(phone=separator, word=" "),
                                                strip=True)[0]
        for good, bad in G2P_OVERRIDES:
            ph = ph.replace(good, bad)
        return ph

    ew._DEF_ESPEAK_LIB, ew._DEF_ESPEAK_VER = "espeak-ng", "1.52.0"
    ew.get_espeakng_version = lambda: "1.52.0"
    ew._is_tool = lambda name: name == "espeak-ng"
    ew.ESpeak.phonemize_espeak = phonemize_espeak
    ew.ESpeak.supported_languages = staticmethod(lambda: {"en-us": "English_(America)"})


def load_tts(name: str):
    from TTS.api import TTS
    tts = TTS(f"tts_models/en/ljspeech/{name}").to("cuda" if torch.cuda.is_available() else "cpu")
    assert tts.synthesizer.output_sample_rate == SR, tts.synthesizer.output_sample_rate
    return tts


def synth(tts, text: str, *, overrides: list[tuple[str, str]] = (), noise_scale: float | None = None) -> np.ndarray:
    """Synthesize with a fixed seed. `overrides` mispronounce words, `noise_scale` (VITS) sets the
    sampling noise of the prior (default 0.667; ~3 makes the model fall apart into garbled speech)."""
    model = tts.synthesizer.tts_model
    default_noise = getattr(model, "inference_noise_scale", None)
    if noise_scale is not None:
        model.inference_noise_scale = noise_scale
    G2P_OVERRIDES[:] = list(overrides)
    torch.manual_seed(SEED)
    try:
        return np.asarray(tts.tts(text), dtype=np.float32)
    finally:
        G2P_OVERRIDES.clear()
        if noise_scale is not None:
            model.inference_noise_scale = default_noise


# ── signal degradations ─────────────────────────────────────────────────────

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
    ap.add_argument("--preview-only", action="store_true",
                    help="only rewrite index.html from the existing examples.csv and audios/")
    args = ap.parse_args()
    if args.preview_only:
        write_preview()
        return
    base = args.audio_base_url.rstrip("/")

    ds = load_dataset("davidguzmanr/CSS10-Multilingual-LJSpeech", "English", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    by_id = {Path(e["audio"]["path"]).stem: e for e in ds}

    def load(utt: str) -> tuple[np.ndarray, str]:
        e = by_id[utt]
        x, sr = sf.read(io.BytesIO(e["audio"]["bytes"]), dtype="float32")
        assert sr == SR, sr
        return x, e["text"]

    _use_bundled_espeak()
    vits, hmm = load_tts("vits"), load_tts("neural_hmm")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for old in AUDIO_DIR.glob("*.wav"):
        old.unlink()
    rows: list[dict] = []

    def emit(nat, intel, audio, ref, shown_text, what, why_nat, why_int, source):
        name = f"{len(rows) + 1:02d}_nat{nat}_int{intel}.wav"
        sf.write(AUDIO_DIR / name, match_rms(audio, ref), SR, subtype="PCM_16")
        rows.append({
            "file": name, "source_utt": source, "duration_s": round(len(audio) / SR, 2),
            "reference_text_shown": shown_text, "naturalness": nat, "intelligibility": intel,
            "manipulation": what, "why_naturalness": why_nat, "why_intelligibility": why_int,
        })

    # 1-4  naturalness ladder: one sentence, from the real recording down to a TTS that falls apart
    x, text = load("LJ001-0009")
    emit(5, 5, x, x, text, "original recording",
         "A real person reading: smooth rhythm, natural intonation, no artifacts.",
         "Every word matches the text.", "LJ001-0009")
    emit(3, 4, synth(hmm, text), x, text, "Neural-HMM (Coqui, trained on LJSpeech), an older statistical TTS",
         "Clearly synthetic: flat, monotone intonation and a muffled, buzzy voice quality, but easy to follow.",
         "All the words are there, but the muffled sound blurs a word or two (e.g. 'purpose'); "
         "the sentence is still easy to follow.", "LJ001-0009")
    emit(2, 3, robotize(x), x, text, "robotization (all STFT phases set to zero)",
         "Monotone robot voice with constant buzz; tiring to listen to.",
         "The words are the right ones, but the buzz makes several of them hard to make out; "
         "with the text in front of you the sentence can still be followed.", "LJ001-0009")
    emit(1, 2, synth(vits, text, noise_scale=3.0), x, text,
         "VITS sampled with far too much noise (noise_scale 3.0 instead of 0.667): the model falls apart",
         "Garbled, slurred and distorted throughout; barely sounds like a person speaking.",
         "Only fragments can be made out, such as 'may be considered as' and 'movable types'; "
         "the start is lost and several words are wrong.", "LJ001-0009")

    # 5-7  intelligibility ladder
    utt = _pick(by_id, "Two of these photographs were taken from the rear of Walker's house")
    x, text = load(utt)
    spoken = text.replace("these", "these these", 1)
    assert spoken != text
    emit(4, 3, synth(vits, spoken, overrides=[("hˈaʊs", "hˈuːs")]), x, text,
         "VITS (Coqui, trained on LJSpeech) repeating a word ('these these') and mispronouncing "
         "'house' as 'hoose'",
         "A good modern TTS: sounds human, but the rhythm is a little too even and smooth.",
         "One word is repeated and another is mispronounced: two errors. The sentence can still be "
         "followed with the text.", utt)

    utt = _pick(by_id, "He left Newgate utterly corrupted")
    x, text = load(utt)
    y, cut_s = truncate_at_pause(x, 0.6)
    emit(5, 2, y, x, text,
         f"natural audio cut off at the pause at {cut_s:.2f} s of {len(x)/SR:.2f} s",
         "The voice itself is natural; a cut-off is an intelligibility problem, not a naturalness one.",
         "The end of the sentence is missing (several words).", utt)

    utt_audio = _pick(by_id, "This was the pernicious plan of governing by prisoners")
    utt_text = _pick(by_id, "Again, there was no regular allowance of fuel")
    x, _ = load(utt_audio)
    _, wrong_text = load(utt_text)
    emit(5, 1, x, x, wrong_text, "natural audio of a different sentence than the one shown",
         "Natural recording, so naturalness is 5.",
         "The audio is a completely different sentence.", f"{utt_audio} (text from {utt_text})")

    # 8  combined: older TTS that skips one word and mispronounces another
    utt = _pick(by_id, "Sick women and their children remained in the ordinary wards")
    x, text = load(utt)
    spoken = text.replace("ordinary ", "")
    assert spoken != text
    emit(3, 3, synth(hmm, spoken, overrides=[("wˈɔːɹdz", "wˈɑːɹdz")]), x, text,
         "Neural-HMM (same as clip 2) skipping 'ordinary' and saying 'wards' as 'wahrds' (like 'cards')",
         "Same flat, buzzy synthetic voice as clip 2.",
         "One word is missing and another is mispronounced, but the sentence can be followed. "
         "Rate each question on its own.", utt)

    with open(HERE / "examples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    instructions = INSTRUCTIONS_HTML.format(language="English")
    tasks = [{"data": {"task_uid": f"tutorial-{i + 1:02d}",
                       "instructions_html": instructions,
                       "reference_text": r["reference_text_shown"],
                       "audio_url": f"{base}/{r['file']}",
                       "expected_naturalness": r["naturalness"],
                       "expected_intelligibility": r["intelligibility"]}}
             for i, r in enumerate(rows)]
    (HERE / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "labeling_config.xml").write_text(LABELING_CONFIG, encoding="utf-8")
    write_preview()

    for r in rows:
        print(f"{r['file']:20s} N{r['naturalness']} I{r['intelligibility']} {r['duration_s']:5.2f}s  {r['manipulation']}")


def write_preview() -> None:
    """Write index.html: one row per clip with a player, target scores, the text shown and the notes.
    Audio links carry ?v=<mtime> so the browser reloads clips after a rebuild instead of using its cache."""
    with open(HERE / "examples.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    e = html.escape
    body = []
    for i, r in enumerate(rows, 1):
        wav = AUDIO_DIR / r["file"]
        src = f"audios/{r['file']}?v={wav.stat().st_mtime_ns}" if wav.exists() else ""
        player = f'<audio controls preload="none" src="{e(src)}"></audio>' if src else '<span class=miss>missing</span>'
        body.append(
            f"<tr><td class=n>{i}</td>"
            f"<td>{player}<div class=lab>{e(r['file'])} · {float(r['duration_s']):.1f} s · {e(r['source_utt'])}</div></td>"
            f"<td class=score>{e(r['naturalness'])}</td><td class=score>{e(r['intelligibility'])}</td>"
            f"<td class=txt>{e(r['reference_text_shown'])}</td>"
            f"<td>{e(r['manipulation'])}</td>"
            f"<td><b>N:</b> {e(r['why_naturalness'])}<br><b>I:</b> {e(r['why_intelligibility'])}</td></tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Annotator tutorial clips</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}} h1{{margin:0 0 4px}} .sub{{color:#666;margin-bottom:16px}}
table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px 8px;vertical-align:top;text-align:left;font-size:14px}}
thead th{{position:sticky;top:0;background:#fff;z-index:1}} tr:nth-child(even) td{{background:#f3f3f3}}
td.n{{color:#666}} td.score{{text-align:center;font-weight:600;font-size:16px}} td.txt{{max-width:320px;font-style:italic}}
.lab{{font-size:11px;color:#666;margin-top:2px}} audio{{width:240px;height:32px}} .miss{{color:#b00}}
</style></head><body>
<h1>Annotator tutorial clips</h1>
<div class=sub>{len(rows)} clips. N = naturalness, I = intelligibility (target scores). The text column is what annotators
see on screen. Regenerate with <code>build_tutorial.py</code>, or <code>build_tutorial.py --preview-only</code> to refresh this page.</div>
<table><thead><tr><th>#</th><th>audio</th><th>N</th><th>I</th><th>text shown</th><th>what was done</th><th>why</th></tr></thead>
<tbody>
{chr(10).join(body)}
</tbody></table></body></html>
"""
    (HERE / "index.html").write_text(page, encoding="utf-8")


def _pick(by_id: dict, prefix: str) -> str:
    hits = [u for u, e in by_id.items() if e["text"].startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


INSTRUCTIONS_HTML = (
    "<p>You will hear a short speech clip in {language}, which may be a real recording or "
    "computer-generated, together with the text it is supposed to say. Listen to the whole clip, "
    "then answer two separate questions. <strong>Naturalness</strong> is about how human-like the voice "
    "sounds: its rhythm, intonation and audio quality. <strong>Intelligibility</strong> is about the words: whether "
    "the audio says exactly the text shown and you can make out every word, with nothing missing, added, "
    "repeated or mispronounced. Answer each question on its own: a natural voice can say the wrong words, "
    "and a poor voice can make the right words hard to catch.</p>"
)

LABELING_CONFIG = """\
<View>
  <HyperText name="instructions" value="$instructions_html" />

  <View style="background:#f5f5f5;border-left:4px solid #aaa;padding:10px 16px;margin:12px 0;border-radius:4px;">
    <Header value="Reference text" size="5" />
    <Text name="ref_text" value="$reference_text" />
  </View>

  <Audio name="audio" value="$audio_url" />

  <Header value="1. Naturalness: does the voice sound like a real person speaking?" size="4" />
  <Choices name="naturalness" toName="audio" choice="single" showInline="true" required="true">
    <Choice value="1 - Completely unnatural" hint="Severe distortion or garbling; barely sounds like speech." />
    <Choice value="2 - Unnatural" hint="Strong artifacts or robotic throughout; tiring to listen to." />
    <Choice value="3 - Somewhat unnatural" hint="Clearly synthetic (metallic, buzzy, flat), but easy to follow." />
    <Choice value="4 - Mostly natural" hint="Small artifacts or slightly odd rhythm, noticeable only when listening closely." />
    <Choice value="5 - Completely natural" hint="Sounds like a real person speaking; no artifacts." />
  </Choices>

  <Header value="2. Intelligibility: does the audio say exactly the reference text, word for word?" size="4" />
  <Choices name="intelligibility" toName="audio" choice="single" showInline="true" required="true">
    <Choice value="1 - Mostly wrong" hint="Most of the text missing or wrong, a different sentence or language, or unintelligible." />
    <Choice value="2 - Many errors" hint="Many words wrong, missing or mispronounced, or part of the sentence cut off; hard to follow." />
    <Choice value="3 - A few errors" hint="Two or three words wrong, missing, added, mispronounced or hard to make out; the sentence can still be followed." />
    <Choice value="4 - Minor slip" hint="One word wrong, missing, mispronounced or hard to make out; the sentence is still easy to follow." />
    <Choice value="5 - Every word correct" hint="Nothing missing, added or mispronounced." />
  </Choices>
</View>
"""

if __name__ == "__main__":
    main()
