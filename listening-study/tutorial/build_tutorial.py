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

Needs coqui-tts, phonemizer-fork and espeakng-loader (no espeak-ng binary required) and a GPU. The seed is
fixed, but CPU and GPU sampling draw different random numbers, so a rebuild on a different device produces
different (audibly different) synthesized clips. The committed wavs are the approved ones: to change only the
wording (instructions, questions, hints), edit the text constants and run with --text-only, which rewrites
tasks.json, labeling_config.xml and index.html from the existing examples.csv and audios/ without synthesis.

    module load miniconda/3 && conda activate Multilingual-MOS
    python listening-study/tutorial/build_tutorial.py [--audio-base-url URL]
    python listening-study/tutorial/build_tutorial.py --text-only      # no synthesis; rewrite text outputs
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
    ap.add_argument("--text-only", action="store_true",
                    help="no synthesis: rewrite tasks.json, labeling_config.xml and index.html from examples.csv")
    ap.add_argument("--preview-only", action="store_true",
                    help="only rewrite index.html from the existing examples.csv and audios/")
    args = ap.parse_args()
    base = args.audio_base_url.rstrip("/")
    if args.preview_only:
        write_preview()
        return
    if args.text_only:
        with open(HERE / "examples.csv", encoding="utf-8") as f:
            write_outputs(list(csv.DictReader(f)), base)
        return

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
         "Clearly synthetic: flat, monotone intonation and a muffled, buzzy voice quality, ",
         "All the words are the right ones, but the muffled sound makes one word ('purpose') hard to "
         "make out: a small problem, almost every word clear.", "LJ001-0009")
    emit(2, 3, robotize(x), x, text, "robotization (all STFT phases set to zero)",
         "Monotone robot voice with constant buzz; tiring to listen to.",
         "The words are the right ones, but the buzz makes some of them unclear; you can still follow the "
         "sentence: most words.", "LJ001-0009")
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
         "One word is repeated and another is mispronounced, but you can still follow the sentence: most words.", utt)

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
         "One word is missing and another is mispronounced, but you can still follow the sentence: most words. "
         "Rate each question on its own.", utt)

    with open(HERE / "examples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_outputs(rows, base)


def write_outputs(rows: list[dict], base: str) -> None:
    """Write tasks.json, labeling_config.xml and index.html from the example rows (no synthesis)."""
    instructions = INSTRUCTIONS_HTML.format(language="English")
    tasks = [{"data": {"task_uid": f"tutorial-{i + 1:02d}",
                       "instructions_html": instructions,
                       "reference_text": r["reference_text_shown"],
                       "audio_url": f"{base}/{r['file']}",
                       "expected_naturalness": int(r["naturalness"]),
                       "expected_intelligibility": int(r["intelligibility"])}}
             for i, r in enumerate(rows)]
    (HERE / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "labeling_config.xml").write_text(LABELING_CONFIG, encoding="utf-8")
    write_preview()

    for r in rows:
        print(f"{r['file']:20s} N{r['naturalness']} I{r['intelligibility']} {float(r['duration_s']):5.2f}s  {r['manipulation']}")


def write_preview() -> None:
    """Write index.html: a mock of one survey task exactly as annotators see it (instructions, reference
    text, player, both questions with their 1-5 options and hints), then one row per clip with a player,
    target scores, the text shown and the notes. Audio links carry ?v=<mtime> so the browser reloads clips
    after a rebuild instead of using its cache."""
    import xml.etree.ElementTree as ET

    with open(HERE / "examples.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    e = html.escape

    def audio_src(file: str) -> str:
        wav = AUDIO_DIR / file
        return f"audios/{file}?v={wav.stat().st_mtime_ns}" if wav.exists() else ""

    # survey mock, built from the same constants that produce labeling_config.xml and tasks.json
    first = rows[0]
    questions = []
    config = ET.fromstring(LABELING_CONFIG)
    question_headers = [h for h in config.iter("Header") if h.get("size") == "4"]
    choice_blocks = list(config.iter("Choices"))
    assert len(question_headers) == len(choice_blocks) == 2, (len(question_headers), len(choice_blocks))
    for header, choices in zip(question_headers, choice_blocks):
        opts = "".join(
            f'<label title="{e(c.get("hint", ""))}"><input type="radio" name="{e(choices.get("name"))}"> '
            f'{e(c.get("value"))}<span class=hint>{e(c.get("hint", ""))}</span></label>'
            for c in choices.findall("Choice"))
        questions.append(f'<h3>{e(header.get("value"))}</h3><div class=opts>{opts}</div>')
    mock = f"""<section class=mock>
<h2>What annotators see</h2>
<div class=sub>One task rendered from <code>labeling_config.xml</code> and <code>tasks.json</code> (clip 1).
Hints appear in HumanSignal when hovering an option; here they are printed under each option.</div>
<div class=task>
{INSTRUCTIONS_HTML.format(language="English")}
<div class=ref><div class=refh>Reference text</div>{e(first["reference_text_shown"])}</div>
<audio controls preload="none" src="{e(audio_src(first["file"]))}"></audio>
{"".join(questions)}
</div></section>
"""

    body = []
    for i, r in enumerate(rows, 1):
        src = audio_src(r["file"])
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
.mock{{margin-bottom:28px}} .mock h2{{margin:0 0 4px}} .task{{max-width:900px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:16px 20px;font-size:14px}}
.task p{{margin:0 0 12px}} .task audio{{width:100%;margin:8px 0 4px}}
.ref{{background:#f5f5f5;border-left:4px solid #aaa;padding:10px 16px;margin:12px 0;border-radius:4px}} .refh{{font-weight:600;font-size:13px;margin-bottom:4px}}
.task h3{{font-size:15px;margin:16px 0 6px}} .opts{{display:flex;gap:8px;flex-wrap:wrap}}
.opts label{{flex:1 1 150px;border:1px solid #ddd;border-radius:4px;padding:6px 8px;background:#fafafa;cursor:pointer}}
.opts label:hover{{background:#eef}} .hint{{display:block;font-size:11px;color:#666;margin-top:4px}}
</style></head><body>
<h1>Annotator tutorial clips</h1>
{mock}
<h2>Tutorial clips</h2>
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
    "then answer two separate questions.</p>"
    "<p><strong>1. Naturalness</strong> is about how the voice sounds.</p>"
    "<ul>"
    "<li>Does it sound like a real person speaking?</li>"
    "<li>Judge only the sound, not whether the words are correct.</li>"
    "</ul>"
    "<p><strong>2. Intelligibility</strong> is about the words.</p>"
    "<ul>"
    "<li>Listen while reading the text: are all the words clear and the same as in the text? Look out for "
    "words that are unclear, wrong, missing, added or repeated, or cut off.</li>"
    "<li>Listen to the pronunciation: a word that is pronounced as a different word, even a similar-sounding "
    "one, is a wrong word.</li>"
    "<li>A word you cannot make out is an error, even if you can guess it from the text.</li>"
    "</ul>"
    "<p>Answer each question on its own: a natural voice can say the wrong words, and a poor voice can say "
    "the right words.</p>"
)

LABELING_CONFIG = """\
<View>
  <!-- Hide the auto-assigned hotkey hints ([1]..[0] superscripts) next to each option: they clash with
       the 1-5 scale values. The number keys still work; only the hint is hidden. -->
  <Style>
    .lsf-choice sup, .choice sup, .lsf-hint, sup.hint { display: none !important; }
  </Style>
  <HyperText name="instructions" value="$instructions_html" />

  <View style="background:#f5f5f5;border-left:4px solid #aaa;padding:10px 16px;margin:12px 0;border-radius:4px;">
    <Header value="Reference text" size="5" />
    <Text name="ref_text" value="$reference_text" />
  </View>

  <Audio name="audio" value="$audio_url" />

  <Header value="1. Naturalness: how natural does the voice sound, like a real person speaking? Judge only how it sounds, not whether the words are correct." size="4" />
  <Choices name="naturalness" toName="audio" choice="single" showInline="true" required="true">
    <Choice value="1 - Completely unnatural" hint="Severe distortion or garbling; barely sounds like speech." />
    <Choice value="2 - Unnatural" hint="Strong artifacts or very unnatural rhythm and intonation throughout; clearly not a person." />
    <Choice value="3 - Somewhat unnatural" hint="You can tell it is not a person, but it is not unpleasant to listen to." />
    <Choice value="4 - Mostly natural" hint="Small artifacts or slightly odd rhythm, noticeable only when listening closely." />
    <Choice value="5 - Completely natural" hint="Sounds like a real person speaking; no artifacts." />
  </Choices>

  <Header value="2. Intelligibility: are all the words clear and the same as in the text? Look out for words that are unclear, wrong, missing, added or repeated, or cut off." size="4" />
  <Choices name="intelligibility" toName="audio" choice="single" showInline="true" required="true">
    <Choice value="1 - Hardly any" hint="Most of the text is wrong or missing, it is a different sentence or a different language, or no words can be made out." />
    <Choice value="2 - Only part of it" hint="Many words are unclear, wrong, missing or repeated, or a part of the text is skipped or cut off. You cannot follow the sentence without reading the text." />
    <Choice value="3 - Most words" hint="Some words are unclear, wrong, missing, added or repeated, but you can still follow the sentence." />
    <Choice value="4 - Almost every word" hint="Nearly everything is clear and as in the text. There is a small problem that hardly gets in the way." />
    <Choice value="5 - Every word" hint="All words are clear and exactly the words in the text. Nothing is missing, added or changed." />
  </Choices>
</View>
"""

if __name__ == "__main__":
    main()
