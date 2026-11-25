"""Build the HumanSignal (Label Studio) survey for each language of the listening study.

One project per language. Every task is one clip: the reference text, the audio and the two 1-5 questions
(naturalness, intelligibility) with the instructions, question wording and hover hints of the annotator
tutorial (listening-study/tutorial/build_tutorial.py); the two text constants below are copied from there
verbatim and must stay identical, so change them in both places.

Stimuli come from the Hugging Face dataset davidguzmanr/multilingual-mos-listening-test. Its root clips.csv
has one row per clip (language, system, dataset, clip_id, text, duration, URL) with URLs pinned to the audio
commit, so the surveys keep working if clips are regenerated later under the same names. A copy is kept here
as clips.csv; --refresh downloads it again from the pinned commit.

Per language this writes surveys/<Language>/:
    labeling_config.xml   paste into the project's labeling interface (same for every language)
    tasks.json            import into the project; blind (no system name in what annotators see)
    tracking.csv          task_uid -> clip_id, dataset, system, URL; join with the exported annotations

    python listening-study/surveys/build_surveys.py                       # all three languages
    python listening-study/surveys/build_surveys.py --languages Yoruba
    python listening-study/surveys/build_surveys.py --refresh             # re-download clips.csv first
    python listening-study/surveys/build_surveys.py --api-key TOKEN       # also create the projects and import the tasks
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
HF_REPO = "davidguzmanr/multilingual-mos-listening-test"
AUDIO_SHA = "a6ea421b24215d966c5e596673c4d45898938175"   # commit with the audio; every URL must point here
CLIPS_SHA = "0db14352e1d4c90c29c1db120b650154d0cc24e5"   # commit whose clips.csv is pinned to AUDIO_SHA
CLIPS_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{CLIPS_SHA}/clips.csv"
LANGUAGES = {"Yoruba": "yo", "Swahili": "sw", "Hausa": "ha"}
SEED = 42
DEFAULT_BASE_URL = "https://app.humansignal.com"

# ---------------------------------------------------------------------------------------------------------
# Survey text: copied verbatim from listening-study/tutorial/build_tutorial.py (keep in sync)
# ---------------------------------------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------------------------------------


def load_clips(refresh: bool) -> list[dict]:
    """Read clips.csv (download it from the pinned commit if missing or --refresh) and check it."""
    path = HERE / "clips.csv"
    if refresh or not path.exists():
        print(f"downloading {CLIPS_URL}")
        with urllib.request.urlopen(CLIPS_URL, timeout=60) as r:
            path.write_bytes(r.read())
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"language", "system", "dataset", "clip_id", "text", "duration_s", "url"}
    assert need <= set(rows[0]), f"clips.csv is missing columns: {need - set(rows[0])}"
    bad = [r["url"] for r in rows if f"/resolve/{AUDIO_SHA}/" not in r["url"]]
    assert not bad, f"{len(bad)} URLs are not pinned to {AUDIO_SHA[:8]}, e.g. {bad[0]}"
    assert all(r["text"].strip() for r in rows), "empty reference text"
    for lang in LANGUAGES:
        sub = [r for r in rows if r["language"] == lang]
        per_sentence = Counter(r["clip_id"] for r in sub)
        n_sys = len({r["system"] for r in sub})
        assert set(per_sentence.values()) == {n_sys}, f"{lang}: not every sentence has all {n_sys} systems"
        texts = {}
        for r in sub:
            assert texts.setdefault(r["clip_id"], r["text"]) == r["text"], f"{lang} {r['clip_id']}: text differs between systems"
    return rows


def build_language(lang: str, rows: list[dict], out_dir: Path) -> list[dict]:
    """Shuffle this language's clips with a fixed seed and write config, tasks and tracking CSV."""
    code = LANGUAGES[lang]
    clips = sorted((r for r in rows if r["language"] == lang), key=lambda r: (r["clip_id"], r["system"]))
    random.Random(SEED).shuffle(clips)
    instructions = INSTRUCTIONS_HTML.format(language=lang)
    tasks, tracking = [], []
    for i, r in enumerate(clips, 1):
        uid = f"{code}-{i:04d}"
        tasks.append({"data": {"task_uid": uid, "instructions_html": instructions,
                               "reference_text": r["text"], "audio_url": r["url"]}})
        tracking.append({"task_uid": uid, "clip_id": r["clip_id"], "dataset": r["dataset"], "system": r["system"],
                         "duration_s": r["duration_s"], "audio_url": r["url"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labeling_config.xml").write_text(LABELING_CONFIG, encoding="utf-8")
    (out_dir / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "tracking.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tracking[0]))
        w.writeheader()
        w.writerows(tracking)
    systems = Counter(r["system"] for r in clips)
    minutes = sum(float(r["duration_s"]) for r in clips) / 60
    print(f"{lang}: {len(tasks)} tasks, {len({r['clip_id'] for r in clips})} sentences x {len(systems)} systems, "
          f"{minutes:.0f} min of audio -> {out_dir.relative_to(HERE.parent.parent)}/")
    return tasks


def create_project_and_import(lang: str, tasks: list[dict], api_key: str, base_url: str) -> None:
    """Create a HumanSignal project with the labeling config and import the tasks (same API as the pilot)."""
    import requests

    base = base_url.rstrip("/")
    headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}
    r = requests.post(f"{base}/api/projects", headers=headers, timeout=30,
                      json={"title": f"Multilingual MOS - {lang}", "label_config": LABELING_CONFIG,
                            "description": f"Blind two-question MOS listening test, {lang}. {len(tasks)} clips, "
                                           f"audio pinned to {HF_REPO}@{AUDIO_SHA[:8]}, shuffle seed {SEED}."})
    r.raise_for_status()
    pid = r.json()["id"]
    r = requests.post(f"{base}/api/projects/{pid}/import", headers=headers, json=tasks, timeout=300)
    r.raise_for_status()
    print(f"  {lang}: project {base}/projects/{pid}/ created, {r.json().get('task_count', len(tasks))} tasks imported")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--languages", nargs="+", choices=list(LANGUAGES), default=list(LANGUAGES))
    ap.add_argument("--refresh", action="store_true", help="re-download clips.csv from the pinned commit")
    ap.add_argument("--api-key", help="HumanSignal API token: also create one project per language and import the tasks")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    a = ap.parse_args()
    rows = load_clips(a.refresh)
    print(f"clips.csv: {len(rows)} stimuli, audio pinned to {HF_REPO}@{AUDIO_SHA[:8]}")
    for lang in a.languages:
        tasks = build_language(lang, rows, HERE / lang)
        if a.api_key:
            create_project_and_import(lang, tasks, a.api_key, a.base_url)


if __name__ == "__main__":
    sys.exit(main())
