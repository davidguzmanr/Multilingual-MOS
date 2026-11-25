# HumanSignal surveys per language

One HumanSignal (Label Studio) project per language. Each task is one clip: the reference text, the audio player and
the two 1-5 questions (naturalness, intelligibility). Instructions, question wording and hover hints are exactly those
of the annotator tutorial in [../tutorial/](../tutorial/) (`dev/tutorial` branch); the two text constants in
`build_surveys.py` are copies of the tutorial's and must be changed in both places.

| Language | Tasks | Sentences | Systems (incl. ground truth) | Audio | Datasets |
|---|---:|---:|---:|---:|---|
| Yoruba | 306 | 34 | 9 | 35 min | Common Voice, FLEURS, WaxalNLP |
| Swahili | 306 | 34 | 9 | 38 min | Common Voice, FLEURS, WaxalNLP |
| Hausa | 304 | 38 | 8 | 37 min | Common Voice, FLEURS, WaxalNLP |

Every sentence is read by every system of its language, so systems are fully crossed with sentences. Audio is served
from the Hugging Face dataset [davidguzmanr/multilingual-mos-listening-test](https://huggingface.co/datasets/davidguzmanr/multilingual-mos-listening-test),
with every URL pinned to the audio commit `a6ea421b` (`resolve/<sha>/...`), so regenerating clips under the same names
later cannot change what annotators hear. Ground-truth clips from Common Voice and WaxalNLP are mp3, the rest wav.
At about 25-30 s per clip plus the second question, one annotator needs roughly 2.5-3.5 hours per language.

## Files

- `build_surveys.py` builds everything below from `clips.csv` (the dataset's root file, one row per clip with its pinned
  URL; `--refresh` downloads it again from the pinned commit). Checks: all URLs pinned, every sentence present for every
  system, identical reference text across systems, no empty text. Tasks are shuffled with seed 42, the same for all
  annotators of a language; `task_uid` is `<lang code>-<0001...>` in that order.
- `<Language>/labeling_config.xml`: the labeling interface (identical for the three languages).
- `<Language>/tasks.json`: the tasks to import. Task data holds only `task_uid`, `instructions_html`, `reference_text`
  and `audio_url`; the system name never appears in what annotators see in the labeling view. It is part of the
  audio URL, which is visible in the Data Manager, so give annotators the Annotator role only.
- `<Language>/tracking.csv`: `task_uid` -> `clip_id`, `dataset`, `system`, `duration_s`, `audio_url`. Join it with the
  exported annotations on `task_uid` (the export's `data.task_uid`).

```
python listening-study/surveys/build_surveys.py                     # all three languages
python listening-study/surveys/build_surveys.py --languages Hausa
python listening-study/surveys/build_surveys.py --api-key TOKEN     # also create the projects and import the tasks
```

## Setting up a project by hand

1. Create a project named after the language and paste `<Language>/labeling_config.xml` into Settings > Labeling
   Interface (Code view). Check in the preview that the options show no hotkey superscripts and that hovering an option
   shows its hint.
2. Import `<Language>/tasks.json` (Import > Upload files). Open one task and play the audio; the first task of each
   language is a quick check that the pinned URLs load.
3. Settings > Quality: overlap of annotations = 5 annotations per task, for 100% of tasks, so all five annotators
   rate every clip. Settings > Annotation: task sampling "Sequential" keeps the shuffled order identical for all
   annotators; "Uniform" gives each annotator a different random order, which averages out order effects. Either is
   fine; pick one and note it.
4. Add the five annotators with the Annotator role, after they have completed the tutorial project.
5. Export as CSV or JSON when done and join with `tracking.csv` on `task_uid`.
