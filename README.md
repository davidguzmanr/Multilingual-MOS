# Multilingual MOS

Human MOS annotations for TTS output in 10 languages, used to evaluate and
fine-tune automatic MOS predictors (UTMOS22, UTMOSv2).

## Data

`data/annotations-anonymized/{Language}.csv` — one row per individual rating.

Fully-crossed design, per language:

- **20 sentences** × **3 corpora** (`bouquet`, `fleurs`, `open-bible`)
- **6 systems**: EveryVoice, F5, Gemini, OmniVoice, VITS, ground-truth
- every clip rated by the **same 3 annotators** (30 annotators total, none shared
  across languages)

3,380 clips / 10,140 ratings. Audio in `data/audios/{corpus}/{system}/{language}/`.

Two gaps: `bouquet` has no ground-truth, and Haitian Creole has no
`fleurs × ground-truth`.

### Clip duration and annotation time

Clip duration from the wav headers; annotation time from the `lead_time` column
(seconds of wall-clock per rating):

| Language | dur. mean | dur. median | time mean | time median |
|---|---:|---:|---:|---:|
| Haitian Creole | 9.09 | 9.37 | 142.7 | 21.0 |
| Hausa | 9.57 | 8.94 | 47.8 | 24.3 |
| Hindi | 8.84 | 8.53 | 36.8 | 17.1 |
| Oromo | 7.99 | 7.11 | 87.1 | 25.4 |
| Shona | 7.67 | 7.35 | 195.6 | 20.1 |
| Swahili | 8.90 | 8.61 | 111.5 | 19.5 |
| Telugu | 8.12 | 8.13 | 43.9 | 21.4 |
| Turkish | 7.73 | 7.52 | 105.2 | 12.7 |
| Vietnamese | 7.45 | 7.18 | 80.8 | 12.6 |
| Yoruba | 8.03 | 7.77 | 71.6 | 25.1 |
| **All** | **8.34** | **8.00** | **92.0** | **19.4** |

7.83 h of audio total. Duration varies far more by corpus than by language
(`bouquet` 4.26 mean, `fleurs` 10.23, `open-bible` 9.87), and ground-truth runs
~35% longer than any system (11.19 vs 7.01–8.77) on the same texts.

`lead_time` is wall-clock on the task, so it absorbs breaks — the raw mean is
unusable (max 22.6 h, std 1409). Prefer the median, ~19 s; trimming the top and
bottom 1% gives a mean of 31.3 s.

Let's say each annotator rates 500 samples, at a median of 19.4s per sample it would
take 2.7 hours to annotate them (just one dimension). Around 30-40 USD seems fair. 
With 5 annotators per  language each language would cost us 150-200 USD.

## Split strategy: unseen system × corpus pairing

One (corpus, system) cell **per system** is held out for test — a Latin-square
diagonal over the 3 × 6 grid — and the **same six cells are held out in every
language**:

| corpus | held-out systems |
|---|---|
| `bouquet` | Gemini, OmniVoice |
| `fleurs` | F5, VITS |
| `open-bible` | EveryVoice, ground-truth |

Each held-out system is still trained on in its other two corpora, and each
corpus is still trained on with its other systems — only the *intersection* is
novel. Test is therefore harder than train without any system being unseen.

Ground-truth must sit in `open-bible`: it is its only cell present in all 10
languages. That makes test cover all 6 systems in all 10 languages, which is
what allows system-level correlations to be computed per language.

The diagonal was picked by enumerating all 30 valid assignments, keeping those
whose pooled test mean MOS is within 0.05 of the grand mean (3.444), and taking
the one with the largest spread between system means — per-language system SRCC
is over only 6 points, so well-separated systems reduce noise.

**Validation** is a *sentence* holdout instead: 3 of the 20 sentences per
(language, corpus), removed from every non-test cell. A second diagonal would
have left ground-truth with a single training cell. Val therefore covers all 6
systems and shares no sentence with train, but its pairings *were* seen in
training — it is fit for checkpoint selection, not as a test proxy.

Note test sentences do appear in training under other systems: the task is
judging a rendering, not a sentence, and the VoiceMOS main track has the same
property.

## Splits

Clips (ratings are 3× each):

| Language | train | val | test | total |
|---|---:|---:|---:|---:|
| Haitian Creole | 170 | 30 | 120 | 320 |
| Hausa | 187 | 33 | 120 | 340 |
| Hindi | 187 | 33 | 120 | 340 |
| Oromo | 187 | 33 | 120 | 340 |
| Shona | 187 | 33 | 120 | 340 |
| Swahili | 187 | 33 | 120 | 340 |
| Telugu | 187 | 33 | 120 | 340 |
| Turkish | 187 | 33 | 120 | 340 |
| Vietnamese | 187 | 33 | 120 | 340 |
| Yoruba | 187 | 33 | 120 | 340 |
| **Total** | **1,853** | **327** | **1,200** | **3,380** |

Test is exactly 120 clips per language (20 × 6 systems) for all 10. Haitian
Creole's smaller train/val follows from its missing `fleurs × ground-truth`
cell; its test set is unaffected.

## Zero-shot results on test

Pretrained UTMOS22 (main-track strong learner, *w/o phoneme encoder* variant)
and UTMOSv2 (`fusion_stage3`), no fine-tuning:

| model | level | n | MSE | LCC | SRCC |
|---|---|---:|---:|---:|---:|
| UTMOS22 | utterance | 1200 | 1.301 | 0.250 | **0.218** |
| UTMOS22 | system (lang × sys) | 60 | 0.853 | 0.288 | **0.266** |
| UTMOSv2 | utterance | 1200 | 1.439 | 0.247 | **0.240** |
| UTMOSv2 | system (lang × sys) | 60 | 0.966 | 0.354 | **0.340** |

For reference UTMOS22 scores 0.88 utterance-SRCC on the VoiceMOS main track.
MSE is dominated by scale mismatch — both models compress into ~2.1–4.5 against
a full 1–5 human range — so prefer SRCC.

Per language (system SRCC is over 6 points; the 5% threshold is |ρ| ≈ 0.886, so
only Swahili / Vietnamese / Haitian Creole are individually meaningful):

| Language | UTMOS22 utt | UTMOS22 sys | UTMOSv2 utt | UTMOSv2 sys |
|---|---:|---:|---:|---:|
| Haitian Creole | 0.557 | 0.829 | 0.508 | 0.943 |
| Hausa | 0.251 | 0.371 | 0.457 | 0.771 |
| Hindi | 0.477 | 0.829 | 0.271 | 0.257 |
| Oromo | 0.111 | 0.029 | −0.063 | −0.371 |
| Shona | 0.268 | 0.371 | 0.019 | −0.086 |
| Swahili | 0.552 | 0.943 | 0.455 | 0.771 |
| Telugu | 0.479 | 0.406 | 0.306 | 0.087 |
| Turkish | 0.221 | −0.143 | 0.241 | 0.371 |
| Vietnamese | 0.626 | 0.886 | 0.531 | 0.943 |
| Yoruba | 0.040 | 0.143 | 0.077 | −0.029 |
| **mean ± sd** | 0.358 ± 0.205 | 0.466 ± 0.388 | 0.280 ± 0.213 | 0.366 ± 0.470 |

**They rank systems, not languages.** Aggregating each system across all
languages (n=6) gives SRCC 0.829 (UTMOS22) and 0.886 (UTMOSv2) — near-correct.
But ranking the 10 *languages* by mean score gives −0.491 and −0.152: both score
Haitian Creole and Yoruba (worst human MOS) above Swahili and Telugu (best).
Removing per-language offsets raises utterance SRCC from 0.218 → 0.348 and
0.240 → 0.287, so the language-level signal is actively hurting.

Expected from the architecture: UTMOS22's demo checkpoint hardcodes `domain=0`
(BVCC) and `judge_id=288` (an English mean listener), and both models were
trained on English listening tests. The domain embedding is exactly the
mechanism fine-tuning would use to fix this.

Utterance SRCC also collapses to ~0 on the 249 high-disagreement clips, and the
top three systems (EveryVoice 3.953, ground-truth 4.015, Gemini 4.017) are
within 0.064 of each other, so those positions are unrankable by any model.

## Usage

```bash
module load miniconda/3 && conda activate UTMOS

python src/build_splits.py                       # -> data/splits/system-corpus/
python src/predict_mos.py --split test           # -> predictions_test.csv
python src/evaluate_mos.py --split test          # utterance + system SRCC
```

| Script | Purpose |
|---|---|
| [src/build_splits.py](src/build_splits.py) | Build train/val/test from the annotation CSVs |
| [src/predict_mos.py](src/predict_mos.py) | Score a split with UTMOS22 and/or UTMOSv2 |
| [src/evaluate_mos.py](src/evaluate_mos.py) | MSE / LCC / SRCC, pooled and per language |
| [src/setup_env.sh](src/setup_env.sh) | Build the shared `UTMOS` conda env |
| [src/verify_env.py](src/verify_env.py) | Check the env can run both models |

Outputs land in `data/splits/system-corpus/`: `ratings.csv` (one row per rating),
`clips.csv` (averaged MOS per clip), `summary.txt`, `config.json`.
