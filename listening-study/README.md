# Listening study plan

Planning notes for the multilingual MOS listening study. Goal: a released, listener-annotated corpus
in low-resource languages that can train a UTMOS-style predictor and seed a multilingual track for
VoiceMOS 2027. Working spreadsheet (MOS datasets and candidate languages):
<https://docs.google.com/spreadsheets/d/1_GVKQY-P4PgjTE9lP4BX22IK1OhiVvyfiEc32w4hEBY/edit?usp=sharing>

## What exists

Every open MOS predictor (SSL-MOS, UTMOS22, UTMOSv2, Audiobox Aesthetics) is trained on English
listening tests. The only multilingual one, Google's SQuId (42 training locales), was never released.
Public non-English TTS-naturalness ratings cover Mandarin, French, Argentine Spanish, Brazilian
Portuguese, Hindi/Tamil (MUSHRA) and Korean; 8 of our 10 pilot languages have none.

| Dataset | Language | Clips | Systems | Ratings/clip | Listeners | Access |
|---|---|---:|---:|---:|---:|---|
| BVCC | English | 7,106 | 187 | 8 | 304 | open (Zenodo) |
| SOMOS | English | 20,100 | 201 | 17-23 | 987 | CC BY-NC-SA |
| BC2019 (VoiceMOS 2022 OOD) | Mandarin | 1,352 | n/s | 10-17 | n/s | Zenodo |
| BC2023 Hub / Spoke (VoiceMOS 2023) | French | 882 / 578 | 21 / 17 | 15 | n/s | restricted |
| HablaMOS | Spanish | 4,326 | 12 TTS + 6 human | 1 | 92 | GitHub |
| IndicMOS L23 / L24 | 3 / 7 Indian langs | 2,403 / 4,413 | 17 / 23 | ~1 | n/s | unverified |
| MANGO | Hindi, Tamil, English | n/s | 3-4 + anchors | MUSHRA | 492 | CC BY 4.0 |
| KOMOS | Korean | 1,000 | n/s | 5 | 29 | not stated |
| Thiomi | 10 African langs | 50 / lang | 1 | 5 | 5 / lang | ratings not stated |
| SQuId | 66 locales | ~1.38M | 1,476 | ~1.4 | n/s | closed |

Full table with links: [mos-datasets/mos-datasets.csv](../mos-datasets/mos-datasets.csv).
Challenge tracks 2024-2026 used 5-10 ratings per clip, 6-31 systems, listener-wise labels with
rater IDs, unseen systems in test, and an open baseline. Details and sources:
[reports/Multilingual MOS and VoiceMOS history.md](../reports/Multilingual%20MOS%20and%20VoiceMOS%20history.md).

## What we have: the OpenBibleTTS pilot

| | |
|---|---|
| Languages | 10 (Hausa, Yoruba, Oromo, Swahili, Shona, Vietnamese, Hindi, Telugu, Haitian Creole, Turkish) |
| Design | 20 sentences x 3 corpora (Open Bible, FLEURS, BOUQuET) x 5 TTS systems + ground truth |
| Clips / ratings | 3,380 / 10,140; same 3 raters per language rate every clip |
| Clip duration | mean 8.34 s, median 8.00 s (7.83 h total) |
| Annotation time | median 19.4 s per rating; trimmed mean 31.3 s (raw mean unusable, wall-clock) |
| Rating noise | within-clip SD 0.73; between-clip MOS SD 1.10; annotator offsets SD 0.31 (range -0.97 to +0.55) |
| Zero-shot UTMOS22 / UTMOSv2 | utterance SRCC 0.22 / 0.24 on the 1,200-clip test split |
| Survey | Label Studio, one clip per task, blind, reference text always shown, single fused 1-5 scale |

Survey generator and configs: `~/scratch/Repositories/open-bible-surveys`.

## Design decisions

- **Metric: utterance-level SRCC**, reported per language and averaged (agreed with Erica Cooper).
  Low-resource languages have too few TTS systems for system-level correlation. VoiceMOS 2026
  already moved its primary metric to utterance-level SRCC.
- **5 listeners per language; overlap not yet decided.** Options: all 5 rate every clip (k = 5,
  ~300-450 clips per language, label ceiling 0.95, panel bias cancels within a language), disjoint
  sets (k = 1, ~1,500-2,300 clips, ceiling 0.82, listener bias not separable from clip quality), or
  a balanced partial overlap (k = 2-3). More clips favours variety for training; more ratings per clip
  favours cleaner utterance-level targets. Whatever the choice, release hashed rater IDs.
- **Two separate 1-5 questions**: naturalness (how the voice sounds: rhythm, intonation, audio quality)
  and intelligibility (whether the audio says exactly the shown text, word for word). The text stays
  visible, so intelligibility here means faithfulness to the text rather than P.800 unaided
  understanding. Expect +20-40% time per clip; pilot on 50 clips. Wording and examples:
  [tutorial/](tutorial/).
- **Widen the quality range cheaply** with degraded conditions (codec resynthesis, noise, speed) rather
  than more TTS systems. This was suggested by Erica.

## Budget: ~200 USD per language

5 listeners -> 40 USD each. Samples per listener by hourly rate and time basis:

| Rate | Hours / listener | Samples at median 19.4 s | Samples at trimmed mean 31.3 s |
|---|---:|---:|---:|
| 10 USD/h | 4.0 | ~740 | ~460 |
| 12.5 USD/h | 3.2 | ~590 | ~370 |
| 15 USD/h | 2.7 | ~500 | ~310 |

Use the median if paying per rating, the trimmed mean if paying per hour worked. Adjustments:
slow languages (Oromo, Yoruba ~25 s median) give ~575 at 10 USD/h; a 33% platform fee (Prolific)
cuts every figure by a quarter; two questions per clip cuts 20-40%. The budget buys roughly **1,500-2,300 ratings per language**; how many clips that
covers depends on the overlap decision (k = 5: 300-450 clips; k = 1: 1,500-2,300 clips).

## Candidate languages

TTS systems = pretrained models or commercial APIs that synthesize the language out of the box, from a
64-model sweep checked 2026-11-06 (research_notes/tts_models_coverage.csv); every language also gets the
3 trainable toolkits from the pilot (EveryVoice, F5-TTS fine-tune, VITS). Datasets = sources with
sentences and human ground-truth audio: core = Open Bible (OB), FLEURS (FL), Common Voice (CV, counted
only above ~1 h); BOUQuET is text only. Full names behind each count:
[mos-datasets/target-languages.csv](../mos-datasets/target-languages.csv).

| Region | Language | Family › Branch | Distinct in region? | TTS systems | Datasets with text + GT audio | BOUQuET text |
|---|---|---|---|---:|---|---|
| East Asia | Japanese | Japonic | yes | 34 (22 open + 12 API) | 2 core (FL, CV) + 5 other | yes |
| East Asia | Korean | Koreanic | yes | 32 (21 open + 11 API) | 2 core (FL, CV 10 h) + 3 other | yes |
| South Asia | Hindi | Indo-European › Indo-Aryan | yes | 31 (20 open + 11 API) | 3 core (OB, FL, CV) + 3 other | yes |
| Southeast Asia | Vietnamese | Austroasiatic › Vietic | yes | 21 (12 open + 9 API) | 3 core (OB, FL, CV small) + 3 other | yes |
| Southeast Asia | Indonesian | Austronesian › Malayo-Polynesian | yes | 18 (9 open + 9 API) | 2 core (FL, CV) + 0 other | yes |
| Africa | Yoruba | Niger-Congo › Volta-Niger | family shared | 11 (10 open + 1 API) | 3 core (OB, FL, CV small) + 3 other | yes |
| Africa | Swahili | Niger-Congo › Bantu | family shared | 19 (13 open + 6 API) | 3 core (OB, FL, CV) + 3 other | yes |
| Africa | Hausa | Afro-Asiatic › Chadic | family shared with Arabic, branch distinct | 11 (8 open + 3 API) | 3 core (OB, FL, CV) + 2 other | no |
| Americas | Haitian Creole | French-lexifier creole | yes | 9 (7 open + 2 API) | 1 core (OB) + 1 other | no |
| Europe | French | Indo-European › Romance | **branch shared with Spanish** | 39 (27 open + 12 API) | 2 core (FL, CV) + 6 other | yes |
| Europe | Spanish | Indo-European › Romance | **branch shared with French** | 38 (26 open + 12 API) | 2 core (FL, CV) + 5 other | yes |
| Europe | German | Indo-European › Germanic | family shared | 37 (25 open + 12 API) | 2 core (FL, CV) + 6 other | yes |
| Middle East | Arabic | Afro-Asiatic › Semitic | yes | 28 (17 open + 11 API) | 3 core (OB, FL, CV) + 4 other | yes |
| Middle East | Turkish | Turkic › Oghuz | yes | 25 (15 open + 10 API) | 3 core (OB, FL, CV) + 0 other | yes |

Shona was dropped and Indonesian added on 2026-11-02 (adds Austronesian, removes the Bantu repeat);
Hausa was added on 2026-11-04 as the third African language (Chadic: same family as Arabic, different
branch). Europe still repeats a branch (French and Spanish are both Romance) and Indo-European appears
four times overall; Hungarian or Finnish (Uralic), Polish (Slavic) or Greek (Hellenic) would replace one
of them. Seven languages already have pilot audio from 5 systems: Hindi, Vietnamese, Yoruba, Swahili,
Hausa, Haitian Creole, Turkish. Chinese, Tamil and Quechua were dropped on 2026-11-06, leaving 14 languages. One sourcing problem:
Haitian Creole has Open Bible as its only ground-truth audio. Japanese, Korean, Indonesian, French,
Spanish and German are not in Open Bible and need another ground-truth source.

## Candidate TTS systems per language

Shortlist from the 64-model sweep (full lists in [mos-datasets/target-languages.csv](../mos-datasets/target-languages.csv)).
Common core in every language: ground truth, Gemini TTS, ElevenLabs, OmniVoice, MMS-TTS (no Japanese), plus
EveryVoice and F5-TTS fine-tunes. Gemini (Yoruba) and ElevenLabs (Haitian Creole) are unlisted but worked in the pilot.
Target 9 systems per language, ~100 sentences each; keep MMS-TTS/Piper as genuinely weak systems and add
text-corrupted or degraded conditions where high-resource languages cluster at the top.

| Language | Open weights beyond the core | APIs |
|---|---|---|
| Japanese, Korean | VoxCPM2, Qwen3-TTS, CosyVoice 3, Fish S2-Pro, Higgs TTS 3, MOSS-TTS v1.5 (Kokoro ja) | Gemini, ElevenLabs |
| Hindi | VoxCPM2, IndicF5, Indic Parler-TTS, Higgs TTS 3, Kokoro, Piper | Gemini, ElevenLabs |
| Vietnamese | VoxCPM2, Fish S2-Pro, Higgs TTS 3, MOSS-TTS v1.5, Piper, NVIDIA Magpie | Gemini, ElevenLabs |
| Indonesian | VoxCPM2, Fish S2-Pro, Higgs TTS 3, Piper | Gemini, ElevenLabs |
| Yoruba | Fish S2-Pro, Sunbird Orpheus-3B, YarnGPT, VoxCPM2-Yoruba, BibleTTS | ElevenLabs (Gemini unofficial) |
| Swahili | VoxCPM2, Fish S2-Pro, Higgs TTS 3, MOSS-TTS v1.5, Chatterbox, Sunbird Orpheus-3B, swa-csm-1b, Piper | Gemini, ElevenLabs |
| Hausa | Higgs TTS 3, Sunbird Orpheus-3B, YarnGPT, BibleTTS | Gemini, ElevenLabs |
| Haitian Creole | Fish S2-Pro, Higgs TTS 3 | Gemini (ElevenLabs unofficial) |
| French, Spanish, German | VoxCPM2, Qwen3-TTS, CosyVoice 3, Fish S2-Pro, Higgs TTS 3, MOSS-TTS v1.5, Kokoro (fr, es), Chatterbox, Piper | Gemini, ElevenLabs |
| Arabic | VoxCPM2, Fish S2-Pro, Higgs TTS 3, MOSS-TTS v1.5, Chatterbox, Piper, IndexTTS-2.5 | Gemini (Egyptian, MSA), ElevenLabs |
| Turkish | VoxCPM2, Fish S2-Pro, Higgs TTS 3, MOSS-TTS v1.5, Chatterbox, Piper | Gemini, ElevenLabs |

Licences: the release will be non-commercial (MMS-TTS CC-BY-NC, Fish and Higgs research licences, YarnGPT NC-SA);
Indic Parler-TTS (Apache-2.0) and IndicF5 (MIT) are gated for access only. Check Gemini and ElevenLabs terms on
redistributing generated audio before synthesis. Cloning models need a reference voice with permission.

## Environments on the cluster

`module load miniconda/3 && conda activate <env>`; scripts, smoke tests and fixes in [env/](env/) (see its README).
All six were verified by synthesizing audio on an L40S (2026-11-06).

| Env | Models | Notes |
|---|---|---|
| `Multilingual-MOS` | OmniVoice, VoxCPM2, MMS-TTS, Kokoro, Piper, F5-TTS, Orpheus/SNAC, swa-csm, coqui-tts (OpenBible VITS), Gemini and ElevenLabs SDKs | py3.10, torch 2.8.0+cu128, transformers 5.19.0 |
| `Multilingual-MOS-qwen3-tts` | Qwen3-TTS, YarnGPT | transformers 4.57.3 |
| `Multilingual-MOS-fish-speech` | Fish S2-Pro | exact tokenizers/pydantic pins; CLI inference |
| `Multilingual-MOS-moss-tts` | MOSS-TTS v1.5 | py3.12, torch 2.9.1, transformers 5.0.0 |
| `Multilingual-MOS-higgs` | Higgs TTS 3 (sglang-omni server) | py3.12; `module load cuda/13.1.0`, `SGLANG_OMNI_STARTUP_TIMEOUT=3600` |
| `Multilingual-MOS-indic` | Indic Parler-TTS, IndicF5 | transformers 4.46.1; gated weights (HF login) |
| `EveryVoice` (existing) | EveryVoice fine-tunes | torch 2.7.1 |

GPU jobs: `sbatch --partition=unkillable --gpus=l40s:1 --cpus-per-task=4 --mem=32G` (the interactive allocation is CPU-only).

## Open questions and issues

- **Annotator overlap.** Full overlap (k = 5): ~300-450 clips, label ceiling 0.95. Disjoint sets (k = 1): ~1,500-2,300
  clips, ceiling 0.82, listener bias not separable. Leaning to mostly disjoint with a 10-20% shared subset; undecided.
- **Top clustering in high-resource languages.** The strong models are near-human in fr/es/de/ja/ko (pilot: 4.7-4.9 in
  Swahili/Hindi vs 2.2-2.4 in Haitian Creole/Yoruba); little variance left to predict there.
- **Bad samples.** Prefer weak real systems (MMS-TTS 3.06, F5 fine-tunes 2.74 in the pilot) and models run out of
  distribution; synthesize from corrupted text for the accuracy axis; signal degradations last, capped at 10-20% of clips.
- **Two questions per clip.** Time cost (+20-40%?) and inter-axis correlation unmeasured; pilot on ~50 clips.
- **Ground truth** for ja/ko/id/fr/es/de must come from FLEURS or Common Voice (no Open Bible).
- **Cloning.** Some models require a reference audio, also consent for the reference audio.
- **Licences.** Release will be CC-BY-NC at best; Gemini/ElevenLabs redistribution terms unchecked.
- **Varieties.** Arabic (MSA/Egyptian) and Spanish (European/Latin American) still to choose.

## VoiceMOS timeline

The VoiceMOS 2026 challenge started in Friday, May 22.

