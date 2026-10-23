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
- **Two separate 1-5 questions**: naturalness ("ignore whether the words are correct") and
  accuracy/faithfulness to the shown text ("ignore how natural it sounds"). Text stays visible, so the
  second axis is content accuracy, not P.800 intelligibility. Expect +20-40% time per clip; pilot on 50 clips.
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

TTS systems = pretrained models that synthesize the language out of the box; every language also gets
the 3 trainable toolkits from the pilot (EveryVoice, F5-TTS fine-tune, VITS). Datasets = sources with
sentences and human ground-truth audio: core = Open Bible (OB), FLEURS (FL), Common Voice (CV, counted
only above ~1 h); BOUQuET is text only. Full names behind each count:
[mos-datasets/target-languages.csv](../mos-datasets/target-languages.csv).

| Region | Language | Family › Branch | Distinct in region? | TTS systems | Datasets with text + GT audio | BOUQuET text |
|---|---|---|---|---:|---|---|
| East Asia | Chinese (Mandarin) | Sino-Tibetan › Sinitic | yes | 11 | 2 core (FL, CV) + 5 other | yes |
| East Asia | Japanese | Japonic | yes | 9 (10 if MMS-TTS) | 2 core (FL, CV) + 5 other | yes |
| East Asia | Korean | Koreanic | yes | 9 | 2 core (FL, CV 10 h) + 3 other | yes |
| South Asia | Hindi | Indo-European › Indo-Aryan | yes | 9 | 3 core (OB, FL, CV) + 3 other | yes |
| South Asia | Tamil | Dravidian › Southern | yes | 6 | 3 core (OB, FL, CV) + 2 other | no |
| Southeast Asia | Vietnamese | Austroasiatic › Vietic | yes | 4 | 3 core (OB, FL, CV small) + 3 other | yes |
| Africa | Yoruba | Niger-Congo › Volta-Niger | family shared | 4 (+Gemini unofficial) | 3 core (OB, FL, CV small) + 3 other | yes |
| Africa | Swahili | Niger-Congo › Bantu | **branch shared with Shona** | 6 | 3 core (OB, FL, CV) + 3 other | yes |
| Africa | Shona | Niger-Congo › Bantu | **branch shared with Swahili** | 2 | 2 core (OB, FL) + 1 other | no |
| Americas | Haitian Creole | French-lexifier creole | yes | 3 (OmniVoice nominal) | 1 core (OB) + 1 other | no |
| Americas | Quechua | Quechuan › Quechua II | yes | 2 (OmniVoice nominal) | 1 core (CV 1.7 h) + 3 other | no |
| Europe | French | Indo-European › Romance | **branch shared with Spanish** | 10 | 2 core (FL, CV) + 6 other | yes |
| Europe | Spanish | Indo-European › Romance | **branch shared with French** | 10 | 2 core (FL, CV) + 5 other | yes |
| Europe | German | Indo-European › Germanic | family shared | 10 | 2 core (FL, CV) + 6 other | yes |
| Middle East | Arabic | Afro-Asiatic › Semitic | yes | 7 | 3 core (OB, FL, CV) + 4 other | yes |
| Middle East | Turkish | Turkic › Oghuz | yes | 6 | 3 core (OB, FL, CV) + 0 other | yes |

Africa repeats a branch (Swahili and Shona are both Bantu) and Europe repeats a branch (French and
Spanish are both Romance); Indo-European appears four times overall. Swaps that remove the repeats and
have corpus coverage: Hausa or Oromo (Afro-Asiatic, in the pilot) or Dholuo (Nilo-Saharan, Open Bible +
FLEURS) for Shona; Hungarian or Finnish (Uralic), Polish (Slavic) or Greek (Hellenic) for French or
Spanish. Seven languages already have pilot audio from 5 systems: Hindi, Vietnamese, Yoruba, Swahili,
Shona, Haitian Creole, Turkish. Two sourcing problems:
Quechua has no Open Bible, FLEURS or BOUQuET coverage (pick a variety; Common Voice Chanka has 1.7 h);
Haitian Creole has Open Bible as its only ground-truth audio. Chinese, Japanese, Korean, French,
Spanish and German are not in Open Bible and need another ground-truth source.

## VoiceMOS timeline

The VoiceMOS 2026 challenge started in Friday, May 22.

