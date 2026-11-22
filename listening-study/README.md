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
- **5 listeners per language, all rating the same ~300 clips (k = 5).** Decided 2026-11-13 after
  discussion with Wen-Chin Huang and Erica Cooper, who want at least 5 ratings per clip for a
  challenge set (a sparse SQuId-style design at ~1.4 ratings per clip was considered and dropped).
  Gives ~1,500 ratings per language, ~4,200 clips and ~21,000 ratings over 14 languages. Spread the 300 clips 
  over as many   sentences as possible (~33 sentences x 9 systems). Release hashed rater IDs.
- **Training at this size is viable** because the predictor is fine-tuned from a pretrained encoder on
  the pooled 14-language set plus OpenBibleTTS: VoiceMOS 2022 OOD trained on 136 Mandarin clips,
  KOMOS on 810 Korean clips (utterance SRCC 0.90), West Frisian gains plateaued at ~66 clips.
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
cuts every figure by a quarter; two questions per clip cuts 20-40%. The budget buys roughly
**1,500 ratings per language**, which at k = 5 is the ~300 clips per language in the plan above.

## Candidate languages

TTS systems = pretrained models or commercial APIs that synthesize the language out of the box, from a
64-model sweep checked 2026-11-06 (research_notes/tts_models_coverage.csv); every language also gets the
2 trainable toolkits kept from the pilot (EveryVoice, VITS). Datasets = sources with
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

## Candidate source datasets/domains per language

Each language needs 34-38 sentences (300 ratings / 8-9 systems), ideally from 3 domains, each with a human recording
to serve as the ground-truth system. Candidates with licences and URLs (166 corpora, checked 2026-11-16):
[mos-datasets/gt-corpora.csv](../mos-datasets/gt-corpora.csv); notes in `research_notes/gt_corpora_group_{A,B}.md`.
"crowd" = multi-speaker volunteer recordings (weaker GT); NC = non-commercial; redistribution flags refer to the GT audio.

Core sources (scripture, Wikipedia, crowd-read sentences), all with human audio; cells give licence and GT quality.
Scripture comes from the OpenBibleTTS release [davidguzmanr/open-bible-resources](https://huggingface.co/datasets/davidguzmanr/open-bible-resources)
(Biblica audio via open.bible, CC BY-SA; fields audio, text, testament/book/chapter/verse, duration, speaker_id): take sentences only from
the **test** split, since EveryVoice and VITS are trained on train, and keep one speaker_id where several narrators exist. Verses run
7-12 s and 112-134 characters on average, longer than the pilot's 8 s clips, so filter by duration. Common Voice audio may be used as
GT in the test but not re-hosted.

| Language | FLEURS (Wikipedia) | Common Voice 27.0 (validated, crowd) | Dataset/Domain 3 | Total |
|---|---|---|---|---:|
| Japanese | yes (crowd) | yes: 300,315 clips, 6,932 speakers; colloquial, JSUT, literature |  | 2 |
| Korean | yes (crowd) | yes: 1,773 clips, 114 speakers; fiction |  | 2 |
| Hindi | yes (crowd) | yes: 11,203 clips, 362 speakers; news headlines | Open Bible: test 1,522 verses, 1 speaker, median 11.1 s | 3 |
| Vietnamese | yes (crowd) | yes: 7,013 clips, 229 speakers; web fiction | Open Bible: test 1,523 verses, 2 speakers (1,459 + 64), median 7.7 s | 3 |
| Indonesian | yes (crowd) | yes: 30,901 clips, 470 speakers; everyday |  | 2 |
| Yoruba | yes (crowd) | yes: 3,549 clips, 113 speakers; news | WaxalNLP yor_tts (Media Trust, CC BY-SA 4.0): test 354 clips, 8 studio speakers (4 f / 4 m), 48 kHz, median 16.7 s (182 clips in 5-20 s), phonetically balanced scripts | 3 |
| Swahili | yes (crowd) | yes: 269,140 clips, 1,206 speakers; news, everyday | WaxalNLP swa_tts (Makerere, CC BY-SA 4.0): test 199 clips, 7 studio speakers (4 f / 3 m), 48 kHz, median 16.9 s (103 clips in 5-20 s), phonetically balanced scripts | 3 |
| Hausa | yes (crowd) | yes: 3,582 clips, 22 speakers; conversational | WaxalNLP hau_tts (Media Trust, CC BY-SA 4.0): test 197 clips, 8 studio speakers (4 f / 4 m), 48 kHz, median 15.1 s (117 clips in 5-20 s), phonetically balanced scripts | 3 |
| Haitian Creole | no (FLORES text only) | yes: 29 clips, 3 speakers; chat (~10 usable) | Open Bible: test 1,530 verses, 1 speaker, median 11.5 s | 2 |
| French | yes (crowd) | yes: 792,011 clips, 20,051 speakers; novels, everyday, parliament |  | 2 |
| Spanish | yes (es_419, crowd) | yes: 440,794 clips, 17,688 speakers; literature |  | 2 |
| German | yes (crowd) | yes: 953,365 clips, 19,748 speakers; parliament, everyday |  | 2 |
| Arabic | yes (Egyptian, crowd) | yes: 80,696 clips, 1,142 speakers; mixed MSA | Open Bible (MSA): test 1,526 verses, 2 speakers (1,094 / 432), median 9.3 s | 3 |
| Turkish | yes (crowd) | yes: 121,612 clips, 1,720 speakers; conversational, news | Open Bible: test 1,488 verses, 1 speaker, median 7.1 s (63.5 h in the paper; narrator identity unverified, rated 3.81 as GT in the pilot) | 3 |

Common Voice Scripted Speech 27.0 per locale (Mozilla Data Collective, CC0; local copies under `data/Common-Voice/`,
analysed 2026-11-17 with `listening-study/tools/analyze_common_voice_sentences.py`). Clips = validated recordings; each
speaker reads a sentence once. Sentence sources come from the `source` column; the `sentence_domain` tags are empty
except in Japanese. Target: ~35 sentences per language in total, a few speakers at most, so even small pools count.

| Language | Locale | Clips | Speakers | Usable source | Domain/register | Verdict |
|---|---|---:|---:|---|---|---|
| Japanese | [ja](https://mozilladatacollective.com/datasets/cmu5xbhnf00fmnq07z9p3aa70) | 300,315 | 6,932 | sentence-collector; JSUT; Aozora Bunko | colloquial; JSUT sentences; literature | 3 domains; JSUT text also has studio GT |
| Korean | [ko](https://mozilladatacollective.com/datasets/cmu5rwtaf003dnq07e7n3q6tf) | 1,773 | 114 | sentence-collector; Self Citation | public-domain fiction, some legal; colloquial | literature usable; colloquial only 69 clips, reserve |
| Hindi | [hi](https://mozilladatacollective.com/datasets/cmu5vlext009to107ev1s9lk8) | 11,203 | 362 | sentence-collector | news headlines (often verbless) | news usable, pick full sentences |
| Vietnamese | [vi](https://mozilladatacollective.com/datasets/cmu5vci3w00agnq07hy4nbauv) | 7,013 | 229 | sentence-collector | web-novel narrative/dialogue (1 speaker: 3,373 clips) | fiction usable; Hue dictionary fragments skip |
| Indonesian | [id](https://mozilladatacollective.com/datasets/cmu5wcato00cknq07yjnu14e0) | 30,901 | 470 | sentence-collector | everyday / Tatoeba-style | everyday usable; CoVoST2 part is translated Wikipedia, skip |
| Yoruba | [yo](https://mozilladatacollective.com/datasets/cmu5q07ui011gmi07vflkzmej) | 3,549 | 113 | sentence-collector | Nigerian news prose | news usable |
| Swahili | [sw](https://mozilladatacollective.com/datasets/cmu5sgles002zo1071nphgtby) | 269,140 | 1,206 | batch-1 contributors; batch-3; agriculture set | East African news; everyday; farming | news + everyday usable; avoid dialect sets (Kimvita, Kibajuni, ...) |
| Hausa | [ha](https://mozilladatacollective.com/datasets/cmu5u0ood006tnq07tmgdpgwi) | 3,582 | 22 | sentence-collector | everyday conversation (1 speaker: 2,048 clips) | conversational usable, near single-voice |
| Haitian Creole | [ht](https://mozilladatacollective.com/datasets/cmu5o29ou00w3mh07e9heh8xv) | 29 | 3 | sentence-collector | informal workplace chat (translated Enron prompts) | ~10-12 clean sentences from 2-3 speakers; rest from Open Bible/CMU |
| French | [fr](https://mozilladatacollective.com/datasets/cmu5kub4600pkmh078n7n5ogl) | 792,011 | 20,051 | sentence-collector; ofis_publik; untagged set | 19th-c. novels; short everyday; parliamentary | 3 domains beyond Wikipedia |
| Spanish | [es](https://mozilladatacollective.com/datasets/cmu5kitx700pemi073feoiqqu) | 440,794 | 17,688 | sentence-collector; untagged set; Reina-Valera | novels and everyday lines; tokenised novel excerpts; scripture (44 clips, 35 speakers) | literature usable (1 speaker: 4,769 sentences); scripture too scattered; 87% of clips are Wikipedia |
| German | [de](https://mozilladatacollective.com/datasets/cmu5ktqhe00pgmh07p1xamabd) | 953,365 | 19,748 | europarl; sentence-collector; jf99 | parliament; everyday; short narrative | 3 domains beyond Wikipedia |
| Arabic | [ar](https://mozilladatacollective.com/datasets/cmu5wom3z00ceo107guxtg4xa) | 80,696 | 1,142 | sentence-collector | mixed MSA: everyday phrases, Quran verses, poetry; no tags | usable after manual filtering by content |
| Turkish | [tr](https://mozilladatacollective.com/datasets/cmu5wkah500c2o10719lem8m8) | 121,612 | 1,720 | sentence-collector; setimes | conversational; Balkan news | 2 domains; Wikipedia is 11k clips only |

Menu for further domains:

| Language | Domain 1 | Domain 2 | Domain 3 |
|---|---|---|---|
| Hindi | scripture: Open Bible (CC BY-SA) | general prose: SYSPIN Hindi (1 studio voice, 48 kHz, CC BY 4.0) | conversational: IndicVoices (CC BY 4.0, crowd) |
| Vietnamese | scripture: Open Bible (CC BY-SA, 48 kHz) | conversational: FOSD (FPT licence, crowd) | encyclopedic: FLEURS (crowd); InfoRe 25 h single speaker if licensed |
| Indonesian | news: INDspeech_NEWS_TTS (1 speaker, NC-SA; own 40-sentence MOS list) | encyclopedic: FLEURS or Common Voice (crowd) | literature: LibriVox Indonesia (CC0, amateur multi-speaker) |
| Yoruba | scripture: BibleTTS SLR129 (1 studio speaker, 48 kHz, CC BY-SA) | news/fiction: IroyinSpeech (2 studio voices; ELRA academic licence, GT likely not releasable) | general: OpenSLR SLR86 (CC BY-SA, crowd) or WAXAL yor_tts (studio, script-like text) |
| Swahili | scripture: Open Bible (CC BY-SA) | general: WAXAL swa_tts (studio, CC BY/BY-SA) | conversational: SALT studio-swa (Sunbird, CC BY-SA, 3,988 sentences) |
| Hausa | scripture: BibleTTS (CC BY-SA; one narrator) | encyclopedic: FLEURS (crowd) | conversational: Gamayun Hausa mini-kit (CC BY-NC, non-studio) or TWB Voice Hausa (studio, NC, on request) |
| Haitian Creole | scripture: Open Bible Bib Sen An (CC BY-SA; needs alignment) | news/political: CMU 2010 single-speaker subset (1 male, 1,273 utts, 16 kHz, permissive) | manuals/primers: CMU 2010 multi-speaker set (crowd-like) |
| Turkish | news: MediaSpeech SLR108 (CC BY 4.0, broadcast, multi-speaker) | encyclopedic: FLEURS (crowd) | scripture: Open Bible YTC audio (CC BY-ND; human vs synthetic narrator UNVERIFIED) or DFKI OT generic sentences (NC-SA) |
| Japanese | conversational: Hi-Fi-CAPTAIN (NICT, 2 speakers; licence page down, NC-SA per snapshot) | literature: Kokoro Speech (public domain, 1 reader) | encyclopedic: FLEURS or JSUT basic5000 |
| Korean | everyday: KSS (1 studio voice, NC-SA) | encyclopedic: FLEURS or Zeroth-Korean (CC BY 4.0, crowd) | user sentences: Common Voice ko (crowd) |
| French | parliament: SIWIS parl (1 professional voice, 44.1 kHz, CC BY 4.0) | literature: SIWIS book/chap or SynPaFlex (NC-SA) | encyclopedic: Vibravox (CC BY 4.0, 48 kHz, multi-speaker) or FLEURS |
| Spanish | everyday + weather: OpenSLR 61/71-75 Latin American (CC BY-SA, crowd) | literature: CSS10 es (1 reader) or M-AILABS | encyclopedic: FLEURS (es_419) |
| German | everyday: Thorsten-Voice (CC0, 1 speaker, 44.1 kHz) | literature: HUI Clean, one reader (CC0) or CSS10 de | encyclopedic: Tuda-De (CC BY 4.0) or Spoken Wikipedia |
| Arabic | scripture: Open Bible MSA (CC BY-SA) | news: ArVoice Part 2 (professional, CC BY 4.0 via data-use agreement) or FLEURS (Egyptian) | classical: ArVoice Part 1 or ClArTTS; conversational: SADA |

Open items: Turkish Open Bible audio may be TTS (24 kHz mono mp3, no narrator credited; listen to
https://ebible.org/turytc/mp3/1JN01.mp3); Korean has no open scripture/literature audio; Haitian Creole has no
single-speaker third domain; Common Voice and CMU Wilderness audio cannot be re-hosted; corpora used to train
EveryVoice/VITS (JSUT, KSS, INDspeech, CSS10) must contribute only held-out sentences.

## Candidate TTS systems per language

Shortlist from the 64-model sweep (full lists in [mos-datasets/target-languages.csv](../mos-datasets/target-languages.csv)).
Common core in every language (6): ground truth, Gemini TTS, ElevenLabs, OmniVoice, VITS (MMS-TTS checkpoints; a
self-trained VITS for Japanese, which MMS lacks) and an EveryVoice fine-tune. F5-TTS was dropped on 2026-11-16 (slow to
train, weak in the pilot). Gemini (Yoruba) and ElevenLabs (Haitian Creole) are unlisted but worked in the pilot.
Target 9 systems per language (core + 3 from the table below), ~33 sentences each; keep VITS/Piper as genuinely weak
systems and add text-corrupted or degraded conditions where high-resource languages cluster at the top.

Systems per language (GT = ground-truth source; MMS = facebook/mms-tts checkpoint; EveryVoice: pilot = fine-tune exists and was rated in the pilot, OpenBibleTTS ckpt = trained but not rated (multilingual-tts org has EveryVoice, F5 and VITS checkpoints for all 37 Open Bible languages), to train = needed; extras: VoxCPM2, Fish S2-Pro, Higgs TTS 3, Sunbird Orpheus-3B (Yoruba, Hausa) as of 2026-11-16; unofficial/nominal entries count toward the total):

| Language | GT | Gemini | ElevenLabs | OmniVoice | VITS | EveryVoice | VoxCPM2 | Fish S2-Pro | Higgs TTS 3 | Sunbird Orpheus-3B | Total |
|---|---|---|---|---|---|---|---|---|---|---|---:|
| Japanese | FLEURS / CV | yes | yes | yes | own VITS (no MMS) | to train | yes | yes | yes | no | 9 |
| Korean | FLEURS / CV | yes | yes | yes | MMS kor | to train | yes | yes | yes | no | 9 |
| Hindi | Open Bible | yes | yes | yes | MMS hin | pilot | yes | yes | yes | no | 9 |
| Vietnamese | Open Bible | yes | yes | yes | MMS vie | pilot | yes | yes | yes | no | 9 |
| Indonesian | FLEURS / CV | yes | yes | yes | MMS ind | to train | yes | yes | yes | no | 9 |
| Yoruba | Open Bible | unofficial | yes | yes | MMS yor | pilot | VoxCPM2-Yoruba (1) | yes | no | yes (2) | 9 |
| Swahili | Open Bible | yes (Flash) | yes | yes | MMS swh | pilot | yes | yes | yes | no | 9 |
| Hausa | Open Bible | yes | yes | yes | MMS hau | pilot | no | no | yes | yes (2) | 8 |
| Haitian Creole | Open Bible | yes | unofficial | nominal | MMS hat | pilot | no | yes | yes | no | 8 |
| French | FLEURS / CV | yes | yes | yes | MMS fra | to train | yes | yes | yes | no | 9 |
| Spanish | FLEURS / CV | yes | yes | yes | MMS spa | to train | yes | yes | yes | no | 9 |
| German | FLEURS / CV | yes | yes | yes | MMS deu | to train | yes | yes | yes | no | 9 |
| Arabic | Open Bible (MSA) | yes (Egyptian, MSA) | yes | yes | MMS ara | OpenBibleTTS ckpt | yes | yes | yes | no | 9 |
| Turkish | Open Bible | yes | yes | yes | MMS tur | pilot | yes | yes | yes | no | 9 |

(1) LyngualLabs/VoxCPM2-Yoruba: a community fine-tune of VoxCPM2, not the base model; CC-BY-NC-4.0.
(2) Sunbird/orpheus-3b-tts-multilingual (Apache-2.0): 20 African-language configs; language is set by the speaker ID (`waxal_yor_*`, `waxal_hau_*`); quality unmeasured, audition first; not yet smoke-tested here.

Menu for further extras:

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
| `Multilingual-MOS` | OmniVoice, VoxCPM2, MMS-TTS, Kokoro, Piper, Orpheus/SNAC, swa-csm, coqui-tts (OpenBible VITS), Gemini and ElevenLabs SDKs (F5-TTS installed but unused) | py3.10, torch 2.8.0+cu128, transformers 5.19.0 |
| `Multilingual-MOS-qwen3-tts` | Qwen3-TTS, YarnGPT | transformers 4.57.3 |
| `Multilingual-MOS-fish-speech` | Fish S2-Pro | exact tokenizers/pydantic pins; CLI inference |
| `Multilingual-MOS-moss-tts` | MOSS-TTS v1.5 | py3.12, torch 2.9.1, transformers 5.0.0 |
| `Multilingual-MOS-higgs` | Higgs TTS 3 (sglang-omni server) | py3.12; `module load cuda/13.1.0`, `SGLANG_OMNI_STARTUP_TIMEOUT=3600` |
| `Multilingual-MOS-indic` | Indic Parler-TTS, IndicF5 | transformers 4.46.1; gated weights (HF login) |
| `EveryVoice` (existing) | EveryVoice fine-tunes | torch 2.7.1 |

GPU jobs: `sbatch --partition=unkillable --gpus=l40s:1 --cpus-per-task=4 --mem=32G` (the interactive allocation is CPU-only).

## Open questions and issues

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

