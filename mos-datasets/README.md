# MOS Prediction Datasets: Landscape & Multilingual Gap Analysis

Survey of publicly available MOS (Mean Opinion Score) prediction datasets, covering their language scope, size, and annotation density. Compiled as background for a new multilingual MOS dataset initiative.

---

## English-Only Datasets (open, publicly available)

| Dataset | Utterances | Systems | Ratings/Utt | Total Ratings | Domain | Paper |
|---------|-----------|---------|-------------|---------------|--------|-------|
| BVCC | 7,106 | 187 | 8 | ~56,848 | TTS/VC naturalness | [Cooper et al., 2022](https://arxiv.org/abs/2105.02373) |
| SOMOS | 20,100 | 200 | ~18.7 | 374,955 | TTS naturalness | [Maniati et al., 2022](https://arxiv.org/abs/2204.03040) |
| PSTN Corpus | 58,709 | — | variable | — | Phone call / DNS quality | [Reddy et al., 2021](https://arxiv.org/abs/2010.15258) |
| NISQA | 14,472 | ~350 conditions | 5–30 | >97,000 | Codec / transmission quality | [Mittag et al., 2021](https://arxiv.org/abs/2104.09494) |
| VCC2018 | 20,580 | 38 | 4 | 82,304 | Voice conversion naturalness | [Lorenzo-Trueba et al., 2018](https://arxiv.org/abs/1804.04262) |
| QualiSpeech | 14,577 | — | ~1 | — | Multi-aspect quality + NL feedback | [Yi et al., 2025](https://arxiv.org/abs/2503.20290) |

**Notes:**
- NISQA includes German-language test splits (TEST_LIVETALK, TEST_NSC) but its training data is English.
- BVCC aggregates Blizzard Challenges, VCC2016, VCC2018, VCC2020, and ESPnet-TTS systems into a unified listening test; it is the standard training set for VoiceMOS Challenge 2022 and later editions.
- SOMOS is the largest English TTS MOS dataset by total rating count, but covers only a single speaker (LJSpeech) and vocoder (LPCNet), limiting its generalization.

---

## Multilingual and Non-English Datasets (open)

| Dataset | Language(s) | Utterances | Ratings/Utt | Domain | Challenge | Paper |
|---------|------------|-----------|-------------|--------|-----------|-------|
| VoiceMOS 2022 OOD | Mandarin (Blizzard 2019) | 1,352 | 10–17 | TTS naturalness | VMC 2022 | [Huang et al., 2022](https://arxiv.org/abs/2203.11389) |
| VoiceMOS 2023 Track 1a | French | ~882 | 15 | TTS naturalness | VMC 2023 | [Cooper et al., 2023](https://arxiv.org/abs/2310.02640) |
| VoiceMOS 2023 Track 1b | French | ~578 | 15 | TTS naturalness | VMC 2023 | [Cooper et al., 2023](https://arxiv.org/abs/2310.02640) |
| VoiceMOS 2023 Track 3 / TMHINT-QI | Taiwanese Mandarin | 11,053 | ~5.3 | Speech enhancement | VMC 2023 | [Cooper et al., 2023](https://arxiv.org/abs/2310.02640) |
| VoiceMOS 2024 Track 2 | Chinese, Japanese | ~3,200 | ~5 | Singing voice synthesis | VMC 2024 | [Huang et al., 2024](https://arxiv.org/abs/2409.07001) |
| SingMOS / SingMOS-Pro | Chinese, Japanese | 6,583 / 7,981 | professional | Singing voice quality | VMC 2024 | [Zhang et al., 2024](https://arxiv.org/abs/2406.10911) |
| Tencent Corpus | Mandarin | ~14,000 | >20 | Conferencing quality | ConferencingSpeech 2022 | [Yi et al., 2022](https://arxiv.org/abs/2203.16032) |
| IndicMOS / LIMMITS | Hindi, Telugu, Marathi, Bengali, Kannada, Chhattisgarhi, English | small | ~1 | TTS naturalness | LIMMITS 2023/24 | [Udupa et al., 2024](https://www.isca-archive.org/interspeech_2024/udupa24b_interspeech.pdf) |

**Notes:**
- The French VMC 2023 data (Tracks 1a/1b) is the only open dataset for European-language TTS naturalness MOS; combined it has ~1,460 utterances — too small for robust training.
- TMHINT-QI covers speech enhancement, not TTS naturalness; it cannot be directly used for TTS MOS prediction without domain adaptation.
- IndicMOS/LIMMITS is the only dataset touching Indian languages but has **~1 rating per utterance**, which is insufficient for reliable utterance-level MOS prediction.
- SingMOS covers singing voice only; acoustic characteristics differ substantially from speech synthesis.

---

## The Closed Reference: SQuId (Google, not public)

The only large-scale multilingual MOS dataset for TTS naturalness is **SQuId** ([Chiang et al., 2023](https://arxiv.org/abs/2210.06324), ICASSP 2023), an internal Google dataset:

| Fact | Value |
|------|-------|
| Languages | 42+ locales for training, 65 for evaluation |
| Utterances | ~1.38 million |
| Total ratings | ~1.9 million |
| TTS systems | 1,476 |
| Scale | 1–9 (not standard 1–5) |
| Availability | Internal only — never released |

**Key SQuId findings:**
- Training across 42+ languages outperforms monolingual baselines for every language, including zero-shot generalization to 23 unseen locales.
- MOS naturalness is "para-lingual": it depends on acoustic artifact patterns and voice quality, not linguistic content. This is why cross-lingual transfer works even across maximally distant language families.

SQuId defines the ceiling. No open equivalent exists.

---

## Meta-Benchmark: MOS-Bench

[MOS-Bench](https://arxiv.org/abs/2411.03715) (2024) is not a new annotation effort but a standardized multi-dataset benchmark that aggregates existing corpora into unified training/evaluation splits:

**Training datasets included:** BVCC, SOMOS (14,100 samples), PSTN (52,839 samples), NISQA, TMHINT-QI, SingMOS, Tencent Corpus.

**Test sets cover:** English, French, Taiwanese Mandarin, Mandarin, Chinese, Japanese.

**Key MOS-Bench finding:** Acoustic diversity beats dataset size. NISQA (14K samples, diverse degradation conditions) and PSTN (58K, phone calls) generalize better across domains than BVCC (7K, synthetic-only). Domain and condition diversity in the acoustic space is the primary driver of cross-domain generalization.

---

## Cross-Lingual Transfer: What Is Already Known

English-trained MOS models tested zero-shot on non-English languages ([arXiv:2502.13004](https://arxiv.org/abs/2502.13004)):

| Target Language | NISQA CNN (Pearson) | AST Transformer (Pearson) |
|----------------|---------------------|--------------------------|
| Mandarin | 0.835 | 0.792 |
| French | 0.826 | 0.790 |
| German | 0.800 | 0.768 |
| Dutch | 0.798 | 0.790 |
| Swedish | 0.673 | 0.649 |

**Observations:**
- Mandarin transfers best despite maximal linguistic distance from English — supporting the para-lingual hypothesis.
- Swedish transfers worst, suggesting acoustic underrepresentation in English training data matters more than linguistic proximity.
- The **Discontinuity** MOS dimension resists cross-lingual transfer most strongly; overall quality transfers best.
- Transformer architectures are more stable cross-lingually than CNNs.

**VoiceMOS 2022 OOD track:** With only 136 labeled Mandarin samples in a semi-supervised setup, meaningful MOS prediction transfer from English-trained models was demonstrated — few-shot cross-lingual adaptation is feasible.

---

## Dataset Size Requirements for a New Multilingual Dataset

Based on findings from the literature:

### What the data says

**System diversity is the bottleneck, not utterance count** ([Tseng et al., 2022](https://arxiv.org/abs/2110.02635)):
The number of distinct TTS/VC architectures seen during training matters more than raw utterance count. "Unseen systems are especially challenging for MOS prediction models."

**Diminishing returns above 30% of available data** ([Leng et al., 2023](https://arxiv.org/abs/2305.19396)):
For fine-tuning SSL-based MOS predictors on low-resource language data, using more than 30% of available labeled samples does not yield significant further improvement. A few hundred labeled samples can be sufficient when adapting a pre-trained model.

**Multi-dataset training consistently helps** (MOS-Bench, 2024):
Naive pooling of multiple training datasets improves cross-domain generalization without hurting in-domain performance. The main challenge is **corpus effect** — different MOS scale offsets across different listening tests — which requires inter-test calibration.

### Recommended annotation density

| Objective | Ratings/Utterance | Utterances/System | Systems/Language | Total Ratings/Language |
|-----------|------------------|-------------------|-----------------|----------------------|
| System-level prediction (minimum) | 4–5 | 20–50 | 10–15 | ~10K–22K |
| Utterance-level prediction (recommended) | 8–10 | 50–100 | 20–30 | ~80K–150K |

For a dataset covering **10 languages** at the recommended level: **800K–1.5M total ratings** — matching SQuId in scale. A more practical academic target: **5 languages × 50K ratings = 250K ratings**, which would already be the largest open multilingual TTS MOS dataset by a significant margin.

---

## Summary of Gaps

| Gap | Current state |
|-----|--------------|
| Open multilingual TTS naturalness MOS | Only French (~1,460 utts) and Mandarin (~1,352 utts) — both too small to train on |
| European languages (DE, ES, IT, PT, NL) | **Not represented in any public MOS dataset** |
| Low-resource / African / SE Asian languages | **No public MOS data exists** |
| Rating density in multilingual efforts | IndicMOS has ~1 rating/utt; unusable for training |
| Open equivalent of SQuId | **Does not exist** |

The open multilingual MOS landscape covers a few thousand utterances of French, Mandarin, and singing voice. An open dataset covering 5+ language families with 8+ ratings/utterance and 15+ systems per language would be a substantial contribution and the most comprehensive open multilingual TTS MOS resource available.

---

## References

- Cooper, E. et al. (2022). *Generalization Ability of MOS Prediction Networks.* [arXiv:2105.02373](https://arxiv.org/abs/2105.02373)
- Cooper, E. et al. (2023). *The VoiceMOS Challenge 2023.* [arXiv:2310.02640](https://arxiv.org/abs/2310.02640)
- Chiang, C. et al. (2023). *SQuId: Measuring Speech Naturalness Across the World's Languages.* [arXiv:2210.06324](https://arxiv.org/abs/2210.06324)
- Huang, W.-C. et al. (2022). *The VoiceMOS Challenge 2022.* [arXiv:2203.11389](https://arxiv.org/abs/2203.11389)
- Huang, W.-C. et al. (2024). *The VoiceMOS Challenge 2024 (Track 2: Singing Voice).* [arXiv:2409.07001](https://arxiv.org/abs/2409.07001)
- Leng, Y. et al. (2023). *MBNet: MOS Prediction for Synthesized Speech with Mean-Bias Network.* [arXiv:2305.19396](https://arxiv.org/abs/2305.19396)
- Lorenzo-Trueba, J. et al. (2018). *The Voice Conversion Challenge 2018.* [arXiv:1804.04262](https://arxiv.org/abs/1804.04262)
- Maniati, G. et al. (2022). *SOMOS: The Samsung Open MOS Dataset.* [arXiv:2204.03040](https://arxiv.org/abs/2204.03040)
- Mittag, G. et al. (2021). *NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction.* [arXiv:2104.09494](https://arxiv.org/abs/2104.09494)
- Udupa, S. et al. (2024). *IndicMOS: A Multilingual MOS Prediction Dataset for Indian Languages.* [ISCA Archive (INTERSPEECH 2024)](https://www.isca-archive.org/interspeech_2024/udupa24b_interspeech.pdf)
- Reddy, C.K.A. et al. (2021). *DNSMOS P.835: A Non-Intrusive Perceptual Objective Speech Quality Metric.* [arXiv:2010.15258](https://arxiv.org/abs/2010.15258)
- Tseng, W. et al. (2022). *Toward Robust Training for Automatic MOS Prediction.* [arXiv:2110.02635](https://arxiv.org/abs/2110.02635)
- Yi, G. et al. (2022). *ConferencingSpeech 2022 Challenge: Non-intrusive Objective Speech Quality Assessment (NISQA) Challenge for Online Conferencing Applications.* [arXiv:2203.16032](https://arxiv.org/abs/2203.16032)
- Yi, J. et al. (2025). *QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning and Descriptions.* [arXiv:2503.20290](https://arxiv.org/abs/2503.20290)
- Zhang, J. et al. (2024). *SingMOS: Multi-Model MOS Corpus for Singing Voice Synthesis.* [arXiv:2406.10911](https://arxiv.org/abs/2406.10911)
- (2024). *MOS-Bench: A Benchmark for Automatic MOS Prediction.* [arXiv:2411.03715](https://arxiv.org/abs/2411.03715)
- (2025). *Cross-lingual Zero-Shot MOS Prediction.* [arXiv:2502.13004](https://arxiv.org/abs/2502.13004)

<!-- Verified 2026-10-23: arXiv:2203.16032 is the ConferencingSpeech 2022 (NISQA) challenge paper, Gaoxiong Yi et al., submitted 30 Mar 2022. BVCC is arXiv:2105.02373, so the two never shared an ID; the earlier suspicion was unfounded. Tencent corpus figures in the table (~10k Mandarin clips without reverb + ~4k with reverb, P.808-style crowdsourcing, >24 raters per clip with >20 retained) come from Section 3 of that paper. -->
