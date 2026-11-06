# Annotator tutorial (English)

Eight LJSpeech clips that show what each score means on the two questions of the study:
**naturalness** (how human the voice sounds) and **accuracy** (whether the audio says exactly the
reference text). Each example changes one axis at a time, so annotators learn to rate them
independently. Rebuild with `python listening-study/tutorial/build_tutorial.py`.

| # | File | Naturalness | Accuracy | What was done | Talking point |
|---|---|:-:|:-:|---|---|
| 1 | `01_nat5_acc5.wav` | 5 | 5 | Original recording | The reference: a real person, every word right |
| 2 | `02_nat4_acc5.wav` | 4 | 5 | Griffin-Lim, 100 iterations | Still human, slightly hollow if you listen closely |
| 3 | `03_nat3_acc5.wav` | 3 | 5 | Griffin-Lim, 16 iterations | Clearly synthetic and metallic, but every word is there |
| 4 | `04_nat2_acc5.wav` | 2 | 5 | Robotization (phase zeroed) | Robotic and tiring, yet accuracy is still 5 |
| 5 | `05_nat5_acc3.wav` | 5 | 3 | Text shows "beautiful", audio says "delightful" | One wrong word; the voice is still natural |
| 6 | `06_nat5_acc2.wav` | 5 | 2 | Audio cut after "after lapsing into crime," | Missing end of sentence is an accuracy problem |
| 7 | `07_nat5_acc1.wav` | 5 | 1 | Audio is a different sentence | Natural voice, completely wrong content |
| 8 | `08_nat3_acc3.wav` | 3 | 3 | Example 3's artifacts plus "weekly" vs "daily" | Both problems at once: rate each separately |

Clips 1 to 4 use the same sentence so the naturalness steps are easy to hear. All clips are
loudness-matched. Full metadata, including the exact text shown, is in `examples.csv`.

**Checks run on the clips.** UTMOS22 orders the naturalness ladder as intended
(4.40, 3.65, 2.73, 1.27 for clips 1 to 4), and Whisper-small transcribes every word of clips 1 to 4
and 8 correctly, so the degradations do not affect accuracy. Whisper also confirms clip 6 ends at
"after lapsing into crime," and clips 5 and 8 say "delightful" and "daily". Listen to all eight
before recording: the target scores are a teaching aid, not ground truth.

## Setting up the HumanSignal project

1. Create a project and paste `labeling_config.xml` into the labeling interface. This is the
   two-question layout intended for the real survey too.
2. Import `tasks.json`. It points at raw GitHub URLs on the `dev/tutorial` branch, which only load
   if the repository is public. **It is currently private**, so either upload the eight wavs to the
   Hugging Face dataset repo and rebuild with
   `--audio-base-url https://huggingface.co/datasets/<user>/<repo>/resolve/main/tutorial`, or push
   this branch to a public repo.
3. Tasks carry `expected_naturalness` and `expected_accuracy` in their data. They are not shown in
   the interface but are handy if you later reuse these clips as a qualification test.

## Recording script (about 4 minutes)

**Intro.** "Thanks for joining. You will listen to short clips of speech, most of them generated
by computers, and answer two questions about each one. On screen you always see the text the
voice was supposed to say."

**The two questions.** "Question one is naturalness: does it sound like a real person speaking?
Think about rhythm, intonation and any buzzing or metallic sound. For this question, ignore
whether the words are right. Question two is accuracy: does the audio say exactly the text on
screen? Listen for wrong, missing or extra words. For this question, ignore how robotic the voice
is. The two questions are independent: a robotic voice can be perfectly accurate, and a natural
voice can say the wrong words."

**Clips 1 to 4.** Play each one. "This first clip is a real person. Natural five, accuracy five.
This one is the same sentence, slightly hollow if you listen carefully: naturalness four. This one
is clearly synthetic and metallic: naturalness three. And this one is robotic and tiring:
naturalness two. Notice that in all four, every word matches the text, so accuracy stays at five."

**Clips 5 to 7.** "Now the voice is natural, so naturalness is five each time, but the words
change. Here the text says 'beautiful' and the voice says 'delightful': one wrong word, accuracy
three. Here the clip stops halfway and the end of the sentence is missing: accuracy two. And here
the voice reads a completely different sentence: accuracy one."

**Clip 8.** "Real clips often have both problems. This one is metallic, naturalness three, and has
one wrong word, accuracy three. Score each question on its own."

**Practical tips.** "Use headphones in a quiet room. Listen to each clip at least once all the
way through; you can replay it as often as you need. Read the text before or while listening.
There are no right or wrong opinions on naturalness, we want your honest impression as a native
speaker, so try to use the whole scale. Each clip should take about twenty to thirty seconds."
