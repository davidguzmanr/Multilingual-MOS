# Annotator tutorial (English)

Eight clips, all in the LJSpeech voice, that show what the scores mean on the two questions of the study:
**naturalness** (how human the voice sounds) and **intelligibility** (whether the audio says exactly the
reference text). Degraded clips come from real TTS systems trained on LJSpeech (Coqui VITS and Neural-HMM)
and reproduce errors annotators will meet in the study: synthetic voice quality, a model that falls apart,
mispronounced words, a skipped word. Rebuild with `python listening-study/tutorial/build_tutorial.py` (env `Multilingual-MOS`,
GPU). Open `index.html` in a browser to listen to all clips with their target scores and notes.

| # | File | Naturalness | Intelligibility | What was done | Talking point |
|---|---|:-:|:-:|---|---|
| 1 | `01_nat5_int5.wav` | 5 | 5 | Original recording | The reference: a real person, every word right |
| 2 | `02_nat3_int4.wav` | 3 | 4 | Neural-HMM, an older TTS | Clearly synthetic: flat, muffled and buzzy; one word ('purpose') is unclear |
| 3 | `03_nat2_int3.wav` | 2 | 3 | Robotization (phase zeroed) | Robotic and tiring; the buzz makes a few words unclear |
| 4 | `04_nat1_int2.wav` | 1 | 2 | VITS sampled with far too much noise | The model falls apart: garbled, barely speech; only fragments of the text survive |
| 5 | `05_nat4_int3.wav` | 4 | 3 | VITS says "these these" and "hoose" for "house" | A repeated word and a mispronounced word: two words off |
| 6 | `06_nat5_int2.wav` | 5 | 2 | Audio cut after "after lapsing into crime," | Missing end of sentence is an intelligibility problem |
| 7 | `07_nat5_int1.wav` | 5 | 1 | Audio is a different sentence | Natural voice, completely wrong content |
| 8 | `08_nat3_int3.wav` | 3 | 3 | Neural-HMM skips "ordinary" and says "wahrds" | Both problems at once: rate each separately |

Clips 1 to 4 use the same sentence so the naturalness steps are easy to hear. The words never change in
them, but as the voice degrades the words get harder to make out, so intelligibility drops along with
naturalness (5, 4, 3, 2). All clips are
loudness-matched to the natural recording of their sentence. Full metadata, including the exact text
shown, is in `examples.csv`.

**Intelligibility scale.** The question in `labeling_config.xml` asks whether all the words are clear and
the same as in the text, and the anchors count words that are unclear, wrong, missing, added or repeated,
or cut off: 5 = every word correct; 4 = one word off; 3 = a few words off (two or three); 2 = many words
off, or part of the text skipped or cut off; 1 = mostly wrong, a different sentence or language, or no
words made out. The instructions add three rules: spelling and missing accent marks in the text are not
errors, a wrong tone or vowel length that makes a different word is an error, and a word you cannot make
out counts even if you can guess it from the text. The naturalness question tells raters to judge only how
the voice sounds, not whether the words are correct. Rationale and sources: `reports/Intelligibility in
TTS listening tests.md` (local, not tracked).

**How the errors were made.** Mispronunciations are G2P errors, the way TTS systems actually make them:
the phonemes of a word are replaced before synthesis (`hˈaʊs` → `hˈuːs`, `wˈɔːɹdz` → `wˈɑːɹdz`) and the
rest is the model's own output. Repeated and skipped words are made by editing the input text ("these these"
in clip 5, "ordinary" left out in clip 8), as happens when an attention-based TTS stutters or skips. The
errors in clip 5 are in everyday words so non-native English listeners hear them easily. Clip 4 is VITS
with `noise_scale` 3.0 instead of 0.667.

**Checks run on the clips.** Whisper-small transcribes every word of clips 1 to 3 correctly; their
intelligibility scores of 4 and 3 reflect how hard the words are for a person to make out, not word
errors. On clip 4 it recovers only fragments ("Maybe consider it as
it or a complimenting box, but it is a movable type."). Whisper corrects mispronunciations back to the
intended word, so clips 5 and 8 were checked with a phoneme recognizer (wav2vec2-lv-60-espeak-cv-ft), which
hears `h uː s` and `w ɑːɹ ɹ uː d`. Whisper itself writes clip 5 as "Two of these, these photographs, were
taken from the rear of Walker's hoose." and confirms "ordinary" is missing in clip 8. The naturalness ladder has not been scored with UTMOS22. Listen to all eight before recording: the
target scores are a teaching aid, not ground truth.

## Setting up the HumanSignal project

1. Create a project and paste `labeling_config.xml` into the labeling interface. This is the
   two-question layout intended for the real survey too.
2. Push this branch, then import `tasks.json`. Its audio URLs point at raw GitHub files on the
   `dev/tutorial` branch, so the clips only load once the current wavs are pushed there. To host them
   elsewhere (e.g. the Hugging Face dataset repo), rebuild with
   `--audio-base-url https://huggingface.co/datasets/<user>/<repo>/resolve/main/tutorial`.
   If the project already has tasks from an earlier version, delete them before importing.
3. Tasks carry `expected_naturalness` and `expected_intelligibility` in their data. They are not shown in
   the interface but are handy if you later reuse these clips as a qualification test.

## Recording script (about 5 minutes)

Read the quoted lines aloud. Lines in *[brackets]* are cues for you, not to be read. Speak slowly:
most annotators are not native English speakers. Record with the HumanSignal tutorial project open
(the 8 clips, in order). Clip numbers match the task order and the files in `audios/`.

### 1. Welcome (20 s)

*[Show the first task in HumanSignal.]*

"Hello, and thank you for joining this study.

In each task, you will hear a short clip of speech. Most clips are made by computers. Some are real
people.

On the screen, you always see the text that the voice should say.

For each clip, you answer two questions."

### 2. The screen and the two questions (90 s)

*[Point to each part of the screen as you mention it.]*

"At the top, you see the instructions. Below, in the grey box, is the reference text. This is what
the voice should say. Then you have the audio player. Listen to the whole clip. You can play it
again as many times as you need.

Question one is **naturalness**. How natural does the voice sound? Does it sound like a real person
speaking? Listen to the rhythm, the melody of the voice, and any buzzing, metallic or robotic sound.
Judge only how it sounds, not whether the words are correct. Five means it sounds like a real person.
One means it barely sounds like speech.

Question two is **intelligibility**. Are all the words clear, and the same as in the text? Listen
while you read the text, and count the words with a problem: words that are unclear, wrong, missing,
added or repeated, or cut off. Five means every word is correct and clear. Four means one word has a
problem. Three means two or three words. Two means many words, or a part of the text is missing. One
means most of the text is wrong, or it is a different sentence.

Three rules. Judge the spoken words, not the spelling: if the text spells a word differently, or has
no accent marks, that is not an error. A regional accent is not an error if it is still the same
word. But if a wrong tone or a wrong vowel length makes it a different word, count it as a wrong
word. And a word you cannot make out counts as an error, even if you can guess it from the text.

If you are not sure, move your mouse over an option to see a short description.

These are two different questions. A natural voice can say the wrong words. A bad voice can say
the right words. Answer each question on its own.

Now let's listen to some examples."

### 3. Naturalness: clips 1 to 4 (90 s)

"The first four clips all say the same sentence: *Printing, then, for our purpose, may be
considered as the art of making books by means of movable types.* Listen to how the voice changes."

*[Play clip 1.]*

"This is a real person. It sounds natural, and every word is clear. Naturalness five.
Intelligibility five."

*[Play clip 2.]*

"This is an older computer voice. It is flat and a bit muffled. You can hear it is not a person.
Naturalness three. All the words are the right ones, but the muffled sound makes one word unclear:
'purpose'. One word off. Intelligibility four."

*[Play clip 3.]*

"This voice is robotic, with a strong buzz. It is tiring to listen to. Naturalness two. The words
are correct, but the buzz makes two or three of them unclear. A few words off. Intelligibility
three."

*[Play clip 4.]*

"Here the computer voice breaks down. It is garbled, and it barely sounds like speech.
Naturalness one. You can only catch a few pieces, like 'considered as' and 'movable types'. Many
words are lost. Intelligibility two.

A voice that sounds almost human, but a little too smooth or too even, is a naturalness four. It
sits between clip one and clip two."

### 4. Intelligibility: clips 5 to 7 (60 s)

*[Play clip 5. Point at the text: "Two of these photographs were taken from the rear of Walker's house."]*

"This is a good computer voice. It sounds almost human. Naturalness four. Now look at the words.
The voice says 'these' two times: 'these these'. And at the end, it says 'hoose' instead of
'house'. That is two words off. Intelligibility three. With only one word off, it would be a four."

*[Play clip 6.]*

"This is a real person, so naturalness is five. But the clip stops in the middle. The end of the
sentence is missing. Intelligibility two."

*[Play clip 7.]*

"This is also a real person. Naturalness five. But the voice reads a completely different sentence
from the text on the screen. Intelligibility one."

### 5. Both problems at once: clip 8 (30 s)

*[Play clip 8. Point at the text: "Sick women and their children remained in the ordinary wards, and propagated disease."]*

"Real clips often have both kinds of problems. This is the older computer voice again, flat and
muffled. Naturalness three. It also skips the word 'ordinary', and it says 'wards' wrongly. Two
words off. Intelligibility three.

Remember: answer each question on its own."

### 6. Practical tips (30 s)

"Some practical tips. Please use headphones, in a quiet room. Always read the text, and listen to
the whole clip at least once. You can replay it as many times as you need.

For naturalness, there is no right or wrong answer. We want your honest opinion. Please use the
whole scale, from one to five.

Each clip should take about twenty to thirty seconds.

Thank you for your help!"
