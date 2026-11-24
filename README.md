# hinglish-captions

Word-exact captions for Hinglish talking-head reels.

Auto-captions usually get the words right and the timing wrong: a word pops up a
beat before it is said, or lines change mid-phrase. On short vertical video that
looks cheap. `hcap` takes the words you already know (your script, or a fixed-up
transcript) and puts every one of them exactly where it is spoken:

* **Forced alignment**, not transcription: it never guesses the words, it only finds them.
  It works on Hinglish written the way it sounds ("kya", "nahi", "chalo"), with no Hindi dictionary.
* **Never early:** after a pause each word moves to the measured start of the voice, inside
  a phrase to the nearest loudness rise. Onsets are rounded up to the video frame.
* **Shown vs spoken:** write `60{sixty}` or `₹500{paanch sau rupaye}` and the screen shows the
  numbers while the aligner listens for the words.
* **Reel-ready output:** short lines (24 characters, sentence ends, pauses),
  SRT, WebVTT, JSON with every word, and a 1080x1920 **karaoke ASS** where each word lights up as it is said.

## Install

Needs Python 3.10+ and [ffmpeg](https://ffmpeg.org) on PATH.

```bash
pip install git+https://github.com/saksham324-aa/hinglish-captions
```

The first run downloads the alignment model (about 1.2 GB) from Hugging Face. A CPU is enough:
roughly 30-50 s for a 90-second reel.

## Use

Write what is said in a text file:

```text
Doston, aaj baat karte hain chai ki.
Ek cup chai mein sirf 2{do} minute lagte hain.

Lekin sahi chai? Usmein patience lagta hai.
```

Then:

```bash
hcap reel.mp4 script.txt                      # writes reel.srt, reel.ass, reel.json
hcap reel.mp4 script.txt --formats srt,vtt -o captions/reel
```

Burn the karaoke captions into the video:

```bash
ffmpeg -i reel.mp4 -vf "ass=reel.ass" -c:a copy reel-captioned.mp4
```

### Script rules

| You write | Screen shows | Aligner listens for |
| --- | --- | --- |
| `chalo,` | chalo, | chalo |
| `60{sixty}` | 60 | sixty |
| `₹500{paanch sau rupaye}` | ₹500 | paanch sau rupaye |
| `AI{a i}` | AI | a i |
| a blank line | starts a new caption | |

Words the speaker does not say (a number written as digits, a symbol) must get a `{spoken}`
form, and `hcap` tells you which line needs one. After a run it lists low-confidence words:
usually a word said differently from how it is written.

### Options

| Option | Default | |
| --- | --- | --- |
| `--fps` | 30 | frame grid onsets are rounded up to |
| `--max-chars` | 24 | longest caption line |
| `--max-words` | 5 | most words on one line |
| `--no-snap` | off | keep the aligner's raw onsets |
| `--font`, `--size` | Poppins, 84 | ASS style at 1080x1920 |
| `--model` | MMS forced aligner | any Hugging Face CTC model with a Roman-letter vocabulary |

## How it works

1. **Audio** is read with ffmpeg as 16 kHz mono.
2. **Emissions:** a multilingual CTC model (Meta's MMS, 1,100+ languages) scores every letter
   for every 20 ms frame, in 30 s windows with 2 s of context.
3. **Viterbi** finds the single best path of your script's letters through those frames,
   giving each word a start, an end and a confidence.
4. **Snap:** a 5 ms loudness envelope finds where the voice starts after each pause and the sharp
   rises inside phrases; onsets move onto them and are rounded up to the frame.
5. **Group** into short lines and write the formats.

The snapping step came from editing reels by hand: with it, every caption appeared within
0-45 ms after its word began, where aligner and Whisper timestamps were 0.2-0.4 s off.

## Model licence

The code here is MIT. The default model,
[`MahmoudAshraf/mms-300m-1130-forced-aligner`](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner),
is released under **CC BY-NC 4.0** (non-commercial). It only produces timings and nothing of it ends up
in your video, but check the licence for your use. You can pass any other CTC model with `--model`.

## Limits

* Spoken words must be Roman letters a-z. Devanagari scripts need romanising first.
* One alignment holds the whole file in memory: fine for reels and shorts (up to a few minutes),
  not for hour-long podcasts.
* The script must match what is said. A skipped or added word pulls its neighbours with it.

## Tests

```bash
pip install -e .
python -m unittest discover -s tests
```
