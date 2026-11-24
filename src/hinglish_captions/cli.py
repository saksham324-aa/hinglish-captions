"""hcap: word-exact captions for Hinglish (and any Roman-script) talking-head video.

    hcap VIDEO_OR_AUDIO SCRIPT.txt [-o OUT] [--formats srt,ass,vtt,json]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from .captions import CapWord, group, to_ass, to_json, to_srt, to_vtt
from .script import ScriptError, parse

FORMATS = {"srt", "vtt", "ass", "json"}


def build(media: str, script_text: str, fps: float = 30.0, snap_to_voice: bool = True,
          model: str | None = None, log=print):
    from . import align, audio, snap

    words = parse(script_text)
    t0 = time.time()
    x = audio.load(media)
    log(f"audio: {len(x) / audio.SR:.1f} s")
    em, vocab = align.emissions(x, model or align.DEFAULT_MODEL)
    timed = align.align_words(em, vocab, [w.spoken for w in words])
    log(f"aligned {len(words)} words in {time.time() - t0:.1f} s")
    onsets = [t.on for t in timed]
    after_pause = [False] * len(words)
    if snap_to_voice:
        # loudness at full bandwidth: sibilants above 8 kHz mark word starts too
        onsets, after_pause = snap.snap(onsets, audio.envelope_db(audio.load(media, 48000), 48000), fps)
    out = []
    for w, t, on, ap in zip(words, timed, onsets, after_pause):
        out.append(CapWord(w.shown, on, max(t.off, on + 1 / fps), t.score, ap, w.break_before))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hcap", description="Word-exact captions for Hinglish video.")
    p.add_argument("media", help="video or audio file (anything ffmpeg reads)")
    p.add_argument("script", help="text file with what is said (see README for the format)")
    p.add_argument("-o", "--out", help="output path without extension (default: next to the media)")
    p.add_argument("--formats", default="srt,ass,json", help="comma list of srt, vtt, ass, json")
    p.add_argument("--fps", type=float, default=30.0, help="frame rate to round onsets up to (default 30)")
    p.add_argument("--max-chars", type=int, default=24, help="longest caption line (default 24)")
    p.add_argument("--max-words", type=int, default=5, help="most words per caption (default 5)")
    p.add_argument("--no-snap", action="store_true", help="keep the aligner's onsets as they are")
    p.add_argument("--font", default="Poppins", help="ASS font name (default Poppins)")
    p.add_argument("--size", type=int, default=84, help="ASS font size at 1080x1920 (default 84)")
    p.add_argument("--model", help="Hugging Face CTC model id (default: MMS forced aligner)")
    p.add_argument("--version", action="version", version=f"hcap {__version__}")
    a = p.parse_args(argv)

    fmts = {f.strip() for f in a.formats.split(",") if f.strip()}
    if not fmts <= FORMATS:
        p.error(f"unknown format(s): {', '.join(sorted(fmts - FORMATS))}")
    try:
        with open(a.script, encoding="utf-8") as f:
            text = f.read()
        words = build(a.media, text, a.fps, not a.no_snap, a.model)
    except (ScriptError, RuntimeError, ValueError) as e:
        print(f"hcap: {e}", file=sys.stderr)
        return 1

    groups = group(words, a.max_chars, a.max_words)
    base = a.out or os.path.splitext(a.media)[0]
    writers = {"srt": lambda: to_srt(groups), "vtt": lambda: to_vtt(groups),
               "ass": lambda: to_ass(groups, font=a.font, size=a.size), "json": lambda: to_json(words, groups)}
    for fmt in sorted(fmts):
        path = f"{base}.{fmt}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(writers[fmt]())
        print(f"wrote {path}")
    weak = [w.text for w in words if w.score < -4]
    if weak:
        print("check these words (low confidence, maybe not said as written):", " ".join(weak))
    return 0


if __name__ == "__main__":
    sys.exit(main())
