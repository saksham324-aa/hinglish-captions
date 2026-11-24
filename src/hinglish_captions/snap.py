"""Snap word onsets to where the voice really starts, so a caption never shows early.

An aligner's onset can land a little before or after the sound. On a reel a
word that pops up before it is heard looks wrong, so:

1. After a pause (>= 60 ms of quiet) the next word moves to the measured voice start.
2. Any other word moves to a clear loudness rise within -30..+100 ms of its onset.
3. Onsets stay in order, at least two frames apart, and are rounded UP to a frame.
"""
from __future__ import annotations

import math

import numpy as np

HOP = 0.005


def voice_starts(e: np.ndarray, min_quiet: float = 0.06) -> list[float]:
    floor, peak = np.percentile(e, 5), np.percentile(e, 95)
    loud = e > floor + 0.3 * (peak - floor)
    need = int(round(min_quiet / HOP))
    out, quiet = [], 10**9
    for i, on in enumerate(loud):
        if on and quiet >= need:
            out.append(round(i * HOP, 3))
        quiet = 0 if on else quiet + 1
    return out


def rises(e: np.ndarray) -> list[float]:
    """Sharp loudness rises (consonant bursts, vowel onsets) inside running speech."""
    out = []
    mid = np.percentile(e, 40)
    for i in range(8, len(e) - 8):
        lo = e[i - 8 : i].min()
        if e[i + 6] - lo >= 9 and e[i] - lo < 4 and e[i + 1] - e[i] > 0.8 and e[i + 6] > mid:
            t = round(i * HOP, 3)
            if not out or t - out[-1] > 0.07:
                out.append(t)
    return out


def snap(onsets: list[float], e: np.ndarray, fps: float = 30.0) -> tuple[list[float], list[bool]]:
    """Return snapped onsets and, per word, whether it follows a pause."""
    on = list(onsets)
    after_pause = [False] * len(on)
    for s in voice_starts(e):
        cands = [(abs(s - t), i) for i, t in enumerate(on) if -0.12 <= s - t <= 0.2 and not after_pause[i]]
        if cands:
            i = min(cands)[1]
            on[i], after_pause[i] = s, True
    rs = rises(e)
    for i, t in enumerate(on):
        if not after_pause[i]:
            near = [r for r in rs if -0.03 <= r - t <= 0.10]
            if near:
                on[i] = near[0]
    gap = 2 / fps
    for i in range(1, len(on)):
        on[i] = max(on[i], on[i - 1] + gap)
    on = [round(math.ceil(t * fps - 1e-6) / fps, 4) for t in on]
    return on, after_pause
