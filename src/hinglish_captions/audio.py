"""Audio in: any file ffmpeg can read (wav, mp3, m4a, mp4, mov ...), as 16 kHz mono float32."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

SR = 16000


def load(path: str, sr: int = SR) -> np.ndarray:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg was not found on PATH. Install it from https://ffmpeg.org and try again.")
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    )
    x = np.frombuffer(raw, dtype=np.float32).copy()
    if not len(x):
        raise RuntimeError(f"no audio found in {path}")
    return x


def envelope_db(x: np.ndarray, sr: int = SR, hop: float = 0.005) -> np.ndarray:
    """Loudness in dB on a 5 ms grid, lightly smoothed."""
    h = int(sr * hop)
    m = len(x) // h
    rms = np.sqrt((x[: m * h].reshape(m, h).astype(np.float64) ** 2).mean(1))
    e = 20 * np.log10(rms + 1e-6)
    return np.convolve(e, np.ones(4) / 4, "same")
