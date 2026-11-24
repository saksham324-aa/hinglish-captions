"""Forced alignment: where does each known word sit in the recording?

Uses a multilingual CTC character model (MMS, 1100+ languages) that only needs
Roman letters, so Hinglish written the way it sounds aligns without a Hindi
dictionary. Resolution: 20 ms per frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SR

DEFAULT_MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
_HOP = 320  # samples per model frame at 16 kHz = 20 ms


@dataclass
class Timed:
    on: float
    off: float
    score: float  # mean log-probability per frame; below about -4 means "check this word"


def emissions(x: np.ndarray, model_name: str = DEFAULT_MODEL, window_s: int = 30, context_s: int = 2):
    """Per-frame log-probabilities over the model's characters, plus the vocabulary."""
    import torch
    from transformers import AutoModelForCTC, AutoTokenizer

    model = AutoModelForCTC.from_pretrained(model_name).eval()
    vocab = AutoTokenizer.from_pretrained(model_name).get_vocab()
    t = torch.from_numpy(x)
    win, ctx = window_s * SR, context_s * SR
    out = []
    with torch.inference_mode():
        for a in range(0, len(t), win):
            lo, hi = max(0, a - ctx), min(len(t), a + win + ctx)
            lg = torch.log_softmax(model(t[lo:hi][None]).logits[0], -1)
            f0 = (a - lo) // _HOP
            out.append(lg[f0 : f0 + (min(len(t), a + win) - a) // _HOP])
    return torch.cat(out).numpy(), vocab


def viterbi(em: np.ndarray, ids: list[int], blank: int = 0):
    """Best path through blank-interleaved CTC states.

    Returns (first, last, score): first and last frame of every token, and its
    summed log-probability.
    """
    T, S = em.shape[0], 2 * len(ids) + 1
    if T < len(ids):
        raise ValueError(f"audio too short: {T} frames for {len(ids)} letters")
    st = np.array([blank if s % 2 == 0 else ids[s // 2] for s in range(S)])
    skip = np.zeros(S, bool)  # may jump over a blank (s-2 -> s) between two different letters
    skip[3::2] = st[3::2] != st[1:-2:2]
    NEG = -1e30
    d = np.full(S, NEG)
    d[0] = em[0, blank]
    d[1] = em[0, st[1]]
    bp = np.zeros((T, S), np.int8)
    for t in range(1, T):
        one = np.concatenate(([NEG], d[:-1]))
        two = np.where(skip, np.concatenate(([NEG, NEG], d[:-2])), NEG)
        k = np.zeros(S, np.int8)
        best = d.copy()
        m = one > best
        best[m], k[m] = one[m], 1
        m = two > best
        best[m], k[m] = two[m], 2
        d = best + em[t, st]
        bp[t] = k
    s = S - 1 if S < 2 or d[S - 1] >= d[S - 2] else S - 2
    first, last, score = {}, {}, {}
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            i = s // 2
            first[i] = t
            last.setdefault(i, t)
            score[i] = score.get(i, 0.0) + float(em[t, st[s]])
        s -= int(bp[t, s])
    return first, last, score


def align_words(em: np.ndarray, vocab: dict, spoken: list[list[str]]) -> list[Timed]:
    """Time every script word (each a list of spoken words) against the emissions."""
    ids, owner = [], []
    for wi, words in enumerate(spoken):
        for ch in "".join(words):
            if ch not in vocab:
                raise ValueError(f"letter {ch!r} is not in the model's alphabet")
            ids.append(vocab[ch])
            owner.append(wi)
    blank = vocab.get("<blank>", vocab.get("<pad>", 0))
    first, last, score = viterbi(em, ids, blank)
    step = _HOP / SR
    out = []
    for wi in range(len(spoken)):
        ks = [k for k, o in enumerate(owner) if o == wi]
        n = sum(last[k] - first[k] + 1 for k in ks)
        out.append(
            Timed(
                on=round(first[ks[0]] * step, 3),
                off=round((last[ks[-1]] + 1) * step, 3),
                score=round(sum(score[k] for k in ks) / max(1, n), 2),
            )
        )
    return out
