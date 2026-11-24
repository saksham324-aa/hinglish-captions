"""Tests that need no model download: python -m unittest discover -s tests"""
import unittest

import numpy as np

from hinglish_captions.align import viterbi
from hinglish_captions.captions import CapWord, group, to_ass, to_srt, to_vtt
from hinglish_captions.script import ScriptError, parse
from hinglish_captions.snap import snap, voice_starts


class ScriptTests(unittest.TestCase):
    def test_plain_and_braced(self):
        ws = parse("Toh chalo, 60{sixty} second ki reel!\n\n₹500{paanch sau} bas")
        self.assertEqual([w.shown for w in ws], ["Toh", "chalo,", "60", "second", "ki", "reel!", "₹500", "bas"])
        self.assertEqual(ws[1].spoken, ["chalo"])
        self.assertEqual(ws[2].spoken, ["sixty"])
        self.assertEqual(ws[6].spoken, ["paanch", "sau"])
        self.assertTrue(ws[6].break_before)
        self.assertFalse(ws[7].break_before)

    def test_number_without_spoken_form_fails(self):
        with self.assertRaises(ScriptError) as e:
            parse("ek reel = 18 ghante")
        self.assertIn("{how it is said}", str(e.exception))

    def test_empty(self):
        with self.assertRaises(ScriptError):
            parse("\n \n")


class ViterbiTests(unittest.TestCase):
    def test_finds_letters_where_they_are(self):
        # alphabet: 0 blank, 1 'a', 2 'b'; 'a' at frames 3-4, 'b' at frames 8-9
        T = 12
        em = np.full((T, 3), -9.0)
        em[:, 0] = -0.1
        for t in (3, 4):
            em[t] = [-9, -0.1, -9]
        for t in (8, 9):
            em[t] = [-9, -9, -0.1]
        first, last, _ = viterbi(em, [1, 2], blank=0)
        self.assertEqual((first[0], last[0]), (3, 4))
        self.assertEqual((first[1], last[1]), (8, 9))

    def test_repeated_letter_needs_a_blank_between(self):
        T = 10
        em = np.full((T, 2), -9.0)
        em[:, 0] = -0.1
        em[2] = [-9, -0.1]
        em[6] = [-9, -0.1]
        first, _, _ = viterbi(em, [1, 1], blank=0)
        self.assertEqual((first[0], first[1]), (2, 6))

    def test_too_short(self):
        with self.assertRaises(ValueError):
            viterbi(np.zeros((1, 3)), [1, 2])


def _env(segments, total=3.0):
    """Synthetic 5 ms loudness envelope: -80 dB silence, -20 dB speech in the given (start, end) spans."""
    e = np.full(int(total / 0.005), -80.0)
    for a, b in segments:
        e[int(a / 0.005):int(b / 0.005)] = -20.0
    return e


class SnapTests(unittest.TestCase):
    def test_voice_starts_after_pauses(self):
        self.assertEqual(voice_starts(_env([(0.5, 1.0), (1.5, 2.0)])), [0.5, 1.5])

    def test_early_onset_moves_to_voice_never_before(self):
        on, ap = snap([0.42, 0.7, 1.45], _env([(0.5, 1.0), (1.5, 2.0)]), fps=30)
        self.assertGreaterEqual(on[0], 0.5)
        self.assertGreaterEqual(on[2], 1.5)
        self.assertEqual(ap, [True, False, True])
        for t in on:  # every onset sits on the frame grid
            self.assertAlmostEqual(t * 30, round(t * 30), places=3)

    def test_order_kept(self):
        on, _ = snap([1.0, 1.0, 1.0], _env([]), fps=30)
        self.assertTrue(on[0] < on[1] < on[2])


class CaptionTests(unittest.TestCase):
    def words(self):
        return [CapWord("Toh", 0.5, 0.7), CapWord("chalo,", 0.72, 1.0), CapWord("aaj", 1.02, 1.2),
                CapWord("math.", 1.22, 1.5), CapWord("Ek", 2.2, 2.4), CapWord("reel", 2.42, 2.7)]

    def test_breaks_on_sentence_and_pause(self):
        gs = group(self.words())
        self.assertEqual([g.text for g in gs], ["Toh chalo, aaj math.", "Ek reel"])
        self.assertAlmostEqual(gs[0].end, 1.9)  # holds 0.4 s, ends before the next line

    def test_max_chars(self):
        gs = group(self.words(), max_chars=10)
        self.assertTrue(all(len(g.text) <= 10 or len(g.words) == 1 for g in gs))

    def test_formats(self):
        gs = group(self.words())
        srt = to_srt(gs)
        self.assertIn("1\n00:00:00,500 --> 00:00:01,900\nToh chalo, aaj math.", srt)
        self.assertTrue(to_vtt(gs).startswith("WEBVTT\n\n00:00:00.500 --> 00:00:01.900"))
        ass = to_ass(gs)
        self.assertIn("Dialogue: 0,0:00:00.50,0:00:01.90,Default,,0,0,0,,{\\k22}Toh {\\k30}chalo,", ass)


if __name__ == "__main__":
    unittest.main()
