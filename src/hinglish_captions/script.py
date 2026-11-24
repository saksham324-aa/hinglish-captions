"""Read a caption script: what the viewer sees, and what the speaker says.

Format (plain UTF-8 text):

    Toh chalo, aaj 60{sixty} second ki reel ka math karte hain.
    Ek reel = 18{eighteen} ghante.

    Blank line = start a new caption here.

* Words are separated by spaces. What you type is what appears on screen,
  punctuation included.
* ``shown{spoken words}`` when the screen shows something the speaker says
  differently: numbers, symbols, short forms. ``₹500{paanch sau rupaye}``,
  ``AI{a i}``, ``60{sixty}``.
* Spoken words must be Roman letters (a-z and '), the way they sound.
  Hinglish in Roman script works as is: "kya", "hai", "nahi".
* A blank line forces a caption break before the next word.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"(\S+?)\{([^}]*)\}|(\S+)")


@dataclass
class Word:
    shown: str             # text on screen
    spoken: list[str]      # one or more spoken words, a-z and ' only
    break_before: bool = False


class ScriptError(ValueError):
    pass


def spoken_form(text: str) -> list[str]:
    """Lower-case a-z/' words of ``text``; empty list if nothing is speakable."""
    out = []
    for part in text.split():
        w = re.sub(r"[^a-z']", "", part.lower()).strip("'")
        if w:
            out.append(w)
    return out


def parse(text: str) -> list[Word]:
    words: list[Word] = []
    pending_break = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            pending_break = bool(words)
            continue
        for m in _TOKEN.finditer(line):
            if m.group(3) is not None:
                shown, spoken = m.group(3), spoken_form(m.group(3))
                if not spoken:
                    raise ScriptError(
                        f"line {lineno}: {shown!r} has no Roman letters to align. "
                        f"Write it as {shown}{{how it is said}}, e.g. 60{{sixty}}."
                    )
            else:
                shown, spoken = m.group(1), spoken_form(m.group(2))
                if not spoken:
                    raise ScriptError(f"line {lineno}: {{{m.group(2)}}} after {shown!r} has no Roman letters.")
            words.append(Word(shown, spoken, pending_break))
            pending_break = False
    if not words:
        raise ScriptError("the script is empty")
    return words
