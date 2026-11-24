"""Group timed words into short caption lines and write SRT, VTT, ASS and JSON."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class CapWord:
    text: str
    on: float
    off: float
    score: float = 0.0
    after_pause: bool = False
    break_before: bool = False


@dataclass
class Group:
    words: list[CapWord] = field(default_factory=list)
    end: float = 0.0

    @property
    def start(self) -> float:
        return self.words[0].on

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def group(words: list[CapWord], max_chars: int = 24, max_words: int = 5, pause: float = 0.3,
          hold: float = 0.4) -> list[Group]:
    """Short lines for vertical video. A new line starts at a forced break, after a pause,
    after a word that ends a sentence, or when the line would get too long."""
    groups: list[Group] = []
    cur = Group()
    for w in words:
        if cur.words:
            prev = cur.words[-1]
            too_long = len(cur.text) + 1 + len(w.text) > max_chars or len(cur.words) >= max_words
            gap = w.on - prev.off >= pause
            sentence_end = prev.text[-1:] in ".?!"
            if w.break_before or gap or sentence_end or too_long:
                groups.append(cur)
                cur = Group()
        cur.words.append(w)
    if cur.words:
        groups.append(cur)
    for i, g in enumerate(groups):
        last_off = g.words[-1].off + hold
        g.end = min(last_off, groups[i + 1].start) if i + 1 < len(groups) else last_off
    return groups


def _ts(t: float, sep: str = ",") -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(groups: list[Group]) -> str:
    return "\n".join(f"{i}\n{_ts(g.start)} --> {_ts(g.end)}\n{g.text}\n" for i, g in enumerate(groups, 1))


def to_vtt(groups: list[Group]) -> str:
    body = "\n".join(f"{_ts(g.start, '.')} --> {_ts(g.end, '.')}\n{g.text}\n" for g in groups)
    return "WEBVTT\n\n" + body


def _ass_ts(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


ASS_HEAD = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H0000D7FF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,0,2,60,60,{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def to_ass(groups: list[Group], width: int = 1080, height: int = 1920, font: str = "Poppins",
           size: int = 84, margin: int = 520) -> str:
    """Word-by-word karaoke captions: each word turns yellow the moment it is spoken."""
    lines = [ASS_HEAD.format(w=width, h=height, font=font, size=size, margin=margin)]
    for g in groups:
        parts = []
        for j, w in enumerate(g.words):
            nxt = g.words[j + 1].on if j + 1 < len(g.words) else g.end
            parts.append("{\\k%d}%s" % (max(1, round((nxt - w.on) * 100)), w.text))
        lines.append(f"Dialogue: 0,{_ass_ts(g.start)},{_ass_ts(g.end)},Default,,0,0,0,,{' '.join(parts)}")
    return "\n".join(lines) + "\n"


def to_json(words: list[CapWord], groups: list[Group]) -> str:
    data = {
        "words": [asdict(w) for w in words],
        "captions": [{"start": g.start, "end": round(g.end, 3), "text": g.text,
                      "words": [w.text for w in g.words]} for g in groups],
    }
    return json.dumps(data, ensure_ascii=False, indent=1)
