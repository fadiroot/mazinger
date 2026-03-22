"""Burn subtitles into a video using ffmpeg."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)

_POSITIONS = {"bottom": 2, "top": 8, "center": 5}

_NAMED_COLORS = {
    "white": "FFFFFF", "black": "000000", "yellow": "FFFF00",
    "red": "FF0000", "green": "00FF00", "blue": "0000FF",
    "cyan": "00FFFF", "magenta": "FF00FF", "gray": "808080",
}


def _parse_color(value: str) -> str:
    """Normalise a color name or ``#RRGGBB`` hex string to bare ``RRGGBB``."""
    low = value.strip().lower()
    if low in _NAMED_COLORS:
        return _NAMED_COLORS[low]
    stripped = value.strip().lstrip("#")
    if len(stripped) == 6 and all(c in "0123456789abcdefABCDEF" for c in stripped):
        return stripped.upper()
    raise ValueError(f"Unsupported color: {value!r} — use a name or #RRGGBB hex.")


def _to_ass_color(rgb: str, opacity: float = 1.0) -> str:
    """Convert ``RRGGBB`` + opacity (1=opaque, 0=transparent) to ASS ``&HAABBGGRR``."""
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    a = f"{int((1.0 - opacity) * 255):02X}"
    return f"&H{a}{b}{g}{r}"


def _escape_filter_path(path: str) -> str:
    """Escape a file path for an ffmpeg filter expression (within single quotes)."""
    return path.replace("\\", "\\\\").replace("'", "'\\''")


@dataclass
class SubtitleStyle:
    """Visual styling for burned-in subtitles."""

    font: str = "Arial"
    font_size: int = 24
    font_color: str = "white"
    bg_color: str = "black"
    bg_alpha: float = 0.5
    outline_color: str = "black"
    outline_width: int = 1
    position: str = "bottom"
    margin_v: int = 20
    bold: bool = False

    def to_force_style(self) -> str:
        """Build an ASS ``force_style`` string for the ffmpeg ``subtitles`` filter."""
        fc = _to_ass_color(_parse_color(self.font_color))
        bc = _to_ass_color(_parse_color(self.bg_color), self.bg_alpha)
        oc = _to_ass_color(_parse_color(self.outline_color), 1.0)
        alignment = _POSITIONS.get(self.position, 2)
        return ",".join([
            f"FontName={self.font}",
            f"FontSize={self.font_size}",
            f"PrimaryColour={fc}",
            f"BackColour={bc}",
            f"OutlineColour={oc}",
            f"Outline={self.outline_width}",
            "BorderStyle=3",
            f"Alignment={alignment}",
            f"MarginV={self.margin_v}",
            f"Bold={-1 if self.bold else 0}",
        ])


def burn_subtitles(
    video_path: str,
    output_path: str,
    srt_path: str,
    style: SubtitleStyle | None = None,
    audio_path: str | None = None,
) -> str | None:
    """Burn subtitles into *video_path*, optionally replacing the audio track.

    Uses the ffmpeg ``subtitles`` filter with ASS ``force_style`` overrides.
    When *audio_path* is given the original audio is replaced in the same
    encoding pass to avoid a double re-encode.

    Returns *output_path* on success, or ``None`` if ffmpeg is unavailable.
    """
    if shutil.which("ffmpeg") is None:
        log.warning(
            "ffmpeg not found — cannot burn subtitles.  "
            "Install ffmpeg and re-run."
        )
        return None

    if style is None:
        style = SubtitleStyle()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    escaped = _escape_filter_path(srt_path)
    vf = f"subtitles='{escaped}':force_style='{style.to_force_style()}'"

    cmd = ["ffmpeg", "-y", "-i", video_path]
    if audio_path:
        cmd += ["-i", audio_path]

    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "23"]

    if audio_path:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0"]

    cmd += ["-shortest", output_path]

    subprocess.run(cmd, check=True, capture_output=True)
    log.info("Subtitled video saved: %s", output_path)
    return output_path
