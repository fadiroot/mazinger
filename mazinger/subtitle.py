"""Burn subtitles into a video using ffmpeg."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

log = logging.getLogger(__name__)

_POSITIONS = {"bottom": 2, "top": 8, "center": 5}

# Right-aligned equivalents for RTL text (ASS numpad layout).
_RTL_POSITIONS = {"bottom": 3, "top": 9, "center": 6}

# Unicode ranges for Arabic, Arabic Supplement, Arabic Extended,
# Arabic Presentation Forms A/B, and Farsi-specific characters.
_RTL_RE = re.compile(
    "[\u0600-\u06FF"     # Arabic
    "\u0750-\u077F"      # Arabic Supplement
    "\u08A0-\u08FF"      # Arabic Extended-A
    "\uFB50-\uFDFF"      # Arabic Presentation Forms-A
    "\uFE70-\uFEFF"      # Arabic Presentation Forms-B
    "\U00010D00-\U00010D3F"  # Arabic Extended-C (Hanifi Rohingya)
    "]"
)

# Unicode directional markers.
_RLE = "\u202B"   # Right-to-Left Embedding
_PDF = "\u202C"   # Pop Directional Formatting

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


def _starts_rtl(text: str) -> bool:
    """Return ``True`` if *text* begins with an Arabic / Farsi character.

    Ignores leading whitespace, digits, and punctuation.
    """
    for ch in text:
        if _RTL_RE.match(ch):
            return True
        # Skip whitespace, digits, common punctuation — keep scanning.
        if ch.isspace() or ch.isdigit() or not ch.isalpha():
            continue
        return False  # First alphabetic char is not RTL.
    return False


def _prepare_rtl_srt(srt_path: str, position: str) -> str | None:
    """Return the path to a temp SRT with per-entry RTL overrides, or ``None``.

    Scans every subtitle entry.  If *any* entry begins with an RTL character
    its text is wrapped with Unicode RLE marks and an ASS alignment override
    tag (``{\\an<N>}``) that right-aligns it.

    Returns ``None`` when no RTL entries are found (no temp file created).
    """
    with open(srt_path, encoding="utf-8") as fh:
        content = fh.read()

    rtl_alignment = _RTL_POSITIONS.get(position, 3)
    blocks = re.split(r"\n\n+", content.strip())
    changed = False
    new_blocks: list[str] = []

    for block in blocks:
        lines = block.split("\n")
        # SRT block: index, timestamp, text line(s)…
        if len(lines) >= 3:
            text_lines = lines[2:]
            first_text = "".join(text_lines).strip()
            if _starts_rtl(first_text):
                changed = True
                # Inject ASS override for right-alignment + wrap with RLE/PDF.
                tag = f"{{\\an{rtl_alignment}}}"
                lines[2] = tag + _RLE + lines[2]
                lines[-1] = lines[-1] + _PDF
        new_blocks.append("\n".join(lines))

    if not changed:
        return None

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", encoding="utf-8", delete=False,
    )
    tmp.write("\n\n".join(new_blocks) + "\n")
    tmp.close()
    log.info("Prepared RTL-aware SRT: %s", tmp.name)
    return tmp.name

@dataclass
class SubtitleStyle:
    """Visual styling for burned-in subtitles."""

    font: str = "Arial"
    font_size: int = 12
    font_color: str = "white"
    bg_color: str = "black"
    bg_alpha: float = 0.2
    outline_color: str = "black"
    outline_width: int = 1
    position: str = "bottom"
    margin_v: int = 20
    bold: bool = False

    def to_force_style(self) -> str:
        """Build an ASS ``force_style`` string for the ffmpeg ``subtitles`` filter."""
        fc = _to_ass_color(_parse_color(self.font_color))
        # BorderStyle=3 (opaque box): the box is drawn with OutlineColour,
        # while BackColour only affects the shadow.  Apply the background
        # color + alpha to OutlineColour so the box opacity works correctly.
        box_color = _to_ass_color(_parse_color(self.bg_color), self.bg_alpha)
        shadow_color = _to_ass_color(_parse_color(self.bg_color), 0.0)
        alignment = _POSITIONS.get(self.position, 2)
        return ",".join([
            f"FontName={self.font}",
            f"FontSize={self.font_size}",
            f"PrimaryColour={fc}",
            f"BackColour={shadow_color}",
            f"OutlineColour={box_color}",
            f"Outline={self.outline_width}",
            "BorderStyle=3",
            "Shadow=0",
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

    # Preprocess SRT for RTL alignment when needed.
    rtl_srt = _prepare_rtl_srt(srt_path, style.position)
    effective_srt = rtl_srt or srt_path

    escaped = _escape_filter_path(effective_srt)
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

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if rtl_srt:
            os.unlink(rtl_srt)

    log.info("Subtitled video saved: %s", output_path)
    return output_path
