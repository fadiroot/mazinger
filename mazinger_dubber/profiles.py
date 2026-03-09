"""Download voice-cloning profiles from the HuggingFace dataset repo."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import urllib.request
from urllib.error import HTTPError

log = logging.getLogger(__name__)

PROFILES_REPO_URL = (
    "https://huggingface.co/datasets/bakrianoo/mazinger-dubber-profiles/resolve/main/profiles"
)

SCRIPT_FILENAME = "script.txt"
VOICE_EXTENSIONS = ("wav", "m4a", "mp3")


def _download_file(url: str, dest: str) -> None:
    """Download *url* to *dest*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 – trusted HF URL


def _convert_to_wav(src: str, dest: str) -> None:
    """Convert any audio file to 16-kHz mono WAV using ffmpeg."""
    log.info("Converting %s -> %s", src, dest)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", src,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            dest,
        ],
        check=True,
        capture_output=True,
    )


def _download_voice(base_url: str, profile_dir: str) -> str:
    """Try each supported extension and return the path to the downloaded file."""
    for ext in VOICE_EXTENSIONS:
        dest = os.path.join(profile_dir, f"voice.{ext}")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            log.info("Using cached voice: %s", dest)
            return dest
        try:
            _download_file(f"{base_url}/voice.{ext}", dest)
            return dest
        except HTTPError:
            # Remove any empty/partial file left by the failed download
            if os.path.exists(dest):
                os.remove(dest)
            continue
    raise FileNotFoundError(
        f"No voice file found for profile at {base_url} "
        f"(tried extensions: {', '.join(VOICE_EXTENSIONS)})"
    )


def _ensure_wav(voice_path: str) -> str:
    """Return a WAV version of *voice_path*, converting if necessary."""
    if voice_path.endswith(".wav"):
        return voice_path
    wav_path = os.path.splitext(voice_path)[0] + ".wav"
    if os.path.exists(wav_path):
        return wav_path
    _convert_to_wav(voice_path, wav_path)
    return wav_path


def fetch_profile(profile_name: str, cache_dir: str | None = None) -> tuple[str, str]:
    """Return ``(voice_sample_path, voice_script_path)`` for *profile_name*.

    The voice file is downloaded (trying ``wav``, ``m4a``, ``mp3`` in order)
    and converted to 16-kHz mono WAV if needed so it is ready for TTS engines.

    Files are downloaded once into *cache_dir* (default: a persistent temp
    directory) and reused on subsequent calls.

    Parameters:
        profile_name: Profile name (e.g. ``abubakr``).
        cache_dir:    Directory to cache downloaded files.  When ``None`` a
                      directory under the system temp dir is used.

    Returns:
        A ``(voice_sample, voice_script)`` tuple of local file paths.
        The voice sample is always a WAV file.
    """
    if cache_dir is None:
        cache_dir = os.path.join(
            tempfile.gettempdir(), "mazinger-dubber-profiles"
        )

    profile_dir = os.path.join(cache_dir, profile_name)
    base = f"{PROFILES_REPO_URL}/{profile_name}"

    # Download script
    script_path = os.path.join(profile_dir, SCRIPT_FILENAME)
    if not os.path.exists(script_path):
        _download_file(f"{base}/{SCRIPT_FILENAME}", script_path)
    else:
        log.info("Using cached script: %s", script_path)

    # Download voice (try wav/m4a/mp3) and convert to WAV
    raw_voice = _download_voice(base, profile_dir)
    voice_path = _ensure_wav(raw_voice)

    return voice_path, script_path
