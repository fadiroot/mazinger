"""Reusable argument groups shared across CLI commands."""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_BASE_DIR = "./mazinger_output"


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--base-dir", default=DEFAULT_BASE_DIR,
        help=f"Root directory for project folders (default: {DEFAULT_BASE_DIR}).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug-level logging.")


def add_openai(p: argparse.ArgumentParser) -> None:
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key.")
    p.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL"), help="Base URL for OpenAI-compatible API.")


def add_llm(p: argparse.ArgumentParser) -> None:
    add_openai(p)
    p.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1"), help="LLM model for translation/analysis.")


def add_voice(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--clone-profile", default=None,
        help="Name of a voice profile from the HuggingFace dataset (e.g. 'abubakr').",
    )
    p.add_argument("--voice-sample", default=None, help="Path to voice reference audio.")
    p.add_argument("--voice-script", default=None, help="Path to voice reference transcript.")


def add_tts_engine(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Qwen TTS model name.")
    p.add_argument("--chatterbox-model", default="ResembleAI/chatterbox", help="Chatterbox TTS model name.")
    p.add_argument("--tts-language", default="English", help="Target TTS language.")
    p.add_argument(
        "--tts-engine", default="qwen", choices=["qwen", "chatterbox"],
        help="TTS engine: 'qwen' (Qwen3-TTS) or 'chatterbox' (ResembleAI Chatterbox).",
    )
    p.add_argument("--chatterbox-exaggeration", type=float, default=0.5,
                   help="Chatterbox exaggeration level (0.0-1.0, default 0.5).")
    p.add_argument("--chatterbox-cfg", type=float, default=0.5,
                   help="Chatterbox CFG weight (0.0-1.0, default 0.5).")


def add_tempo(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dynamic-tempo", action="store_true",
                   help="Enable per-segment dynamic tempo adjustment.")
    p.add_argument("--fixed-tempo", type=float, default=None,
                   help="Apply a fixed tempo rate to all segments (e.g. 1.1). Overrides --dynamic-tempo.")
    p.add_argument("--max-tempo", type=float, default=1.3,
                   help="Maximum speed-up factor for dynamic tempo (default: 1.3).")


def add_transcription(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--transcribe-method", default="openai", choices=["openai", "faster-whisper", "whisperx"],
        help="Transcription backend: 'openai', 'faster-whisper', or 'whisperx'.",
    )
    p.add_argument("--whisper-model", default=None, help="Whisper model name.")
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu.")


def add_cookies(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cookies-from-browser", default=None,
                   help="Pass through to yt-dlp --cookies-from-browser.")
    p.add_argument("--cookies", default=None,
                   help="Pass through to yt-dlp --cookies (path to Netscape cookie file).")


def add_translation(p: argparse.ArgumentParser) -> None:
    p.add_argument("--words-per-second", type=float, default=None,
                   help="Target speech rate in words/sec for duration matching (default: 2.0).")
    p.add_argument("--duration-budget", type=float, default=None,
                   help="Fraction of time window to fill with translated speech, 0.0-1.0 (default: 0.80).")


def resolve_voice(args: argparse.Namespace) -> tuple[str | None, str | None]:
    voice_sample = args.voice_sample
    voice_script = args.voice_script
    if args.clone_profile:
        from mazinger.profiles import fetch_profile
        pv, ps = fetch_profile(args.clone_profile)
        voice_sample = voice_sample or pv
        voice_script = voice_script or ps
    return voice_sample, voice_script


def require_voice(args: argparse.Namespace) -> tuple[str, str]:
    sample, script = resolve_voice(args)
    if not sample or not script:
        sys.exit("Error: --voice-sample and --voice-script are required (or use --clone-profile).")
    return sample, script


def make_openai_client(args: argparse.Namespace):
    from openai import OpenAI
    kw = {}
    if args.openai_api_key:
        kw["api_key"] = args.openai_api_key
    if args.openai_base_url:
        kw["base_url"] = args.openai_base_url
    return OpenAI(**kw)


def tempo_mode_from_args(args: argparse.Namespace) -> str:
    if args.fixed_tempo:
        return "fixed"
    if args.dynamic_tempo:
        return "dynamic"
    return "auto"
