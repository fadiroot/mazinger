"""mazinger dub — full dubbing pipeline."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import (
    add_common, add_cookies, add_llm, add_tempo, add_tts_engine,
    add_transcription, add_translation, add_voice,
    require_voice, tempo_mode_from_args,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("dub", help="Run the full dubbing pipeline.")
    p.add_argument("source", help="Video URL, local video path, or local audio path.")
    add_voice(p)
    p.add_argument("--slug", default=None, help="Override project slug.")
    add_transcription(p)
    add_tts_engine(p)
    add_llm(p)
    add_cookies(p)
    p.add_argument("--use-resegmented", action="store_true",
                   help="Translate from the resegmented SRT instead of the raw transcript.")
    p.add_argument("--output-type", choices=["audio", "video"], default="audio",
                   help="Output type: 'audio' (default) or 'video'.")
    add_tempo(p)
    add_translation(p)
    p.add_argument("--force-reset", action="store_true",
                   help="Discard all cached outputs and re-run every stage.")
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    from mazinger.pipeline import MazingerDubber

    voice_sample, voice_script = require_voice(args)

    dubber = MazingerDubber(
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        llm_model=args.llm_model,
        base_dir=args.base_dir,
    )
    proj = dubber.dub(
        source=args.source,
        voice_sample=voice_sample,
        voice_script=voice_script,
        slug=args.slug,
        device=args.device,
        transcribe_method=args.transcribe_method,
        whisper_model=args.whisper_model,
        tts_model_name=args.tts_model,
        tts_language=args.tts_language,
        tts_engine=args.tts_engine,
        source_language=args.source_language,
        target_language=args.target_language,
        chatterbox_model=args.chatterbox_model,
        chatterbox_exaggeration=args.chatterbox_exaggeration,
        chatterbox_cfg=args.chatterbox_cfg,
        cookies_from_browser=args.cookies_from_browser,
        cookies=args.cookies,
        force_reset=args.force_reset,
        use_resegmented=args.use_resegmented,
        output_type=args.output_type,
        tempo_mode=tempo_mode_from_args(args),
        fixed_tempo=args.fixed_tempo,
        max_tempo=args.max_tempo,
        words_per_second=args.words_per_second,
        duration_budget=args.duration_budget,
    )
    print(proj.summary())
