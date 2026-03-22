"""mazinger speak — synthesise dubbed audio from SRT (voice cloning + assembly)."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import (
    add_common, add_tempo, add_tts_engine, add_voice,
    require_voice, tempo_mode_from_args,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("speak", help="Synthesise dubbed audio from SRT.")
    p.add_argument("--srt", required=True, help="Path to translated SRT.")
    p.add_argument("--original-audio", required=True, help="Original audio (for duration matching).")
    add_voice(p)
    p.add_argument("-o", "--output", required=True, help="Output dubbed WAV path.")
    p.add_argument("--segments-dir", default="./tts_segments", help="Directory for segment WAVs.")
    add_tts_engine(p)
    p.add_argument("--device", default="cuda:0", help="Device.")
    p.add_argument("--dtype", default="bfloat16", help="Weight dtype (for Qwen engine).")
    add_tempo(p)
    p.add_argument("--force-reset", action="store_true",
                   help="Delete existing TTS segment files and re-synthesise from scratch.")
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    from mazinger import tts, assemble
    from mazinger.srt import parse_file
    from mazinger.utils import get_audio_duration

    srt_entries = parse_file(args.srt)
    original_duration = get_audio_duration(args.original_audio)
    voice_sample, voice_script = require_voice(args)

    with open(voice_script, encoding="utf-8") as fh:
        ref_text = fh.read().strip()

    engine = args.tts_engine
    model = tts.load_model(
        args.tts_model, device=args.device, dtype=args.dtype, engine=engine,
        chatterbox_model=args.chatterbox_model,
    )
    voice_prompt = tts.create_voice_prompt(
        model, voice_sample, ref_text,
        engine=engine,
        chatterbox_exaggeration=args.chatterbox_exaggeration,
        chatterbox_cfg=args.chatterbox_cfg,
    )
    segment_info = tts.synthesize_segments(
        model, voice_prompt, srt_entries, args.segments_dir,
        language=args.tts_language or "English",
        force_reset=args.force_reset,
    )
    tts.unload_model(voice_prompt)

    assemble.assemble_timeline(
        segment_info, original_duration, args.output,
        tempo_mode=tempo_mode_from_args(args),
        fixed_tempo=args.fixed_tempo,
        max_tempo=args.max_tempo,
    )
    print(f"Dubbed audio saved: {args.output}")
