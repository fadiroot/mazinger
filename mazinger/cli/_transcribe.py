"""mazinger transcribe — audio to SRT."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import add_common, add_openai, add_source, resolve_project


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("transcribe", help="Transcribe audio to SRT.")
    add_source(p)
    p.add_argument("--audio", default=None, help="Path to audio file (overrides source).")
    p.add_argument("-o", "--output", default=None,
                   help="Output SRT path (default: project transcription/source.srt).")
    p.add_argument("--method", default="openai", choices=["openai", "faster-whisper", "whisperx"],
                   help="Transcription backend.")
    p.add_argument("--model", default=None,
                   help="Model name. Defaults to 'whisper-1' for OpenAI, 'large-v3' for local.")
    p.add_argument("--device", default="auto", help="Device: auto (default), cuda, or cpu.")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for local methods.")
    p.add_argument("--compute-type", default="float16", help="Compute type: float16, int8, int8_float16.")
    p.add_argument("--language", default=None, help="Force language code (e.g., en, ar).")
    p.add_argument("--max-chars", type=int, default=120, help="Max chars per subtitle.")
    p.add_argument("--max-duration", type=float, default=10.0, help="Max seconds per subtitle.")
    p.add_argument("--no-resegment", action="store_true", help="Skip resegmentation.")
    add_openai(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    import sys
    from mazinger.transcribe import transcribe
    from mazinger.cli._groups import resolve_device

    args.device = resolve_device(args.device)
    proj = resolve_project(args)

    audio = args.audio or (proj.audio if proj else None)
    output = args.output or (proj.source_srt if proj else None)

    if not audio:
        sys.exit("Error: provide a source (positional) or --audio.")
    if not output:
        sys.exit("Error: provide a source (positional) or -o/--output.")

    transcribe(
        audio, output,
        method=args.method,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
        language=args.language,
        max_chars=args.max_chars,
        max_duration=args.max_duration,
        skip_resegment=args.no_resegment,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
    )
    print(f"SRT saved: {output}")
