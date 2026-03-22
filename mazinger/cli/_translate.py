"""mazinger translate — translate SRT to target language."""

from __future__ import annotations

import argparse

from mazinger.cli._groups import add_common, add_llm, add_translation, make_openai_client


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("translate", help="Translate SRT to target language.")
    p.add_argument("--srt", required=True, help="Path to source SRT file.")
    p.add_argument("--description", required=True, help="Path to description JSON.")
    p.add_argument("--thumbnails-meta", default=None, help="Path to thumbnails meta.json.")
    p.add_argument("-o", "--output", required=True, help="Output SRT path.")
    add_llm(p)
    add_translation(p)
    add_common(p)


def handler(args: argparse.Namespace) -> None:
    from mazinger.translate import translate_srt
    from mazinger.utils import load_json

    client = make_openai_client(args)
    with open(args.srt, encoding="utf-8") as fh:
        srt_text = fh.read()
    description = load_json(args.description)
    thumb_paths = load_json(args.thumbnails_meta) if args.thumbnails_meta else []

    result = translate_srt(
        srt_text, description, thumb_paths, client, llm_model=args.llm_model,
        source_language=args.source_language,
        target_language=args.target_language,
        **(dict(words_per_second=args.words_per_second) if args.words_per_second is not None else {}),
        **(dict(duration_budget=args.duration_budget) if args.duration_budget is not None else {}),
    )
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(result)
    print(f"Translated SRT saved: {args.output}")
