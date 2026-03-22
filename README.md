# Mazinger Dubber

End-to-end video dubbing pipeline.
Download a video, transcribe it, translate the subtitles, and generate a voice-cloned dubbed audio track — all in one command.

---

## Features

| Stage          | What it does                                          |
|----------------|-------------------------------------------------------|
| **Download**   | Fetch video + extract audio (yt-dlp)                  |
| **Transcribe** | Speech-to-text (OpenAI API / faster-whisper / WhisperX) |
| **Thumbnails** | LLM-selected key frames for visual context            |
| **Describe**   | Structured content analysis (title, summary, keywords)|
| **Translate**  | Context-aware SRT translation with visual grounding   |
| **Re-segment** | Split long subtitles into readable caption blocks     |
| **TTS**        | Voice-cloned speech (Qwen3-TTS or Chatterbox)         |
| **Assemble**   | Time-aligned final audio matching original duration   |
| **Subtitles**  | Embed styled subtitles into the video (ffmpeg)        |
| **LLM Usage**  | Per-stage token tracking with summary report          |

Every stage works **independently** or chained through the `MazingerDubber` class / `mazinger` CLI.

---

## Prerequisites

- **Python 3.10+**
- **ffmpeg** — `apt install ffmpeg` or `brew install ffmpeg`
- **OpenAI API key** — set `OPENAI_API_KEY` env var (or pass via `--openai-api-key`)
- **CUDA GPU** — only needed for local transcription (`faster-whisper`, `whisperx`) and TTS (`speak`/`dub`)

---

## Installation

The base install is lightweight — it includes **download**, **thumbnails**, **describe**, **translate**, and **resegment** (all OpenAI-based). Local transcription and TTS engines are optional extras.

```bash
# Core only (OpenAI transcription + LLM tasks, no GPU needed)
pip install .
```

### Add local transcription

```bash
pip install ".[transcribe-faster]"    # faster-whisper — fast, Chatterbox-compatible
pip install ".[transcribe-whisperx]"  # WhisperX — best word-level alignment (Qwen-compatible)
```

### Add TTS

```bash
pip install ".[tts]"                  # Qwen3-TTS (needs voice sample + transcript)
pip install ".[tts-chatterbox]"       # Chatterbox (voice sample only, emotion control)
```

### Full bundles

```bash
pip install ".[all-qwen]"            # WhisperX + Qwen TTS
pip install ".[all-chatterbox]"      # faster-whisper + Chatterbox TTS
```

> **Qwen and Chatterbox cannot coexist** in the same environment (conflicting `transformers` versions).
> **WhisperX conflicts with Chatterbox** — use `faster-whisper` or OpenAI transcription with Chatterbox.
> See [DOCS.md](DOCS.md#installation-options) for advanced install methods (venv, Colab, uv overrides).

### What each task requires

| Task            | Command               | Core install | Extra needed                                   |
|-----------------|------------------------|:------------:|-------------------------------------------------|
| Download        | `mazinger download`    | ✅           | —                                               |
| Transcribe (cloud) | `mazinger transcribe` | ✅        | — (uses OpenAI API)                             |
| Transcribe (local) | `mazinger transcribe --method faster-whisper` | — | `transcribe-faster` (+ CUDA GPU) |
| Transcribe (local) | `mazinger transcribe --method whisperx` | — | `transcribe-whisperx` (+ CUDA GPU) |
| Thumbnails      | `mazinger thumbnails`  | ✅           | —                                               |
| Describe        | `mazinger describe`    | ✅           | —                                               |
| Translate       | `mazinger translate`   | ✅           | —                                               |
| Re-segment      | `mazinger resegment`   | ✅           | —                                               |
| Speak (Qwen)    | `mazinger speak`       | —            | `tts` (+ CUDA GPU)                             |
| Speak (Chatterbox) | `mazinger speak --tts-engine chatterbox` | — | `tts-chatterbox` (+ CUDA GPU) |
| Subtitle embed  | `mazinger subtitle`    | ✅           | — (ffmpeg only)                                 |
| Full dub (Qwen) | `mazinger dub`         | —            | `all-qwen` (+ CUDA GPU)                        |
| Full dub (Chatterbox) | `mazinger dub --tts-engine chatterbox` | — | `all-chatterbox` (+ CUDA GPU) |

---

## Quick Start

### One command — dub a video

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --voice-sample reference.m4a \
    --voice-script reference_transcript.txt \
    --base-dir ./output
```

Add `--tts-engine chatterbox` to use Chatterbox instead of Qwen.

### Resuming an interrupted run

By default, every stage caches its outputs. If a run is interrupted (e.g. during TTS), simply re-run the same command — already-completed stages and TTS segments are skipped automatically.

To **discard all cached outputs** and start from scratch, add `--force-reset`:

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --force-reset
```

`--force-reset` also works with the standalone `speak` sub-command.

### Using a voice profile

Instead of providing `--voice-sample` and `--voice-script` manually, use a named profile from the [voice profiles dataset](https://huggingface.co/datasets/bakrianoo/mazinger-dubber-profiles):

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --base-dir ./output
```

To produce a dubbed **video** (replaces audio track in the source video):

```bash
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr \
    --output-type video
```

The voice sample and script are downloaded automatically (no auth required). You can also use `--clone-profile` with local files:

```bash
# Local video with a profile
mazinger dub ./my_video.mp4 --clone-profile abubakr

# Local audio with a profile
mazinger dub ./my_audio.mp3 --clone-profile abubakr
```

### Python

```python
from mazinger import MazingerDubber

dubber = MazingerDubber(openai_api_key="sk-...", base_dir="./output")

# With explicit voice files
proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_sample="reference.m4a",
    voice_script="reference_transcript.txt",
)

# Or resolve a profile first
from mazinger.profiles import fetch_profile
voice, script = fetch_profile("abubakr")
proj = dubber.dub(
    source="https://youtube.com/watch?v=VIDEO_ID",
    voice_sample=voice,
    voice_script=script,
)

print(proj.final_audio)  # ./output/projects/<slug>/tts/dubbed.wav
```

### Embed subtitles into the video

```bash
# During dubbing — adds translated subtitles to the output video
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr --embed-subtitles

# Standalone — burn subtitles into any video + optionally replace audio
mazinger subtitle video.mp4 --srt translated.srt -o output.mp4
mazinger subtitle video.mp4 --srt translated.srt --audio dubbed.wav -o output.mp4
```

All `--subtitle-*` styling flags (font, size, color, position, etc.) work with both commands. See [DOCS.md](DOCS.md#subtitle-embedding) for the full reference.

### Run individual stages

Each pipeline stage has its own sub-command:

```bash
mazinger download   ...
mazinger transcribe ...
mazinger thumbnails ...
mazinger describe   ...
mazinger translate  ...
mazinger resegment  ...
mazinger speak      ...
mazinger subtitle   ...
```

Run any command with `--help` for all options. Full step-by-step guide in [DOCS.md](DOCS.md#step-by-step-usage).

---

## Voice Profiles

Voice profiles let you reuse a speaker's voice sample and transcript without passing file paths every time. Profiles are stored on the [mazinger-dubber-profiles](https://huggingface.co/datasets/bakrianoo/mazinger-dubber-profiles) HuggingFace dataset.

| Profile   | Language | Description      |
|-----------|----------|------------------|
| `abubakr` | English  | Abu Bakr Soliman |

See the [profiles README](https://huggingface.co/datasets/bakrianoo/mazinger-dubber-profiles) for how to upload your own profile.

---

## FAQ

**Which TTS engine should I use?**
Use **Chatterbox** if you only have a voice sample (no transcript needed) or want emotion/pacing control.
Use **Qwen** for multilingual support with a reference transcript.

**Do I need a GPU?**
Only for local transcription (`faster-whisper`, `whisperx`) and TTS. If you use `--transcribe-method openai` and an external TTS, a CPU-only machine works fine.

**Can I use Qwen and Chatterbox together?**
No. They require different `transformers` versions. Use separate virtual environments if you need both.

**Which transcription method works with Chatterbox?**
`openai` (cloud) and `faster-whisper` (local). WhisperX has a dependency conflict with Chatterbox.

**Where do the output files go?**
Under `<base-dir>/projects/<slug>/` (default: `./mazinger_output/projects/<slug>/`) — organised into `source/`, `transcription/`, `subtitles/`, `thumbnails/`, `analysis/`, and `tts/` folders.

**My run was interrupted — do I have to start over?**
No. Re-run the same command and all completed stages (including individual TTS segments) are skipped automatically. Use `--force-reset` only if you want to regenerate everything from scratch.

**How do I pass YouTube cookies for age-restricted videos?**
Use `--cookies-from-browser chrome` or `--cookies path/to/cookies.txt` with any command.

**flash-attn won't install — is that a problem?**
No. It's an optional acceleration for TTS. Chatterbox and Qwen both fall back to standard attention automatically.

---

## Documentation

Full installation guides, step-by-step API reference, project structure details, and configuration options:

**[DOCS.md](DOCS.md)**

---

## License

[MIT](LICENSE)
