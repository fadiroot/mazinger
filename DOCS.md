# Mazinger Dubber — Full Documentation

Detailed reference for installation options, pipeline stages, CLI commands, Python API, and project internals.

---

## Table of Contents

- [Installation Options](#installation-options)
  - [Chatterbox TTS (all-chatterbox)](#all-chatterbox)
  - [Qwen TTS (all-qwen)](#all-qwen)
  - [Flash Attention (optional)](#flash-attention-optional)
  - [System Dependencies](#system-dependencies)
- [Transcription Methods](#transcription-methods)
- [TTS Engines](#tts-engines)
- [Voice Profiles](#voice-profiles)
- [Step-by-Step Usage](#step-by-step-usage)
  - [1. Download](#1-download)
  - [2. Transcribe](#2-transcribe)
  - [3. Extract Thumbnails](#3-extract-thumbnails)
  - [4. Describe Content](#4-describe-content)
  - [5. Translate](#5-translate)
  - [6. Re-segment](#6-re-segment)
  - [7. Synthesise & Assemble](#7-synthesise--assemble)
- [Project Output Structure](#project-output-structure)
- [Subtitle Embedding](#subtitle-embedding)
- [Configuration](#configuration)
- [Package Layout](#package-layout)

---

## Installation Options

### Core only

The base install is lightweight — download, OpenAI transcription, thumbnails, describe, translate, and resegment. No GPU required.

```bash
pip install .
```

Core dependencies: `yt-dlp`, `openai`, `json-repair`, `Pillow`, `soundfile`, `numpy`, `tqdm`.

### Selective extras

```bash
pip install ".[transcribe-faster]"    # faster-whisper (fast local, Chatterbox compatible)
pip install ".[transcribe-whisperx]"  # WhisperX (PyTorch + CUDA required)
pip install ".[tts]"                  # Qwen TTS
pip install ".[tts-chatterbox]"       # Chatterbox TTS
```

### Full bundles

```bash
pip install ".[all-qwen]"        # WhisperX + Qwen TTS
pip install ".[all-chatterbox]"  # faster-whisper + Chatterbox TTS
```

> **Qwen TTS and Chatterbox TTS cannot coexist** in the same environment due to conflicting `transformers` versions. Pick one per venv.
>
> **WhisperX conflicts with Chatterbox** (`transformers>=4.48` vs `==4.46.3`). Use `faster-whisper` or OpenAI transcription with Chatterbox.

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

### `all-chatterbox`

#### Method 1 — Fresh venv (Python 3.12)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate

# numpy must exist before the pkuseg C extension is built
uv pip install "numpy>=1.26"

# CUDA-enabled PyTorch (adjust cu128 to match your driver)
uv pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

uv pip install --no-build-isolation ".[all-chatterbox]"
```

#### Method 2 — Google Colab

```bash
cat > /tmp/cb_overrides.txt << 'EOF'
torch>=2.0
torchaudio>=2.0
numpy>=1.26
pandas>=2.2
gradio>=5.0
safetensors>=0.3
EOF

uv pip install --system --no-build-isolation \
    --override /tmp/cb_overrides.txt \
    ".[all-chatterbox]"
```

---

### `all-qwen`

#### Method 1 — Fresh venv (Python 3.12)

```bash
uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

cat > /tmp/qwen_overrides.txt << 'EOF'
torch>=2.0
torchaudio>=2.0
EOF

uv pip install --override /tmp/qwen_overrides.txt ".[all-qwen]"
```

#### Method 2 — Google Colab

```bash
cat > /tmp/qwen_overrides.txt << 'EOF'
torch>=2.0
torchaudio>=2.0
EOF

uv pip install --system \
    --override /tmp/qwen_overrides.txt \
    ".[all-qwen]"
```

---

### Flash Attention (optional)

Speeds up TTS inference on supported GPUs. Not required — Chatterbox falls back to standard attention.

```bash
uv pip install --no-build-isolation flash-attn   # or: pip install ".[flash-attn]"
```

---

### System Dependencies

| Tool      | Used by                        | Install                                      |
|-----------|--------------------------------|----------------------------------------------|
| `ffmpeg`  | download, thumbnails, assemble | `apt install ffmpeg` / `brew install ffmpeg`  |
| `ffprobe` | assemble (duration check)      | Bundled with ffmpeg                           |
| CUDA GPU  | transcribe (local), tts        | NVIDIA driver + CUDA toolkit                  |

---

## Transcription Methods

| Feature                  | OpenAI Whisper API        | faster-whisper            | WhisperX                          |
|--------------------------|---------------------------|---------------------------|-----------------------------------|
| Setup complexity         | Simple (API key)          | Medium (CUDA)             | Complex (CUDA + PyTorch)          |
| GPU required             | No                        | Yes (or CPU)              | Yes                               |
| Cost                     | Pay per audio minute      | Free                      | Free                              |
| Speed                    | Fast (cloud)              | 4× faster than Whisper    | Depends on GPU                    |
| Word-level timestamps    | Segment-level             | Yes                       | Yes (via wav2vec2)                |
| Chatterbox compatible    | ✅                        | ✅                        | ❌                                |
| Default model            | `whisper-1`               | `large-v3`                | `large-v3`                        |

**Which should I pick?**

- Using **Chatterbox TTS** → `openai` or `faster-whisper`
- Want **offline/local** processing → `faster-whisper`
- Need **best word-level alignment** → `whisperx` with Qwen TTS

---

## TTS Engines

| Feature                    | Qwen3-TTS                           | Chatterbox                       |
|----------------------------|--------------------------------------|----------------------------------|
| Voice cloning requires     | Audio + transcript                   | Audio only                       |
| Emotion control            | ❌                                   | ✅ (exaggeration param)          |
| Pacing control             | ❌                                   | ✅ (cfg param)                   |
| Multilingual               | Yes                                  | Yes (23 languages)               |
| Default model              | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`     | `ResembleAI/chatterbox`          |

### Chatterbox Tips

| Scenario          | exaggeration | cfg  |
|-------------------|:------------:|:----:|
| General use       | 0.5          | 0.5  |
| Fast speakers     | 0.5          | 0.3  |
| Expressive speech | 0.7          | 0.3  |

---

## Voice Profiles

Voice profiles provide a convenient way to clone a speaker's voice without specifying `--voice-sample` and `--voice-script` manually. Profiles are hosted on a public HuggingFace dataset and downloaded automatically.

**Dataset:** <https://huggingface.co/datasets/bakrianoo/mazinger-dubber-profiles>

### Using a profile

```bash
# Full pipeline with a profile
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" --clone-profile abubakr

# Speak sub-command with a profile
mazinger speak --srt translated.srt --original-audio audio.mp3 \
    --clone-profile abubakr -o dubbed.wav

# Profile + override (use profile voice but custom script)
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr --voice-script my_custom_script.txt
```

### Python API

```python
from mazinger.profiles import fetch_profile

# Downloads voice sample + script, converts audio to 16kHz mono WAV
voice_path, script_path = fetch_profile("abubakr")
# voice_path  -> /tmp/mazinger-dubber-profiles/abubakr/voice.wav
# script_path -> /tmp/mazinger-dubber-profiles/abubakr/script.txt
```

Files are cached in `/tmp/mazinger-dubber-profiles/` and reused on subsequent calls.

### Profile structure

Each profile is a folder with two files:

| File         | Description                                                      |
|--------------|------------------------------------------------------------------|
| `script.txt` | Plain-text transcript matching the voice sample                  |
| `voice.*`    | Voice sample audio (`.wav`, `.m4a`, or `.mp3` — auto-detected)   |

Non-WAV voice files are automatically converted to 16-kHz mono WAV for TTS compatibility.

### Uploading a new profile

```bash
# 1. Log in to HuggingFace
hf auth login

# 2. Prepare your files
mkdir -p profiles/my-name
# Add voice.m4a (or .wav/.mp3) and script.txt

# 3. Upload
hf upload bakrianoo/mazinger-dubber-profiles \
    ./profiles/my-name my-name \
    --repo-type=dataset \
    --commit-message "Add profile: my-name"
```

### Available profiles

| Profile    | Language | Description      |
|------------|----------|------------------|
| `abubakr`  | English  | Abu Bakr Soliman |

---

## Step-by-Step Usage

Each stage has a CLI sub-command **and** a matching Python function.

### 1. Download

```bash
mazinger download "https://youtube.com/watch?v=VIDEO_ID" --base-dir ./output
```

```python
from mazinger.download import resolve_slug, download_video, extract_audio

slug, info = resolve_slug("https://youtube.com/watch?v=VIDEO_ID")
download_video("https://youtube.com/watch?v=VIDEO_ID", "video.mp4")
extract_audio("video.mp4", "audio.mp3")
```

### 2. Transcribe

```bash
# OpenAI Whisper API (default)
mazinger transcribe audio.mp3 -o subtitles.srt

# faster-whisper (local, Chatterbox compatible)
mazinger transcribe audio.mp3 -o subtitles.srt --method faster-whisper --device cuda

# WhisperX (local, best alignment)
mazinger transcribe audio.mp3 -o subtitles.srt --method whisperx --device cuda
```

```python
from mazinger.transcribe import transcribe

transcribe("audio.mp3", "subtitles.srt")                                          # OpenAI
transcribe("audio.mp3", "subtitles.srt", method="faster-whisper", device="cuda")  # faster-whisper
transcribe("audio.mp3", "subtitles.srt", method="whisperx", device="cuda")        # WhisperX
```

### 3. Extract Thumbnails

```bash
mazinger thumbnails \
    --video video.mp4 \
    --srt subtitles.srt \
    --output-dir ./thumbs
```

```python
from openai import OpenAI
from mazinger.thumbnails import select_timestamps, extract_frames

client = OpenAI()
with open("subtitles.srt") as f:
    srt_text = f.read()

timestamps = select_timestamps(srt_text, client)
frames = extract_frames("video.mp4", timestamps, "./thumbs")
```

### 4. Describe Content

```bash
mazinger describe \
    --srt subtitles.srt \
    --thumbnails-meta ./thumbs/meta.json \
    -o description.json
```

```python
from mazinger.describe import describe_content
from mazinger.utils import load_json

thumb_paths = load_json("./thumbs/meta.json")
desc = describe_content(srt_text, thumb_paths, client)
```

### 5. Translate

```bash
mazinger translate \
    --srt subtitles.srt \
    --description description.json \
    --thumbnails-meta ./thumbs/meta.json \
    -o translated.srt
```

```python
from mazinger.translate import translate_srt

translated = translate_srt(srt_text, desc, thumb_paths, client)
with open("translated.srt", "w") as f:
    f.write(translated)
```

### 6. Re-segment

```bash
mazinger resegment --srt translated.srt -o final.srt
```

```python
from mazinger.resegment import resegment_srt

final_srt = resegment_srt(translated, client=client)
```

### 7. Synthesise & Assemble

```bash
# Qwen (default, no tempo adjustment)
mazinger speak \
    --srt translated.srt \
    --original-audio audio.mp3 \
    --voice-sample reference.m4a \
    --voice-script reference_transcript.txt \
    -o dubbed.wav

# Using a voice profile instead of explicit files
mazinger speak \
    --srt translated.srt \
    --original-audio audio.mp3 \
    --clone-profile abubakr \
    -o dubbed.wav

# Chatterbox
mazinger speak \
    --srt translated.srt \
    --original-audio audio.mp3 \
    --voice-sample reference.m4a \
    --voice-script reference_transcript.txt \
    --tts-engine chatterbox \
    -o dubbed.wav

# With fixed tempo (speed up all segments by 1.1×)
mazinger speak \
    --srt translated.srt \
    --original-audio audio.mp3 \
    --voice-sample reference.m4a \
    --voice-script reference_transcript.txt \
    --fixed-tempo 1.1 \
    -o dubbed.wav

# With dynamic tempo (per-segment, max 1.3×)
mazinger speak \
    --srt translated.srt \
    --original-audio audio.mp3 \
    --voice-sample reference.m4a \
    --voice-script reference_transcript.txt \
    --dynamic-tempo --max-tempo 1.3 \
    -o dubbed.wav
```

```python
from mazinger import tts, assemble
from mazinger.srt import parse_file
from mazinger.utils import get_audio_duration

model = tts.load_model(engine="qwen")
prompt = tts.create_voice_prompt(model, "reference.m4a", ref_text, engine="qwen")
segments = tts.synthesize_segments(model, prompt, parse_file("translated.srt"), "./segments")

# To discard cached segments and re-generate from scratch:
# segments = tts.synthesize_segments(model, prompt, parse_file("translated.srt"), "./segments",
#                                    force_reset=True)
tts.unload_model(prompt)

# No tempo adjustment (default)
assemble.assemble_timeline(segments, get_audio_duration("audio.mp3"), "dubbed.wav")

# Fixed tempo
assemble.assemble_timeline(segments, get_audio_duration("audio.mp3"), "dubbed.wav",
                           tempo_mode="fixed", fixed_tempo=1.1)

# Dynamic tempo
assemble.assemble_timeline(segments, get_audio_duration("audio.mp3"), "dubbed.wav",
                           tempo_mode="dynamic", max_tempo=1.3)
```

---

## Subtitle Embedding

Embed styled subtitles directly into a video using the ffmpeg `subtitles` filter. Available as a standalone command or integrated into `mazinger dub`.

### Standalone command

```bash
# Burn translated subtitles into a video (keeps original audio)
mazinger subtitle video.mp4 --srt translated.srt -o output.mp4

# Replace audio and add subtitles in one pass
mazinger subtitle video.mp4 --srt translated.srt --audio dubbed.wav -o output.mp4

# Custom styling
mazinger subtitle video.mp4 --srt translated.srt -o output.mp4 \
    --subtitle-font-size 32 \
    --subtitle-font-color yellow \
    --subtitle-bg-color black --subtitle-bg-alpha 0.6 \
    --subtitle-position top \
    --subtitle-bold

# Use a local font file
mazinger subtitle video.mp4 --srt translated.srt -o output.mp4 \
    --subtitle-font-file /path/to/MyFont-Regular.ttf

# Download and use a Google Font
mazinger subtitle video.mp4 --srt translated.srt -o output.mp4 \
    --subtitle-google-font "Noto Sans Arabic" --subtitle-font-size 24
```

### Integrated with `dub`

```bash
# Dub and embed translated subtitles
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr --embed-subtitles

# Dub with original-language subtitles
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr --embed-subtitles --subtitle-source original

# Use a custom SRT file
mazinger dub "https://youtube.com/watch?v=VIDEO_ID" \
    --clone-profile abubakr --embed-subtitles --subtitle-source /path/to/custom.srt
```

`--embed-subtitles` implies video output — no need to add `--output-type video`.

### Integrated with `translate`

```bash
# Translate and embed subtitles into the source video in one step
mazinger translate --srt source.srt --description description.json \
    -o translated.srt --embed-subtitles --video video.mp4

# Custom video output path
mazinger translate --srt source.srt --description description.json \
    -o translated.srt --embed-subtitles --video video.mp4 --video-output subtitled.mp4
```

When `--video-output` is omitted, the video is saved alongside the SRT (e.g. `translated.mp4`).

### Python API

```python
from mazinger.subtitle import SubtitleStyle, burn_subtitles

style = SubtitleStyle(
    font="DejaVu Sans",
    font_size=28,
    font_color="yellow",
    position="bottom",
    bold=True,
)

# Subtitles only (keeps original audio)
burn_subtitles("video.mp4", "output.mp4", "translated.srt", style)

# Subtitles + audio replacement in a single encoding pass
burn_subtitles("video.mp4", "output.mp4", "translated.srt", style, audio_path="dubbed.wav")

# Use a local font file (font family is auto-detected)
style = SubtitleStyle(font_file="/path/to/NotoSansArabic-Regular.ttf", font_size=24)
burn_subtitles("video.mp4", "output.mp4", "translated.srt", style)

# Download and use a Google Font
from mazinger.subtitle import download_google_font
font_path = download_google_font("Noto Sans Arabic")
style = SubtitleStyle(font_file=font_path, font_size=24)
burn_subtitles("video.mp4", "output.mp4", "translated.srt", style)
```

### Styling options

| Parameter                  | Default   | Description                                |
|----------------------------|-----------|--------------------------------------------|
| `--subtitle-font`          | Arial     | Font family name                           || `--subtitle-font-file`     | —         | Path to a local TTF/OTF font file          |
| `--subtitle-google-font`   | —         | Google Font name (auto-downloaded)         || `--subtitle-font-size`     | 14        | Font size in pixels                        |
| `--subtitle-font-color`    | white     | Text color (name or `#RRGGBB`)             |
| `--subtitle-bg-color`      | black     | Background box color                       |
| `--subtitle-bg-alpha`      | 0.6       | Background opacity (0.0–1.0)               |
| `--subtitle-outline-color` | black     | Outline color                              |
| `--subtitle-outline-width` | 1         | Outline thickness                          |
| `--subtitle-position`      | bottom    | `top`, `center`, or `bottom`               |
| `--subtitle-margin`        | 20        | Vertical margin from edge (pixels)         |
| `--subtitle-bold`          | off       | Bold text                                  |
| `--subtitle-line-spacing`  | 8         | Extra vertical spacing between lines (px)  |

---

## Project Output Structure

All artefacts are organised under `<base_dir>/projects/<slug>/`:

```
projects/<slug>/
├── source/
│   ├── video.mp4
│   └── audio.mp3
├── transcription/
│   ├── source.srt
│   ├── source.raw.srt
│   └── translated.raw.srt
├── subtitles/
│   └── translated.srt
├── thumbnails/
│   ├── thumb_000_12.5s.jpg
│   └── meta.json
├── analysis/
│   └── description.json
├── tts/
│   ├── segments/
│   │   ├── seg_0001.wav
│   │   └── ...
│   └── dubbed.wav
└── llm_usage.json
```

---

## Configuration

| Environment variable | Purpose                                          | CLI flag            |
|----------------------|--------------------------------------------------|---------------------|
| `OPENAI_API_KEY`     | API key (transcription + LLM tasks)              | `--openai-api-key`  |
| `OPENAI_BASE_URL`    | Base URL for OpenAI-compatible API providers     | `--openai-base-url` |
| `OPENAI_MODEL`       | Default LLM model for translation/analysis       | `--llm-model`       |

CLI flags take precedence over environment variables. If neither is set, `OPENAI_MODEL` defaults to `gpt-4.1`.

### Caching & Force Reset

By default, every pipeline stage and individual TTS segment is cached. If a run is interrupted, re-running the same command resumes from where it stopped.

| Flag | Effect |
|------|--------|
| *(default)* | Skip stages/segments whose output files already exist |
| `--force-reset` | Discard all cached outputs and re-run every stage from scratch |

`--force-reset` works with both the `dub` and `speak` sub-commands.

### LLM Usage Tracking

Every LLM call (thumbnails, describe, translate, resegment) is automatically tracked. After the pipeline completes, a usage report is logged showing per-stage token counts:

```
═══ LLM Usage Report ═══
  thumbnails        model=gpt-4.1                     calls=1  in=   3,420  out=    812
  describe          model=gpt-4.1                     calls=1  in=  12,105  out=    534
  translate         model=gpt-4.1                     calls=4  in=  45,230  out=  6,102
  resegment-merge   model=gpt-4.1                     calls=2  in=   8,400  out=  1,230
  ────────────────────────────────────────────────────────────────────────────
  TOTAL                                               calls=8  in=  69,155  out=  8,678
══════════════════════════
```

The raw records are also saved to `<project>/llm_usage.json` for programmatic access.

To use the tracker in Python outside the pipeline:

```python
from mazinger.utils import LLMUsageTracker

tracker = LLMUsageTracker()
# Pass tracker to any LLM-calling function:
translated = translate_srt(srt_text, desc, thumbs, client, usage_tracker=tracker)
print(tracker.report())
```

### Tempo Control

By default, dubbed segments are placed at their original timestamps with no speed adjustment. Use these flags to control pacing:

| Flag | Effect |
|------|--------|
| *(default)* | No tempo change — segments placed as-is |
| `--fixed-tempo <rate>` | Apply a constant speed multiplier to all segments (e.g. `1.1` = 10% faster) |
| `--dynamic-tempo` | Per-segment speed matching to fit original timing |
| `--max-tempo <rate>` | Cap for dynamic mode speed-up (default: `1.3`) |

`--fixed-tempo` takes precedence if both `--fixed-tempo` and `--dynamic-tempo` are given.

---

## Package Layout

```
mazinger/
├── __init__.py      # public API surface
├── __main__.py      # python -m mazinger
├── cli/             # CLI package
│   ├── __init__.py  # parser + main entry point
│   ├── _groups.py   # shared argument groups (DRY)
│   ├── _dub.py      # dub command
│   ├── _download.py # download command
│   ├── _transcribe.py
│   ├── _thumbnails.py
│   ├── _describe.py
│   ├── _translate.py
│   ├── _resegment.py
│   └── _speak.py    # speak command (voice synthesis)
├── pipeline.py      # MazingerDubber orchestrator class
├── paths.py         # ProjectPaths
├── download.py      # yt-dlp + ffmpeg
├── transcribe.py    # OpenAI Whisper / faster-whisper / WhisperX
├── thumbnails.py    # LLM timestamp selection + frame extraction
├── describe.py      # LLM content analysis
├── translate.py     # LLM SRT translation
├── resegment.py     # subtitle re-segmentation
├── tts.py           # Qwen3-TTS / Chatterbox voice cloning
├── assemble.py      # time-aligned audio assembly
├── srt.py           # SRT parsing/formatting
├── profiles.py      # HuggingFace voice profile downloader
└── utils.py         # shared helpers
```
