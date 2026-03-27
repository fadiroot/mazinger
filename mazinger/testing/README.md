# Mazinger Testing Scripts

Benchmark and smoke-test scripts for the Mazinger dubbing pipeline.

Run any script from the repo root with the virtual environment activated:

```bash
source /workspace/.venv/bin/activate
python -m mazinger.testing.<script_name>
```

## Scripts

| Script | Purpose | GPU required |
|--------|---------|:------------:|
| `bench_qwen_tts` | Benchmark Qwen3-TTS inference speed (latency, RTF, throughput) on the current GPU | yes |

### bench_qwen_tts

Benchmark native Qwen3-TTS models. Uses a real mazinger voice theme for reference audio.

```bash
# Default: 1.7B model, narrator-m theme
python -m mazinger.testing.bench_qwen_tts --warmup 1 --runs 2

# 0.6B model
python -m mazinger.testing.bench_qwen_tts --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --warmup 1 --runs 2

# Custom voice theme
python -m mazinger.testing.bench_qwen_tts --voice-theme warm-f --warmup 1 --runs 2
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace model ID |
| `--device` | `cuda:0` | Device |
| `--dtype` | `bfloat16` | Weight dtype (`bfloat16`, `float16`, `float32`) |
| `--warmup` | `2` | Warmup iterations |
| `--runs` | `3` | Benchmark runs per sentence |
| `--voice-theme` | `narrator-m` | Mazinger voice theme for reference audio |
| `--output-dir` | `mazinger/testing/output/` | Directory for output WAVs and results JSON |

**Output:** WAV files and `results.json` saved to `output/<model-short-name>/` (git-ignored).
