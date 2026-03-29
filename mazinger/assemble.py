"""Time-align TTS segments and assemble the final dubbed audio track."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from mazinger.utils import get_audio_duration

log = logging.getLogger(__name__)

TARGET_SR = 24_000


def _load_and_resample(wav_path: str, target_sr: int) -> np.ndarray:
    """Load a WAV and convert to mono at *target_sr* using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_path,
            "-ar", str(target_sr), "-ac", "1", "-f", "f32le", "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def _tempo_stretch(
    wav_path: str,
    factor: float,
    out_path: str,
    sr: int,
) -> np.ndarray:
    """Change playback speed by *factor* using the ffmpeg ``atempo`` filter.

    ``factor > 1`` speeds up, ``factor < 1`` slows down.
    """
    filters: list[str] = []
    remaining = factor
    while remaining > 100.0:
        filters.append("atempo=100.0")
        remaining /= 100.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_path,
            "-filter:a", ",".join(filters),
            "-ar", str(sr), "-ac", "1", out_path,
        ],
        capture_output=True,
        check=True,
    )
    data, _ = sf.read(out_path, dtype="float32")
    return data


def _fade(audio: np.ndarray, sr: int, fade_ms: int = 30) -> np.ndarray:
    """Apply a short fade-in and fade-out to avoid clicks at segment edges."""
    fade_len = min(int(sr * fade_ms / 1000), len(audio) // 2)
    if fade_len < 2:
        return audio
    audio = audio.copy()
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    audio[:fade_len] *= ramp
    audio[-fade_len:] *= ramp[::-1]
    return audio


def _rms_energy(audio: np.ndarray, frame_len: int) -> np.ndarray:
    """Compute per-frame RMS energy (non-overlapping windows)."""
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return np.array([0.0], dtype=np.float32)
    trimmed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def _find_last_silence(audio: np.ndarray, sr: int, budget_samps: int,
                        silence_thresh_db: float = -40.0) -> int:
    """Find the last silence boundary before *budget_samps*.

    Returns a sample index where the audio can be safely trimmed without
    cutting through voiced speech.  Falls back to *budget_samps* when no
    silence is found.
    """
    frame_len = int(sr * 0.02)  # 20 ms frames
    energy = _rms_energy(audio, frame_len)
    thresh = 10 ** (silence_thresh_db / 20.0)
    budget_frame = min(budget_samps // frame_len, len(energy))

    # Walk backwards from the budget boundary to find a silent frame
    for i in range(budget_frame - 1, budget_frame // 2, -1):
        if energy[i] < thresh:
            return (i + 1) * frame_len
    return budget_samps


def _speech_density(audio: np.ndarray, sr: int,
                     silence_thresh_db: float = -40.0) -> float:
    """Fraction of frames containing voiced speech (0.0–1.0)."""
    frame_len = int(sr * 0.02)
    energy = _rms_energy(audio, frame_len)
    thresh = 10 ** (silence_thresh_db / 20.0)
    if len(energy) == 0:
        return 1.0
    return float(np.mean(energy >= thresh))


def assemble_timeline(
    segment_info: list[dict],
    original_duration: float,
    output_path: str,
    *,
    sample_rate: int = TARGET_SR,
    speed_threshold: float = 0.05,
    min_speed_ratio: float = 0.5,
    tempo_mode: str = "auto",
    fixed_tempo: float | None = None,
    max_tempo: float = 1.5,
    crossfade_ms: int = 50,
) -> str:
    """Assemble per-segment TTS WAVs into a single time-aligned audio file.

    Each segment is placed at its SRT start time on a silence-filled
    timeline matching *original_duration*.

    Parameters:
        segment_info:      List of dicts from :func:`mazinger.tts.synthesize_segments`.
        original_duration: Duration of the original audio in seconds.
        output_path:       Where to write the final WAV.
        sample_rate:       Target sample rate.
        speed_threshold:   Fractional tolerance before tempo-stretching is applied.
        min_speed_ratio:   Lowest allowed slowdown factor (default 0.5 = max 2× slower).
                           Prevents extreme stretching that sounds unnatural.
        tempo_mode:        ``auto`` — speed up only overflowing segments (default);
                           ``off`` — no tempo adjustment;
                           ``dynamic`` — per-segment speed matching (up and down);
                           ``fixed`` — apply *fixed_tempo* to every segment.
        fixed_tempo:       Tempo rate applied to all segments when
                           ``tempo_mode="fixed"`` (e.g. 1.1).
        max_tempo:         Upper speed limit for dynamic/auto mode (default 1.5).
        crossfade_ms:      Fade-in/out duration (ms) applied to each segment edge
                           to prevent clicks and smooth overlaps (default 50).

    Returns:
        The *output_path*.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_samples = int(original_duration * sample_rate)
    timeline = np.zeros(total_samples, dtype=np.float32)

    stats = {"sped_up": 0, "slowed_down": 0, "ok": 0, "skipped": 0, "truncated": 0}
    overflow_total = 0.0

    valid_segs = [s for s in segment_info if s.get("wav_path") is not None]
    valid_segs.sort(key=lambda s: s["start"])

    # -- Pass 1: load audio, compute per-segment overflow/slack -----------
    loaded: list[dict | None] = []
    for seg_i, seg in enumerate(valid_segs):
        if seg_i + 1 < len(valid_segs):
            gap_to_next = max(0.0, valid_segs[seg_i + 1]["start"] - seg["end"])
        else:
            gap_to_next = max(0.0, original_duration - seg["end"])

        raw_audio = _load_and_resample(seg["wav_path"], sample_rate)
        actual_dur = len(raw_audio) / sample_rate
        if actual_dur <= 0:
            loaded.append(None)
            continue

        target_dur = seg["target_dur"]
        budget_dur = target_dur + gap_to_next
        slack = budget_dur - actual_dur  # positive = headroom, negative = overflow

        loaded.append({
            "raw": raw_audio,
            "actual_dur": actual_dur,
            "target_dur": target_dur,
            "budget_dur": budget_dur,
            "gap": gap_to_next,
            "slack": slack,
        })

    # -- Pass 2: redistribute slack from short neighbours to overflowing --
    if tempo_mode in ("auto", "dynamic"):
        for i, info in enumerate(loaded):
            if info is None or info["slack"] >= 0:
                continue
            needed = -info["slack"]
            # Try borrowing from the previous segment, then the next
            for nb in (i - 1, i + 1):
                if nb < 0 or nb >= len(loaded) or loaded[nb] is None:
                    continue
                give = min(loaded[nb]["slack"] * 0.5, needed)
                if give <= 0:
                    continue
                info["budget_dur"] += give
                info["slack"] += give
                loaded[nb]["budget_dur"] -= give
                loaded[nb]["slack"] -= give
                needed -= give
                if needed <= 0:
                    break

    # -- Pass 3: tempo-stretch, trim, and place each segment --------------
    placed_end = 0  # sample index: rightmost edge placed so far

    for seg_i, seg in enumerate(tqdm(valid_segs, desc="Aligning")):
        info = loaded[seg_i]
        if info is None:
            stats["skipped"] += 1
            continue

        raw_audio = info["raw"]
        target_dur = info["target_dur"]
        budget_dur = info["budget_dur"]
        actual_dur = info["actual_dur"]
        budget_samps = int(budget_dur * sample_rate)
        start_samp = int(seg["start"] * sample_rate)

        speed_ratio = actual_dur / target_dur

        # -- Adaptive max_tempo: allow more stretch for sparser speech ----
        density = _speech_density(raw_audio, sample_rate)
        seg_max_tempo = max_tempo + (1.0 - density) * 0.4  # up to +0.4 extra

        if tempo_mode == "fixed" and fixed_tempo is not None:
            stretched_path = seg["wav_path"].replace(".wav", "_stretched.wav")
            audio = _tempo_stretch(seg["wav_path"], fixed_tempo, stretched_path, sample_rate)
            stats["sped_up"] += 1
        elif tempo_mode in ("dynamic", "auto"):
            if speed_ratio > 1.0 + speed_threshold:
                effective_ratio = min(speed_ratio, seg_max_tempo)
                stretched_path = seg["wav_path"].replace(".wav", "_stretched.wav")
                audio = _tempo_stretch(seg["wav_path"], effective_ratio, stretched_path, sample_rate)
                stats["sped_up"] += 1
                if effective_ratio < speed_ratio:
                    stretched_dur = actual_dur / effective_ratio
                    overflow_secs = max(0, stretched_dur - budget_dur)
                    overflow_total += overflow_secs
                    if overflow_secs > 0:
                        log.warning(
                            "Seg %s: TTS=%.2fs > target=%.2fs (budget=%.2fs), "
                            "capped speed-up at %.2fx (needed %.2fx) — "
                            "%.2fs overflow",
                            seg["idx"], actual_dur, target_dur, budget_dur,
                            effective_ratio, speed_ratio, overflow_secs,
                        )
            elif tempo_mode == "dynamic" and speed_ratio < 1.0 - speed_threshold:
                effective_ratio = max(speed_ratio, min_speed_ratio)
                slowed_path = seg["wav_path"].replace(".wav", "_slowed.wav")
                audio = _tempo_stretch(seg["wav_path"], effective_ratio, slowed_path, sample_rate)
                stats["slowed_down"] += 1
            else:
                audio = raw_audio
                stats["ok"] += 1
        else:
            if speed_ratio > 1.0 + speed_threshold:
                overflow_secs = max(0, actual_dur - budget_dur)
                if overflow_secs > 0:
                    overflow_total += overflow_secs
                    log.warning(
                        "Seg %s: TTS=%.2fs > target=%.2fs (budget=%.2fs, overflow %.2fs). "
                        "Consider using tempo_mode='auto'.",
                        seg["idx"], actual_dur, target_dur, budget_dur, overflow_secs,
                    )
                    stats["truncated"] += 1
                else:
                    stats["ok"] += 1
            else:
                stats["ok"] += 1
            audio = raw_audio

        # -- Trim at a silence boundary instead of a hard sample cut ------
        if len(audio) > budget_samps:
            trim_at = _find_last_silence(audio, sample_rate, budget_samps)
            audio = audio[:trim_at]

        audio = _fade(audio, sample_rate, crossfade_ms)

        # -- Center short segments in their target window -----------------
        placed_dur = len(audio) / sample_rate
        if placed_dur < target_dur * 0.85:
            pad = int((target_dur - placed_dur) * sample_rate * 0.5)
            start_samp += pad

        # -- Resolve overlap with previous segment via crossfade ----------
        if start_samp < placed_end:
            overlap_len = placed_end - start_samp
            xf_len = min(overlap_len, int(sample_rate * crossfade_ms / 1000))
            if xf_len > 1 and xf_len <= len(audio):
                ramp = np.linspace(0.0, 1.0, xf_len, dtype=np.float32)
                # Fade out the existing timeline in the overlap zone
                timeline[start_samp: start_samp + xf_len] *= (1.0 - ramp)
                # Fade in the new segment's leading edge
                audio = audio.copy()
                audio[:xf_len] *= ramp

        # -- Place on timeline --------------------------------------------
        end_samp = min(start_samp + len(audio), total_samples)
        actual_len = end_samp - start_samp
        if actual_len > 0:
            timeline[start_samp:end_samp] += audio[:actual_len]
            placed_end = max(placed_end, end_samp)

    stats["skipped"] += len(segment_info) - len(valid_segs)

    peak = np.max(np.abs(timeline))
    if peak > 1.0:
        log.info("Normalising peak %.2f to 1.0", peak)
        timeline /= peak

    sf.write(output_path, timeline, sample_rate)

    if overflow_total > 0.1:
        log.warning(
            "Total overflow: %.2fs of TTS audio was trimmed to fit the timeline. "
            "Consider reducing translation word count (--duration-budget) "
            "or increasing --max-tempo.",
            overflow_total,
        )

    log.info(
        "Timeline assembled: %.2fs (sped_up=%d, slowed_down=%d, ok=%d, skipped=%d, truncated=%d)",
        total_samples / sample_rate,
        stats["sped_up"], stats["slowed_down"], stats["ok"], stats["skipped"],
        stats["truncated"],
    )
    return output_path


def _measure_loudness(path: str) -> float:
    """Return integrated loudness (LUFS) of an audio file via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path,
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    import json as _json, re as _re
    m = _re.search(r'\{[^}]+"input_i"[^}]+\}', result.stderr, _re.DOTALL)
    if m:
        data = _json.loads(m.group())
        return float(data["input_i"])
    return -24.0


def _extract_background(audio_path: str, out_path: str, sr: int = TARGET_SR) -> str:
    """Extract non-vocal background from *audio_path*.

    Uses demucs (htdemucs model) for high-quality source separation.
    Falls back to spectral masking via librosa when demucs is unavailable.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        import torch, torchaudio
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        log.info("Extracting background with demucs")
        model = get_model("htdemucs")
        model.eval()
        wav, wav_sr = torchaudio.load(audio_path)
        if wav_sr != model.samplerate:
            wav = torchaudio.functional.resample(wav, wav_sr, model.samplerate)
        with torch.no_grad():
            sources = apply_model(model, wav.unsqueeze(0))
        # sources: (1, num_stems, channels, samples)
        vocals_idx = model.sources.index("vocals")
        bg = sources[0]
        bg[vocals_idx] = 0
        bg_np = bg.sum(dim=0).mean(dim=0).cpu().numpy()
        import librosa
        if model.samplerate != sr:
            bg_np = librosa.resample(bg_np, orig_sr=model.samplerate, target_sr=sr)
        sf.write(out_path, bg_np, sr)
    except Exception as exc:
        log.info("Demucs unavailable (%s), using spectral masking fallback", exc)
        import librosa
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        S = librosa.stft(y)
        H, P = librosa.decompose.hpss(np.abs(S), kernel_size=31, margin=4.0)
        mask = P / (H + P + 1e-10)
        bg = librosa.istft(S * mask, length=len(y))
        sf.write(out_path, bg, sr)
    return out_path


def post_process(
    dubbed_path: str,
    original_audio: str,
    output_path: str,
    *,
    loudness_match: bool = True,
    mix_background: bool = True,
    background_volume: float = 0.15,
) -> str:
    """Apply loudness normalisation and background audio mixing.

    Parameters:
        dubbed_path:       Path to the assembled TTS audio.
        original_audio:    Path to the original source audio.
        output_path:       Where to write the processed result.
        loudness_match:    Match dubbed loudness to the original.
        mix_background:    Extract and mix background from original.
        background_volume: Gain multiplier for the background layer (0.0–1.0).
    """
    if not loudness_match and not mix_background:
        if dubbed_path != output_path:
            shutil.copy2(dubbed_path, output_path)
        return output_path

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    work = dubbed_path

    # -- loudness matching ------------------------------------------------
    if loudness_match:
        target_lufs = _measure_loudness(original_audio)
        target_lufs = max(target_lufs, -30.0)  # safety floor
        norm_path = output_path + ".norm.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", work,
             "-af", f"loudnorm=I={target_lufs:.1f}:TP=-1.5:LRA=11",
             "-ar", str(TARGET_SR), "-ac", "1", norm_path],
            capture_output=True, check=True,
        )
        work = norm_path
        log.info("Loudness matched to %.1f LUFS", target_lufs)

    # -- background mixing ------------------------------------------------
    if mix_background:
        bg_path = os.path.join(os.path.dirname(output_path), "background.wav")
        _extract_background(original_audio, bg_path, sr=TARGET_SR)
        log.info("Background audio saved: %s", bg_path)

        dur_dub = get_audio_duration(work)
        mix_path = output_path + ".mix.wav"
        filt = (
            f"[1:a]atrim=0:{dur_dub:.3f},asetpts=PTS-STARTPTS,"
            f"volume={background_volume:.2f}[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:weights=1 {background_volume:.2f}[out]"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", work, "-i", bg_path,
             "-filter_complex", filt, "-map", "[out]",
             "-ar", str(TARGET_SR), "-ac", "1", mix_path],
            capture_output=True, check=True,
        )
        work = mix_path
        log.info("Mixed background at volume %.0f%%", background_volume * 100)

    # -- move final result into place ------------------------------------
    if work != output_path:
        shutil.move(work, output_path)

    # cleanup temp files (background.wav is kept for inspection)
    for suffix in (".norm.wav", ".mix.wav"):
        tmp = output_path + suffix
        if os.path.exists(tmp) and tmp != output_path:
            os.remove(tmp)

    return output_path


def mux_video(video_path: str, audio_path: str, output_path: str) -> str | None:
    """Replace the audio track of *video_path* with *audio_path*.

    Uses ffmpeg to copy the video stream and encode the new audio.
    Returns *output_path*, or ``None`` if ffmpeg is not installed.
    """
    if shutil.which("ffmpeg") is None:
        log.warning(
            "ffmpeg not found — cannot produce dubbed video. "
            "Install ffmpeg (e.g. 'apt install ffmpeg' or 'brew install ffmpeg') "
            "and re-run with --output-type video."
        )
        return None
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    log.info("Muxed video saved: %s", output_path)
    return output_path
