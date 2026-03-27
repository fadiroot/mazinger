"""Voice-cloned text-to-speech synthesis using Qwen3-TTS, Chatterbox, or vLLM-Omni."""

from __future__ import annotations

import abc
import gc
import logging
import os
from typing import Any, Literal

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

# Module-level model cache — keeps loaded TTS models in memory for reuse
_model_cache: dict[str, Any] = {}


def _cache_key(engine: str, model_name: str, device: str, dtype: str) -> str:
    """Build a unique key for the model cache."""
    return f"{engine}|{model_name}|{device}|{dtype}"


def _remove_from_cache(obj: Any) -> None:
    keys = [k for k, v in _model_cache.items() if v is obj]
    for k in keys:
        del _model_cache[k]

# ═══════════════════════════════════════════════════════════════════════════════
#  TTS Engine Type
# ═══════════════════════════════════════════════════════════════════════════════

TTSEngine = Literal["qwen", "chatterbox", "qwen-vllm"]

SUPPORTED_LANGUAGES = (
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian",
)


def validate_language(language: str) -> None:
    """Raise *ValueError* if *language* is not supported by Qwen TTS."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}. "
            f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Base Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class TTSWrapper(abc.ABC):
    """Unified adapter for TTS engines.

    Subclasses implement :meth:`synthesize` for single-segment generation
    and may override :meth:`synthesize_batch` for true batch support.
    """

    engine: str

    @abc.abstractmethod
    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        """Generate audio for *text*.  Returns ``(audio_array, sample_rate)``."""

    def synthesize_batch(
        self, items: list[tuple[str, str]],
    ) -> list[tuple[np.ndarray, int]]:
        """Synthesize multiple ``(text, language)`` pairs.

        The default implementation loops sequentially.  Engines with native
        batch support (e.g. vLLM-Omni) override this for parallel decoding.
        """
        return [self.synthesize(t, lang) for t, lang in tqdm(items, desc="Synthesising")]

    @abc.abstractmethod
    def unload(self) -> None:
        """Release GPU memory held by this engine."""


# ═══════════════════════════════════════════════════════════════════════════════
#  Qwen3-TTS Backend
# ═══════════════════════════════════════════════════════════════════════════════

def _load_qwen_model(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> Any:
    """Load a Qwen3-TTS model and return it."""
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Auto-detect compatible dtype when the device doesn't support the
    # requested precision (e.g. CPU or older GPUs without bfloat16).
    _is_cpu = "cpu" in str(device)
    if torch_dtype == torch.bfloat16 and (
        _is_cpu or not torch.cuda.is_bf16_supported()
    ):
        log.warning(
            "bfloat16 not supported on %s — falling back to float32", device,
        )
        torch_dtype = torch.float32
        dtype = "float32"
    elif torch_dtype == torch.float16 and _is_cpu:
        log.warning(
            "float16 not efficient on CPU — falling back to float32",
        )
        torch_dtype = torch.float32
        dtype = "float32"

    model = Qwen3TTSModel.from_pretrained(
        model_name, device_map=device, dtype=torch_dtype,
    )
    log.info("Loaded Qwen TTS model: %s on %s (%s)", model_name, device, dtype)
    return model


def _create_qwen_voice_prompt(model: Any, ref_audio: str, ref_text: str) -> Any:
    """Build a reusable voice-clone prompt for Qwen3-TTS."""
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    log.info("Qwen voice clone prompt created from %s", ref_audio)
    return prompt


def _synthesize_qwen(
    model: Any,
    voice_prompt: Any,
    text: str,
    language: str = "English",
) -> tuple[np.ndarray, int]:
    """Generate audio using Qwen3-TTS. Returns (audio_array, sample_rate)."""
    validate_language(language)
    wavs, sr = model.generate_voice_clone(
        text=text, language=language, voice_clone_prompt=voice_prompt,
    )
    return wavs[0], sr


class _QwenTTSWrapper(TTSWrapper):

    engine = "qwen"

    def __init__(self, model: Any, voice_prompt: Any):
        self.model = model
        self.voice_prompt = voice_prompt

    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        return _synthesize_qwen(self.model, self.voice_prompt, text, language)

    def unload(self) -> None:
        import torch
        _remove_from_cache(self.model)
        del self.voice_prompt, self.model
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Qwen TTS model unloaded, GPU memory freed.")


VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def design_voice(
    text: str,
    language: str,
    instruct: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> tuple[np.ndarray, int]:
    """Synthesise a reference clip using the Qwen3-TTS VoiceDesign model.

    The clip matches the voice described by *instruct* and is intended as
    input for :func:`create_voice_clone_prompt`.  The VoiceDesign model is
    freed from GPU memory after generation.
    """
    validate_language(language)
    model = load_model(VOICE_DESIGN_MODEL, device=device, dtype=dtype, engine="qwen")
    wavs, sr = model.generate_voice_design(
        text=text, language=language, instruct=instruct,
    )
    unload_model(model, force=True)
    return wavs[0], sr


# ═══════════════════════════════════════════════════════════════════════════════
#  Chatterbox Backend
# ═══════════════════════════════════════════════════════════════════════════════

def _load_chatterbox_model(device: str = "cuda", model_name: str = "ResembleAI/chatterbox") -> Any:
    """Load a Chatterbox TTS model and return it."""
    from chatterbox.tts import ChatterboxTTS

    # Chatterbox expects device without index for simple cases
    device_clean = device.split(":")[0] if ":" in device else device
    model = ChatterboxTTS.from_pretrained(device=device_clean)
    log.info("Loaded Chatterbox TTS model: %s on %s", model_name, device_clean)
    return model


def _synthesize_chatterbox(
    model: Any,
    audio_prompt_path: str,
    text: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Generate audio using Chatterbox. Returns (audio_array, sample_rate)."""
    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    # Chatterbox returns a torch tensor, convert to numpy
    audio_data = wav.squeeze().cpu().numpy()
    return audio_data, model.sr


class _ChatterboxTTSWrapper(TTSWrapper):

    engine = "chatterbox"

    def __init__(
        self,
        model: Any,
        ref_audio_path: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ):
        self.model = model
        self.ref_audio_path = ref_audio_path
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight

    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        return _synthesize_chatterbox(
            self.model, self.ref_audio_path, text,
            self.exaggeration, self.cfg_weight,
        )

    def unload(self) -> None:
        import torch
        _remove_from_cache(self.model)
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Chatterbox TTS model unloaded, GPU memory freed.")


# ═══════════════════════════════════════════════════════════════════════════════
#  vLLM-Omni Backend
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum reference audio duration (seconds) for the vLLM-Omni Talker.
# Longer clips overflow the Talker's ``max_model_len`` and produce empty audio.
_VLLM_REF_AUDIO_MAX_SECS = 15.0


def _maybe_trim_ref_audio(ref_audio: str, max_seconds: float = _VLLM_REF_AUDIO_MAX_SECS) -> str:
    """Return *ref_audio* if short enough, otherwise write a trimmed copy.

    Selects the best middle segment of *max_seconds* to preserve voice
    characteristics while fitting the Talker's context window.
    """
    try:
        info = sf.info(ref_audio)
        if info.duration <= max_seconds:
            return ref_audio
        import tempfile
        data, sr = sf.read(ref_audio)
        max_samples = int(max_seconds * sr)
        # Centre-crop for best voice representation
        start = max(0, (len(data) - max_samples) // 2)
        trimmed = data[start : start + max_samples]
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp_path, trimmed, sr)
        log.info(
            "Trimmed ref audio from %.1fs to %.1fs for vLLM-Omni: %s",
            info.duration,
            max_seconds,
            tmp_path,
        )
        return tmp_path
    except Exception as exc:
        log.warning("Could not check/trim ref audio, using as-is: %s", exc)
        return ref_audio

def _load_vllm_omni(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    stage_configs_path: str | None = None,
) -> Any:
    """Start the vLLM-Omni engine for Qwen3-TTS.

    The default stage config allocates ~30 % GPU per stage (~60 % total),
    leaving headroom for other models.  Pass *stage_configs_path* to a
    custom YAML if you need different memory limits.

    Uses ``enforce_eager=true`` for both stages to avoid ``torch.compile``
    compatibility issues.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm_omni import Omni

    # Use our bundled eager config when no custom path is provided.
    if stage_configs_path is None:
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        stage_configs_path = os.path.join(_pkg_dir, "qwen3_tts_eager.yaml")

    omni = Omni(
        model=model_name,
        stage_configs_path=stage_configs_path,
        stage_init_timeout=300,
    )
    log.info("Loaded vLLM-Omni TTS engine: %s", model_name)
    return omni


# Cached tokenizer / config for prompt-length estimation
_vllm_prompt_cache: dict[str, Any] = {}


def _estimate_vllm_prompt_len(
    additional_information: dict[str, Any],
    model_name: str,
) -> int:
    """Estimate the placeholder ``prompt_token_ids`` length for the Talker stage.

    Uses the speech tokenizer for exact ref_code_len when available,
    falling back to a duration-based estimate or 2048.
    """
    try:
        import torch as _torch
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        if model_name not in _vllm_prompt_cache:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, padding_side="left",
            )
            cfg = Qwen3TTSConfig.from_pretrained(
                model_name, trust_remote_code=True,
            )
            # Try loading the speech tokenizer for exact ref_code_len.
            speech_tok = None
            try:
                from transformers.utils import cached_file
                from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import (
                    Qwen3TTSTokenizer,
                )

                st_cfg_path = cached_file(model_name, "speech_tokenizer/config.json")
                if st_cfg_path:
                    speech_tok = Qwen3TTSTokenizer.from_pretrained(
                        os.path.dirname(st_cfg_path), torch_dtype=_torch.bfloat16,
                    )
                    log.info("Loaded speech tokenizer for exact ref_code_len estimation")
            except Exception as e:
                log.debug("Could not load speech tokenizer: %s", e)

            _vllm_prompt_cache[model_name] = (
                tok, getattr(cfg, "talker_config", None), speech_tok,
            )

        tok, tcfg, speech_tok = _vllm_prompt_cache[model_name]
        task_type = (additional_information.get("task_type") or ["Base"])[0]

        def _estimate_ref_code_len(ref_audio: object) -> int | None:
            audio_path = ref_audio[0] if isinstance(ref_audio, list) else ref_audio
            if not isinstance(audio_path, str) or not audio_path.strip():
                return None
            try:
                from vllm.multimodal.media import MediaConnector

                connector = MediaConnector(allowed_local_media_path="/")
                audio, sr = connector.fetch_audio(audio_path)

                # Use speech tokenizer for exact frame count when available.
                if speech_tok is not None:
                    import numpy as _np

                    wav_np = _np.asarray(audio, dtype=_np.float32)
                    enc = speech_tok.encode(wav_np, sr=int(sr), return_dict=True)
                    ref_code = getattr(enc, "audio_codes", None)
                    if isinstance(ref_code, list):
                        ref_code = ref_code[0] if ref_code else None
                    if ref_code is not None and hasattr(ref_code, "shape"):
                        shape = ref_code.shape
                        if len(shape) == 2:
                            return int(shape[0])
                        if len(shape) == 3:
                            return int(shape[1])

                # Fallback: estimate from duration.
                codec_hz = getattr(tcfg, "codec_frame_rate", None) or 12
                return int(len(audio) / sr * codec_hz)
            except Exception:
                return None

        return Qwen3TTSTalkerForConditionalGeneration \
            .estimate_prompt_len_from_additional_information(
                additional_information=additional_information,
                task_type=task_type,
                tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
                codec_language_id=getattr(tcfg, "codec_language_id", None),
                spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
                estimate_ref_code_len=_estimate_ref_code_len,
            )
    except Exception as exc:
        log.debug("Prompt-length estimation unavailable, using 2048: %s", exc)
        return 2048


class _VllmOmniTTSWrapper(TTSWrapper):

    engine = "qwen-vllm"

    def __init__(
        self, omni: Any, model_name: str, ref_audio: str, ref_text: str,
    ):
        self._omni = omni
        self._model_name = model_name
        self._ref_audio = _maybe_trim_ref_audio(ref_audio)
        self._ref_text = ref_text

    def _build_request(self, text: str, language: str) -> dict:
        additional = {
            "task_type": ["Base"],
            "ref_audio": [self._ref_audio],
            "ref_text": [self._ref_text],
            "text": [text],
            "language": [language],
            "x_vector_only_mode": [False],
            "max_new_tokens": [2048],
        }
        prompt_len = _estimate_vllm_prompt_len(additional, self._model_name)
        return {
            "prompt_token_ids": [0] * prompt_len,
            "additional_information": additional,
        }

    @staticmethod
    def _extract_audio(mm: dict) -> tuple[np.ndarray, int]:
        import torch

        audio = mm["audio"]
        sr_raw = mm["sr"]
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
        tensor = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
        return tensor.float().cpu().numpy().flatten(), sr

    def synthesize(self, text: str, language: str = "English") -> tuple[np.ndarray, int]:
        validate_language(language)
        req = self._build_request(text, language)
        mm = None
        for stage_out in self._omni.generate([req]):
            mm = stage_out.request_output.outputs[0].multimodal_output
        return self._extract_audio(mm)

    def synthesize_batch(
        self, items: list[tuple[str, str]],
    ) -> list[tuple[np.ndarray, int]]:
        for _, lang in items:
            validate_language(lang)
        requests = [self._build_request(text, lang) for text, lang in items]
        log.info("Batch-synthesising %d segments via vLLM-Omni", len(requests))
        results: dict[str, tuple[np.ndarray, int]] = {}
        for stage_out in self._omni.generate(requests):
            out = stage_out.request_output
            mm = out.outputs[0].multimodal_output
            results[out.request_id] = self._extract_audio(mm)
        return [results[str(i)] for i in range(len(items))]

    def unload(self) -> None:
        _remove_from_cache(self._omni)
        del self._omni
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("vLLM-Omni TTS engine unloaded.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    engine: TTSEngine = "qwen",
    chatterbox_model: str = "ResembleAI/chatterbox",
    stage_configs_path: str | None = None,
) -> Any:
    """Load a TTS model and return it.

    Parameters:
        model_name:        HuggingFace model identifier (used for Qwen / vLLM-Omni).
        device:            Target device (e.g. ``cuda:0``).
        dtype:             Weight dtype for Qwen (``bfloat16``, ``float16``, ``float32``).
        engine:            TTS engine: ``qwen``, ``chatterbox``, or ``qwen-vllm``.
        chatterbox_model:  HuggingFace model identifier for Chatterbox.
        stage_configs_path: Path to a vLLM-Omni stage config YAML (optional).

    Returns:
        The loaded model instance (or ``Omni`` engine for vLLM-Omni).
    """
    name = chatterbox_model if engine == "chatterbox" else model_name
    key = _cache_key(engine, name, device, dtype)
    if key in _model_cache:
        log.info("Reusing cached TTS model: %s", key)
        return _model_cache[key]

    if engine == "qwen":
        model = _load_qwen_model(model_name, device, dtype)
    elif engine == "chatterbox":
        model = _load_chatterbox_model(device, chatterbox_model)
    elif engine == "qwen-vllm":
        model = _load_vllm_omni(model_name, stage_configs_path)
    else:
        raise ValueError(f"Unknown TTS engine: {engine!r}")

    _model_cache[key] = model
    return model


def create_voice_prompt(
    model: Any,
    ref_audio: str,
    ref_text: str,
    engine: TTSEngine = "qwen",
    chatterbox_exaggeration: float = 0.5,
    chatterbox_cfg: float = 0.5,
    model_name: str | None = None,
) -> TTSWrapper:
    """Build a reusable voice-clone prompt from a reference recording.

    Parameters:
        model:     A loaded TTS model (from :func:`load_model`).
        ref_audio: Path to the reference audio file.
        ref_text:  Transcript of the reference audio (required for Qwen /
                   vLLM-Omni, ignored for Chatterbox).
        engine:    TTS engine: ``qwen``, ``chatterbox``, or ``qwen-vllm``.
        chatterbox_exaggeration: Exaggeration level for Chatterbox (0.0-1.0).
        chatterbox_cfg:          CFG weight for Chatterbox (0.0-1.0).
        model_name: HuggingFace model name (used by vLLM-Omni for prompt
                    estimation; defaults to ``Qwen/Qwen3-TTS-12Hz-1.7B-Base``).

    Returns:
        A :class:`TTSWrapper` instance ready for synthesis.
    """
    if engine == "qwen":
        voice_prompt = _create_qwen_voice_prompt(model, ref_audio, ref_text)
        return _QwenTTSWrapper(model, voice_prompt)
    elif engine == "chatterbox":
        log.info("Chatterbox voice clone configured from %s", ref_audio)
        return _ChatterboxTTSWrapper(
            model, ref_audio, chatterbox_exaggeration, chatterbox_cfg,
        )
    elif engine == "qwen-vllm":
        name = model_name or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        log.info("vLLM-Omni voice clone configured from %s", ref_audio)
        return _VllmOmniTTSWrapper(model, name, ref_audio, ref_text)
    else:
        raise ValueError(f"Unknown TTS engine: {engine!r}")


def synthesize_segments(
    model: Any,
    voice_prompt: TTSWrapper | Any,
    srt_entries: list[dict],
    output_dir: str,
    *,
    language: str = "English",
    force_reset: bool = False,
) -> list[dict]:
    """Generate TTS audio for each SRT entry and save as WAV files.

    Parameters:
        model:        A loaded TTS model (can be ignored if voice_prompt is TTSWrapper).
        voice_prompt: The voice-clone prompt from :func:`create_voice_prompt`.
                      Can be a :class:`TTSWrapper` or a legacy Qwen prompt.
        srt_entries:  Parsed SRT entries (list of dicts with ``idx``, ``start``,
                      ``end``, ``text``).
        output_dir:   Directory in which to save individual segment WAV files.
        language:     Target language name (e.g. ``English``).
        force_reset:  When ``True``, delete all existing segment files in
                      *output_dir* before generating, so every segment is
                      re-synthesised from scratch.

    Returns:
        A list of segment info dicts with keys ``idx``, ``start``, ``end``,
        ``target_dur``, ``wav_path``, and ``actual_dur``.
    """
    if force_reset and os.path.isdir(output_dir):
        import glob
        for f in glob.glob(os.path.join(output_dir, "seg_*.wav")):
            os.remove(f)
        log.info("Force-reset: cleared existing segments in %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    segment_info: list[dict] = []
    pending: list[tuple[int, str, str]] = []  # (index, text, wav_path)

    for entry in srt_entries:
        target_dur = entry["end"] - entry["start"]
        text = entry["text"].strip()
        wav_path = os.path.join(output_dir, f"seg_{entry['idx'].zfill(4)}.wav")

        rec: dict[str, Any] = {
            "idx": entry["idx"],
            "start": entry["start"],
            "end": entry["end"],
            "target_dur": target_dur,
        }

        if not text:
            rec.update(wav_path=None, actual_dur=0)
        elif os.path.isfile(wav_path) and os.path.getsize(wav_path) > 0:
            actual_dur = sf.info(wav_path).duration
            log.debug("Skipping existing segment %s (%.2fs)", wav_path, actual_dur)
            rec.update(wav_path=wav_path, actual_dur=actual_dur, _skipped=True)
        else:
            rec.update(wav_path=wav_path, actual_dur=0)
            pending.append((len(segment_info), text, wav_path))

        segment_info.append(rec)

    # Synthesize all pending segments (batch when the engine supports it)
    if pending:
        use_wrapper = isinstance(voice_prompt, TTSWrapper)
        items = [(text, language) for _, text, _ in pending]

        if use_wrapper:
            results = voice_prompt.synthesize_batch(items)
        else:
            # Legacy Qwen API (backward compatibility)
            results = []
            for text, lang in tqdm(items, desc="Synthesising"):
                wavs, sr = model.generate_voice_clone(
                    text=text, language=lang, voice_clone_prompt=voice_prompt,
                )
                results.append((wavs[0], sr))

        for (seg_idx, _, wav_path), (audio_data, sr) in zip(pending, results):
            actual_dur = len(audio_data) / sr
            sf.write(wav_path, audio_data, sr)
            segment_info[seg_idx]["actual_dur"] = actual_dur

    produced = sum(1 for s in segment_info if s["wav_path"])
    skipped = sum(1 for s in segment_info if s.get("_skipped"))
    overflow_segs = [
        s for s in segment_info
        if s["wav_path"] and s["actual_dur"] > s["target_dur"] * 1.05
    ]
    log.info(
        "Synthesised %d/%d segments (%d cached) -> %s",
        produced, len(srt_entries), skipped, output_dir,
    )
    if overflow_segs:
        total_overflow = sum(s["actual_dur"] - s["target_dur"] for s in overflow_segs)
        log.warning(
            "%d/%d segments exceed target duration (total overflow: %.2fs). "
            "Segments: %s",
            len(overflow_segs), len(srt_entries), total_overflow,
            ", ".join(
                f'{s["idx"]}({s["actual_dur"]:.1f}s/{s["target_dur"]:.1f}s)'
                for s in overflow_segs
            ),
        )
    return segment_info


def unload_model(model: Any, *, force: bool = False) -> None:
    """Unload a TTS model and free GPU memory.

    By default the model is kept in the module-level cache so subsequent
    calls to :func:`load_model` with the same parameters return instantly.
    Pass ``force=True`` to actually remove the model from memory.
    """
    if not force:
        log.info("TTS model kept in memory for reuse (pass force=True to free)")
        return

    if isinstance(model, TTSWrapper):
        model.unload()
        return

    import torch

    _remove_from_cache(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log.info("TTS model unloaded, GPU memory freed.")
