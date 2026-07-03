from __future__ import annotations
from faster_whisper import WhisperModel

_MODEL: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        # Downloads ~145 MB on first call; cached in ~/.cache/huggingface/
        _MODEL = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _MODEL


def transcribe(wav_path: str) -> str:
    """Transcribe a 16 kHz mono WAV file. Returns stripped text."""
    model = _get_model()
    segments, _ = model.transcribe(wav_path, beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()
