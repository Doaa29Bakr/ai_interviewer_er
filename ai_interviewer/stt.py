"""
Speech-to-Text (STT) — Groq Whisper
=====================================

Transcribes candidate audio to text using Groq's Whisper Large V3 API.

The WebSocket handler receives audio bytes from the frontend, passes them
here, and gets back the transcription text to feed into the orchestrator.

Flow
----
React (mic) -> WebSocket (audio bytes) -> ``transcribe()`` -> text
-> ``orchestrator.handle_answer(text)``
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from config import get_key

from groq import Groq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WHISPER_MODEL = get_key("WHISPER_MODEL", "whisper-large-v3")

# Default language for transcription.
# Set to None or "auto" to let Whisper auto-detect.
# Useful values: "en", "ar", "fr", "de", etc.
DEFAULT_LANGUAGE = get_key("STT_LANGUAGE", "en")

# Supported audio formats that Whisper accepts
SUPPORTED_FORMATS = {"wav", "mp3", "webm", "ogg", "flac", "m4a", "mp4"}


# ---------------------------------------------------------------------------
# Response object
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    """
    Result of a speech-to-text transcription.

    Attributes
    ----------
    text : str
        The transcribed text.
    language : str
        Detected or specified language.
    duration_ms : float
        How long the transcription API call took (milliseconds).
    audio_size_bytes : int
        Size of the input audio.
    model : str
        Which Whisper model was used.
    """
    text: str
    language: str = ""
    duration_ms: float = 0.0
    audio_size_bytes: int = 0
    model: str = WHISPER_MODEL

    @property
    def is_empty(self) -> bool:
        """True if no meaningful text was transcribed."""
        return len(self.text.strip()) == 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "duration_ms": round(self.duration_ms, 1),
            "audio_size_bytes": self.audio_size_bytes,
            "model": self.model,
            "is_empty": self.is_empty,
        }


# ---------------------------------------------------------------------------
# Groq Whisper client
# ---------------------------------------------------------------------------

_client: Optional[Groq] = None


def _get_client() -> Groq:
    """Lazy-init the Groq client (reuse across calls)."""
    global _client
    if _client is None:
        api_key = get_key("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set")
        _client = Groq(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Main transcription function
# ---------------------------------------------------------------------------

def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> TranscriptionResult:
    """
    Transcribe audio bytes to text using Groq Whisper.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio data (WAV, WebM, MP3, OGG, FLAC, M4A).
    filename : str
        Filename hint for the audio format (e.g. "audio.webm").
        Whisper uses the extension to determine the codec.
    language : str | None
        ISO 639-1 language code. Defaults to ``DEFAULT_LANGUAGE``.
        Set to ``None`` for auto-detection.
    prompt : str | None
        Optional prompt to guide transcription (helps with technical terms).
        Example: "Python, Django, REST API, PostgreSQL, overfitting"

    Returns
    -------
    TranscriptionResult
        The transcription text and metadata.

    Raises
    ------
    ValueError
        If audio_bytes is empty or the format is unsupported.
    RuntimeError
        If the Groq API call fails.
    """
    # -- Validate input ----------------------------------------------------
    if not audio_bytes:
        raise ValueError("audio_bytes is empty")

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension and extension not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format: .{extension}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    # -- Resolve language --------------------------------------------------
    lang = language or DEFAULT_LANGUAGE
    if lang and lang.lower() == "auto":
        lang = None  # Let Whisper auto-detect

    # -- Build the file-like object ----------------------------------------
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename  # Groq SDK uses .name to detect format

    # -- Call Groq Whisper -------------------------------------------------
    client = _get_client()
    start = time.perf_counter()

    try:
        kwargs = {
            "model": WHISPER_MODEL,
            "file": audio_file,
        }
        if lang:
            kwargs["language"] = lang
        if prompt:
            kwargs["prompt"] = prompt

        transcription = client.audio.transcriptions.create(**kwargs)

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = TranscriptionResult(
            text=transcription.text.strip(),
            language=lang or "auto",
            duration_ms=elapsed_ms,
            audio_size_bytes=len(audio_bytes),
        )

        logger.info(
            f"STT | {result.audio_size_bytes} bytes -> "
            f"'{result.text}' "
            f"({result.duration_ms:.0f}ms, lang={result.language})"
        )

        return result

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(f"STT failed after {elapsed_ms:.0f}ms: {exc}")
        raise RuntimeError(f"Whisper transcription failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Convenience: transcribe from a file path (for testing)
# ---------------------------------------------------------------------------

def transcribe_file(
    filepath: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> TranscriptionResult:
    """
    Transcribe an audio file from disk.

    Parameters
    ----------
    filepath : str
        Path to the audio file.
    language : str | None
        Language code or None for auto-detect.
    prompt : str | None
        Optional transcription hint.

    Returns
    -------
    TranscriptionResult
    """
    filename = os.path.basename(filepath)

    with open(filepath, "rb") as f:
        audio_bytes = f.read()

    return transcribe(
        audio_bytes=audio_bytes,
        filename=filename,
        language=language,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Technical prompt builder (helps Whisper with domain-specific terms)
# ---------------------------------------------------------------------------

def build_technical_prompt(skills: list[str], topic: str = "") -> str:
    """
    Build a Whisper prompt hint from the candidate's skills and interview topic.

    Whisper uses the prompt as a "style guide" — including technical terms
    helps it correctly transcribe jargon like "PyTorch", "k-means",
    "Django ORM", etc.

    Parameters
    ----------
    skills : list[str]
        Candidate's skills from the interview plan.
    topic : str
        Interview topic.

    Returns
    -------
    str
        A prompt string for Whisper.
    """
    terms = list(skills)
    if topic:
        terms.append(topic)

    # Common technical terms that Whisper often misses
    common_tech = [
        "Python", "API", "REST", "SQL", "NoSQL", "Docker",
        "Kubernetes", "AWS", "GPU", "CPU", "RAM",
    ]

    # Only add common terms that aren't already covered
    existing_lower = {t.lower() for t in terms}
    for t in common_tech:
        if t.lower() not in existing_lower:
            terms.append(t)

    return ", ".join(terms)
