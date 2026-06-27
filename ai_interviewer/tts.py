from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from config import get_key, get_key_int

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEEPGRAM_API_KEY = get_key("DEEPGRAM_API_KEY", "")
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"

# Voice model - confident, professional male voice for the interviewer
# Options: aura-orion-en (deep authoritative), aura-zeus-en (commanding),
#          aura-arcas-en (professional), aura-orpheus-en (smooth articulate)
DEFAULT_VOICE = get_key("TTS_VOICE", "aura-2-orion-en")

# Audio encoding settings
DEFAULT_ENCODING = get_key("TTS_ENCODING", "linear16")  # PCM 16-bit
DEFAULT_SAMPLE_RATE = get_key_int("TTS_SAMPLE_RATE", "24000")

# Container format for browser playback
DEFAULT_CONTAINER = get_key("TTS_CONTAINER", "wav")

# Chunk size for streaming (bytes)
STREAM_CHUNK_SIZE = get_key_int("TTS_CHUNK_SIZE", "4096")

# Available interviewer voices (for easy switching)
INTERVIEWER_VOICES = {
    "orion":   "aura-2-orion-en",    # Deep, authoritative (default)
    "zeus":    "aura-2-zeus-en",     # Commanding, powerful
    "arcas":   "aura-2-arcas-en",    # Professional
    "orpheus": "aura-2-orpheus-en",  # Smooth, articulate
    "helios":  "aura-2-helios-en",   # Warm, energetic
    "perseus": "aura-2-perseus-en",  # Friendly, approachable
}


# ---------------------------------------------------------------------------
# Response objects
# ---------------------------------------------------------------------------

@dataclass
class TTSResult:
    """
    Result of a full (non-streaming) TTS synthesis.

    Attributes
    ----------
    audio_bytes : bytes
        The complete audio data.
    content_type : str
        MIME type of the audio (e.g. "audio/wav").
    duration_ms : float
        How long the API call took.
    text_length : int
        Length of the input text.
    voice : str
        Which voice model was used.
    """
    audio_bytes: bytes
    content_type: str = "audio/wav"
    duration_ms: float = 0.0
    text_length: int = 0
    voice: str = DEFAULT_VOICE

    @property
    def size_bytes(self) -> int:
        return len(self.audio_bytes)

    def to_dict(self) -> dict:
        return {
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "duration_ms": round(self.duration_ms, 1),
            "text_length": self.text_length,
            "voice": self.voice,
        }


# ---------------------------------------------------------------------------
# Internal: API request builder
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Get the Deepgram API key from env (read at call time, not import time)."""
    key = get_key("DEEPGRAM_API_KEY", "")
    if not key:
        raise RuntimeError(
            "DEEPGRAM_API_KEY environment variable is not set. "
            "Get one at https://console.deepgram.com/"
        )
    return key


def _build_headers() -> dict[str, str]:
    """Build HTTP headers for the Deepgram API."""
    return {
        "Authorization": f"Token {_get_api_key()}",
        "Content-Type": "application/json",
    }


def _build_params(
    voice: Optional[str] = None,
    encoding: Optional[str] = None,
    sample_rate: Optional[int] = None,
    container: Optional[str] = None,
) -> dict[str, str]:
    """Build query parameters for the TTS API."""
    params = {
        "model": voice or DEFAULT_VOICE,
        "encoding": encoding or DEFAULT_ENCODING,
        "sample_rate": str(sample_rate or DEFAULT_SAMPLE_RATE),
    }
    cont = container or DEFAULT_CONTAINER
    if cont:
        params["container"] = cont
    return params


# ---------------------------------------------------------------------------
# Full synthesis (non-streaming)
# ---------------------------------------------------------------------------

def synthesize(
    text: str,
    voice: Optional[str] = None,
    encoding: Optional[str] = None,
    sample_rate: Optional[int] = None,
    container: Optional[str] = None,
) -> TTSResult:
    """
    Convert text to speech and return the full audio.

    Use this for short texts or when you need the complete audio
    before sending (e.g. saving to file, non-streaming playback).

    Parameters
    ----------
    text : str
        The text to synthesize.
    voice : str | None
        Deepgram voice model. Defaults to DEFAULT_VOICE.
    encoding : str | None
        Audio encoding. Defaults to "linear16".
    sample_rate : int | None
        Sample rate in Hz. Defaults to 24000.
    container : str | None
        Container format ("wav", "none"). Defaults to "wav".

    Returns
    -------
    TTSResult
        Complete audio bytes and metadata.
    """
    if not text.strip():
        raise ValueError("Text is empty")

    headers = _build_headers()
    params = _build_params(voice, encoding, sample_rate, container)
    body = {"text": text}

    start = time.perf_counter()

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            DEEPGRAM_TTS_URL,
            headers=headers,
            params=params,
            json=body,
        )
        response.raise_for_status()

    elapsed_ms = (time.perf_counter() - start) * 1000

    content_type = response.headers.get("content-type", "audio/wav")

    result = TTSResult(
        audio_bytes=response.content,
        content_type=content_type,
        duration_ms=elapsed_ms,
        text_length=len(text),
        voice=params["model"],
    )

    logger.info(
        f"TTS | {result.text_length} chars -> "
        f"{result.size_bytes:,} bytes "
        f"({result.duration_ms:.0f}ms, voice={result.voice})"
    )

    return result


# ---------------------------------------------------------------------------
# Streaming synthesis (chunk-by-chunk for WebSocket)
# ---------------------------------------------------------------------------

async def synthesize_streaming(
    text: str,
    voice: Optional[str] = None,
    encoding: Optional[str] = None,
    sample_rate: Optional[int] = None,
    container: Optional[str] = None,
    chunk_size: int = STREAM_CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """
    Stream TTS audio chunks as they arrive from Deepgram.

    Use this in the WebSocket handler to send audio chunks to the
    frontend in real-time. The first chunk arrives in ~200ms.

    Parameters
    ----------
    text : str
        The text to synthesize.
    voice, encoding, sample_rate, container
        Same as ``synthesize()``.
    chunk_size : int
        Size of each audio chunk in bytes. Default: 4096.

    Yields
    ------
    bytes
        Audio data chunks (stream to WebSocket as they arrive).

    Example
    -------
    >>> async for chunk in synthesize_streaming("Hello, how are you?"):
    ...     await websocket.send_bytes(chunk)
    """
    if not text.strip():
        return

    headers = _build_headers()
    params = _build_params(voice, encoding, sample_rate, container)
    body = {"text": text}

    start = time.perf_counter()
    total_bytes = 0
    chunk_count = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            "POST",
            DEEPGRAM_TTS_URL,
            headers=headers,
            params=params,
            json=body,
        ) as response:
            response.raise_for_status()

            async for chunk in response.aiter_bytes(chunk_size):
                total_bytes += len(chunk)
                chunk_count += 1
                yield chunk

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        f"TTS stream | {len(text)} chars -> "
        f"{total_bytes:,} bytes in {chunk_count} chunks "
        f"({elapsed_ms:.0f}ms)"
    )


# ---------------------------------------------------------------------------
# Convenience: synthesize and save to file
# ---------------------------------------------------------------------------

def synthesize_to_file(
    text: str,
    filepath: str,
    voice: Optional[str] = None,
) -> TTSResult:
    """
    Synthesize text and save the audio to a file.

    Parameters
    ----------
    text : str
        Text to speak.
    filepath : str
        Output file path (e.g. "output.wav").
    voice : str | None
        Voice model to use.

    Returns
    -------
    TTSResult
    """
    result = synthesize(text, voice=voice)

    with open(filepath, "wb") as f:
        f.write(result.audio_bytes)

    logger.info(f"TTS saved to: {filepath} ({result.size_bytes:,} bytes)")
    return result


# ---------------------------------------------------------------------------
# Voice info helper
# ---------------------------------------------------------------------------

def get_available_voices() -> dict[str, str]:
    """Return the available interviewer voice presets."""
    return dict(INTERVIEWER_VOICES)


def resolve_voice(name_or_model: str) -> str:
    """
    Resolve a voice name or model ID.

    Accepts either a preset name ("orion", "zeus") or a full model ID
    ("aura-2-orion-en"). Returns the full model ID.
    """
    if name_or_model in INTERVIEWER_VOICES:
        return INTERVIEWER_VOICES[name_or_model]
    return name_or_model
