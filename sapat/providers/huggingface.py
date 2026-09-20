# ABOUTME: Hugging Face Inference transcription provider
# ABOUTME: Sends audio to Hugging Face's hosted ASR models (e.g. Whisper)

import os
from typing import Optional

import requests

from sapat.providers import register
from sapat.providers.base import (
    AudioFormat,
    ProviderConfig,
    TranscriptionProvider,
    TranscriptionResult,
)

_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}


@register
class HuggingFaceProvider(TranscriptionProvider):
    """Transcribe via the Hugging Face Inference API.

    Posts raw audio bytes to ``hf-inference`` for a hosted ASR model such as
    ``openai/whisper-large-v3``. The endpoint auto-detects language, so
    ``language``, ``prompt``, and ``temperature`` are accepted for interface
    compatibility but not sent.
    """

    name = "huggingface"
    config = ProviderConfig(
        required_env_vars=["HF_TOKEN"],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=False,
        default_model="openai/whisper-large-v3",
    )

    def transcribe(
        self,
        audio_file: str,
        model: str,
        language: str = "en",
        prompt: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ) -> TranscriptionResult:
        url = f"https://router.huggingface.co/hf-inference/models/{model}"
        ext = os.path.splitext(audio_file)[1].lstrip(".").lower()
        headers = {
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
            "Content-Type": _CONTENT_TYPES.get(ext, "audio/mpeg"),
            "x-wait-for-model": "true",
        }

        with open(audio_file, "rb") as f:
            response = requests.post(url, headers=headers, data=f.read())

        if response.status_code != 200:
            raise RuntimeError(
                f"Hugging Face transcription failed "
                f"({response.status_code}): {response.text}"
            )

        result = response.json()
        return TranscriptionResult(text=result.get("text", ""))
