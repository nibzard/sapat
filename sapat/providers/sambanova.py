# ABOUTME: SambaNova Cloud transcription provider with LLM correction support
# ABOUTME: Uses SambaNova's OpenAI-compatible Whisper endpoint and chat completions API

import os
from typing import Dict

import requests

from sapat.providers import register
from sapat.providers.base import AudioFormat, ProviderConfig
from sapat.providers.openai_compat import OpenAICompatProvider


@register
class SambaNovaProvider(OpenAICompatProvider):
    name = "sambanova"
    config = ProviderConfig(
        required_env_vars=["SAMBANOVA_API_KEY"],
        max_file_size_mb=25.0,
        preferred_format=AudioFormat.MP3,
        supports_correction=True,
        default_model="Whisper-Large-v3",
    )
    base_url = "https://api.sambanova.ai/v1/audio/transcriptions"
    _env_key_for_auth = "SAMBANOVA_API_KEY"

    _MODEL_ALIASES: Dict[str, str] = {
        "w": "Whisper-Large-v3",
        "whisper": "Whisper-Large-v3",
        "whisper-large-v3": "Whisper-Large-v3",
    }

    def resolve_model(self, model_alias: str) -> str:
        return self._MODEL_ALIASES.get(model_alias, model_alias)

    def correct_transcript(self, text: str, temperature: float = 0) -> str:
        url = "https://api.sambanova.ai/v1/chat/completions"
        api_key = os.getenv("SAMBANOVA_API_KEY", "")
        chat_model = os.getenv(
            "SAMBANOVA_CHAT_MODEL", "Meta-Llama-3.3-70B-Instruct"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = (
            "You are a helpful assistant. Your task is to correct any "
            "spelling discrepancies in the transcribed text. Make sure "
            "that the names of the products and persons are spelled "
            "correctly. Only add necessary punctuation such as periods, "
            "commas, and capitalization, and use only the context provided."
        )

        payload = {
            "model": chat_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"SambaNova correction failed ({response.status_code}): {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]
