# ABOUTME: Unit tests for SambaNova Cloud transcription and LLM correction provider
# ABOUTME: Verifies Whisper transcription requests, model aliases, chat correction, and discovery

import os
from unittest.mock import patch

import pytest

from sapat.providers import get_available_providers
from sapat.providers.base import AudioFormat, TranscriptionResult


@pytest.fixture
def audio_file(tmp_path):
    """Create a temporary MP3 audio file for testing."""
    path = tmp_path / "test.mp3"
    path.write_bytes(b"fake audio data")
    return str(path)


class FakeResponse:
    """Minimal fake requests.Response."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def json(self):
        return self.payload


class TestSambaNovaProvider:
    @patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.openai_compat.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello sambanova"})

        from sapat.providers.sambanova import SambaNovaProvider

        provider = SambaNovaProvider()
        result = provider.transcribe(
            audio_file,
            model="Whisper-Large-v3",
            language="en",
            prompt="Daytona SAPAT",
            temperature=0.0,
        )

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello sambanova"

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert (
            mock_post.call_args.args[0]
            == "https://api.sambanova.ai/v1/audio/transcriptions"
        )
        assert kwargs["data"]["model"] == "Whisper-Large-v3"
        assert kwargs["data"]["language"] == "en"
        assert kwargs["data"]["prompt"] == "Daytona SAPAT"
        assert kwargs["data"]["temperature"] == 0.0
        assert kwargs["data"]["response_format"] == "json"
        assert "file" in kwargs["files"]

    @patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-key"}, clear=False)
    def test_default_model_and_config(self):
        from sapat.providers.sambanova import SambaNovaProvider

        assert SambaNovaProvider.config.default_model == "Whisper-Large-v3"
        assert SambaNovaProvider.config.supports_correction is True
        assert SambaNovaProvider.config.max_file_size_mb == 25.0
        assert SambaNovaProvider.config.preferred_format == AudioFormat.MP3

    @patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-key"}, clear=False)
    def test_resolve_model_aliases(self):
        from sapat.providers.sambanova import SambaNovaProvider

        provider = SambaNovaProvider()
        assert provider.resolve_model("w") == "Whisper-Large-v3"
        assert provider.resolve_model("whisper") == "Whisper-Large-v3"
        assert provider.resolve_model("whisper-large-v3") == "Whisper-Large-v3"
        assert provider.resolve_model("Whisper-Large-v3") == "Whisper-Large-v3"
        assert provider.resolve_model("custom-model") == "custom-model"

    @patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.sambanova.requests.post")
    def test_correct_transcript_uses_chat_endpoint(self, mock_post):
        mock_post.return_value = FakeResponse(
            payload={"choices": [{"message": {"content": "Corrected transcript."}}]}
        )

        from sapat.providers.sambanova import SambaNovaProvider

        provider = SambaNovaProvider()
        result = provider.correct_transcript("raw transcript", temperature=0.1)

        assert result == "Corrected transcript."
        _, kwargs = mock_post.call_args
        assert (
            mock_post.call_args.args[0]
            == "https://api.sambanova.ai/v1/chat/completions"
        )
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["json"]["model"] == "Meta-Llama-3.3-70B-Instruct"
        assert kwargs["json"]["temperature"] == 0.1
        assert kwargs["json"]["messages"][1]["content"] == "raw transcript"

    @patch.dict(
        os.environ,
        {
            "SAMBANOVA_API_KEY": "test-key",
            "SAMBANOVA_CHAT_MODEL": "Meta-Llama-3.1-8B-Instruct",
        },
        clear=False,
    )
    @patch("sapat.providers.sambanova.requests.post")
    def test_correct_transcript_custom_chat_model(self, mock_post):
        mock_post.return_value = FakeResponse(
            payload={"choices": [{"message": {"content": "Custom model corrected."}}]}
        )

        from sapat.providers.sambanova import SambaNovaProvider

        provider = SambaNovaProvider()
        result = provider.correct_transcript("raw text")

        assert result == "Custom model corrected."
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "Meta-Llama-3.1-8B-Instruct"

    @patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-key"}, clear=False)
    @patch("sapat.providers.sambanova.requests.post")
    def test_correct_transcript_raises_on_api_error(self, mock_post):
        mock_post.return_value = FakeResponse(status_code=500, text="server error")

        from sapat.providers.sambanova import SambaNovaProvider

        provider = SambaNovaProvider()
        with pytest.raises(RuntimeError, match="500"):
            provider.correct_transcript("raw text")

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.sambanova import SambaNovaProvider

        assert SambaNovaProvider.is_available() is False

    @patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-key"}, clear=False)
    def test_available_with_key(self):
        from sapat.providers.sambanova import SambaNovaProvider

        assert SambaNovaProvider.is_available() is True

    @patch.dict(os.environ, {"SAMBANOVA_API_KEY": "bad-key"}, clear=False)
    @patch("sapat.providers.openai_compat.requests.post")
    def test_raises_on_transcription_api_error(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")

        from sapat.providers.sambanova import SambaNovaProvider

        provider = SambaNovaProvider()
        with pytest.raises(RuntimeError, match="401"):
            provider.transcribe(audio_file, model="Whisper-Large-v3")

    @patch.dict(os.environ, {"SAMBANOVA_API_KEY": "test-key"}, clear=False)
    def test_auto_discovery_registers_sambanova(self):
        import sapat.providers as reg
        from sapat.providers import register
        from sapat.providers.sambanova import SambaNovaProvider

        reg._registry.clear()
        reg._discovered = False
        register(SambaNovaProvider)
        providers = get_available_providers()
        assert "sambanova" in providers
