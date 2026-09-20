# ABOUTME: Mock-based tests for the Hugging Face Inference provider
# ABOUTME: Tests verify each provider sends correct auth, URL, and payload

import os
from unittest.mock import patch

import pytest

from sapat.providers.base import TranscriptionResult


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


class TestHuggingFaceProvider:
    @patch.dict(os.environ, {"HF_TOKEN": "test-key"}, clear=False)
    @patch("sapat.providers.huggingface.requests.post")
    def test_transcribe_sends_correct_request(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(payload={"text": "hello hf"})

        from sapat.providers.huggingface import HuggingFaceProvider

        provider = HuggingFaceProvider()
        result = provider.transcribe(audio_file, model="openai/whisper-large-v3")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello hf"

        args, kwargs = mock_post.call_args
        assert (
            args[0]
            == "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
        )
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["headers"]["Content-Type"] == "audio/mpeg"
        assert kwargs["data"] == b"fake audio data"

    @patch.dict(os.environ, {"HF_TOKEN": "test-key"}, clear=False)
    def test_default_model(self):
        from sapat.providers.huggingface import HuggingFaceProvider

        assert HuggingFaceProvider.config.default_model == "openai/whisper-large-v3"

    @patch.dict(os.environ, {}, clear=True)
    def test_not_available_without_key(self):
        from sapat.providers.huggingface import HuggingFaceProvider

        assert HuggingFaceProvider.is_available() is False

    @patch.dict(os.environ, {"HF_TOKEN": "test-key"}, clear=False)
    def test_available_with_key(self):
        from sapat.providers.huggingface import HuggingFaceProvider

        assert HuggingFaceProvider.is_available() is True

    @patch.dict(os.environ, {"HF_TOKEN": "bad-key"}, clear=False)
    @patch("sapat.providers.huggingface.requests.post")
    def test_raises_on_api_error(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(status_code=401, text="unauthorized")

        from sapat.providers.huggingface import HuggingFaceProvider

        provider = HuggingFaceProvider()
        with pytest.raises(RuntimeError, match="401"):
            provider.transcribe(audio_file, model="openai/whisper-large-v3")

    @patch.dict(os.environ, {"HF_TOKEN": "test-key"}, clear=False)
    @patch("sapat.providers.huggingface.requests.post")
    def test_raises_on_model_loading_503(self, mock_post, audio_file):
        mock_post.return_value = FakeResponse(
            status_code=503, text='{"error": "model is currently loading"}'
        )

        from sapat.providers.huggingface import HuggingFaceProvider

        provider = HuggingFaceProvider()
        with pytest.raises(RuntimeError, match="503"):
            provider.transcribe(audio_file, model="openai/whisper-large-v3")

    @patch.dict(os.environ, {"HF_TOKEN": "test-key"}, clear=False)
    @patch("sapat.providers.huggingface.requests.post")
    def test_content_type_follows_extension(self, mock_post, tmp_path):
        mock_post.return_value = FakeResponse(payload={"text": "x"})

        from sapat.providers.huggingface import HuggingFaceProvider

        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"wav bytes")
        provider = HuggingFaceProvider()
        provider.transcribe(str(wav), model="openai/whisper-large-v3")

        _, kwargs = mock_post.call_args
        assert kwargs["headers"]["Content-Type"] == "audio/wav"

    @patch.dict(os.environ, {"HF_TOKEN": "test-key"}, clear=False)
    def test_registered_in_registry(self):
        import sapat.providers as reg
        from sapat.providers import register
        from sapat.providers.huggingface import HuggingFaceProvider

        reg._registry.clear()
        reg._discovered = False
        try:
            register(HuggingFaceProvider)
            assert reg._registry["huggingface"] is HuggingFaceProvider
        finally:
            reg._registry.clear()
            reg._discovered = False
