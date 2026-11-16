# ABOUTME: Regression tests for preserving caller-owned audio and transcripts
# ABOUTME: Uses a fake provider; the integration case runs real ffmpeg conversion
import shutil
import subprocess
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import sapat.process as processing
from sapat.providers.base import AudioFormat, TranscriptionResult


@pytest.fixture
def provider():
    p = MagicMock()
    p.config = SimpleNamespace(
        preferred_format=AudioFormat.WAV, max_file_size_mb=25, supports_correction=True
    )
    p.transcribe.return_value = TranscriptionResult(text="new transcript")
    p.correct_transcript.return_value = "corrected"
    return p


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "input.mp3"
    source.write_bytes(b"original source")
    paths = []

    def convert(src, dst, *args):
        paths.append(Path(dst))
        Path(dst).write_bytes(b"converted")

    monkeypatch.setattr(processing, "convert_audio", convert)
    monkeypatch.setattr(processing, "should_split_file", lambda *a, **kw: False)
    monkeypatch.setattr(
        processing, "spinner_context", lambda *a: nullcontext(MagicMock())
    )
    return source, paths


def run(source, provider, correct=False):
    processing.process_file(str(source), provider, "model", "en", None, 0, "L", correct)


@pytest.mark.parametrize("suffix", [".wav", ".mp3"])
def test_preserves_audio(setup, provider, suffix):
    source, paths = setup
    source = source.rename(source.with_suffix(suffix))
    neighbor = source.with_suffix(".wav")
    if neighbor != source:
        neighbor.write_bytes(b"valuable neighbor")
    before = {p: p.read_bytes() for p in {source, neighbor}}
    run(source, provider)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert source.with_suffix(".txt").read_text() == "new transcript"
    assert all(not p.parent.exists() for p in paths)


@pytest.mark.parametrize("failure", ["convert", "transcribe", "correct", "split"])
def test_cleans_temp_on_errors(setup, provider, monkeypatch, failure):
    source, paths = setup
    if failure == "convert":
        original = processing.convert_audio

        def failed(*args):
            original(*args)
            raise RuntimeError("conversion error")

        monkeypatch.setattr(processing, "convert_audio", failed)
    elif failure == "transcribe":
        provider.transcribe.side_effect = RuntimeError("transcribe error")
    elif failure == "correct":
        provider.correct_transcript.side_effect = RuntimeError("correct error")
    else:
        monkeypatch.setattr(processing, "should_split_file", lambda *a, **kw: True)
        monkeypatch.setattr(
            processing,
            "split_audio_file",
            MagicMock(side_effect=RuntimeError("split error")),
        )
    if failure in {"transcribe", "correct"}:
        with pytest.raises(RuntimeError):
            run(source, provider, correct=True)
    else:
        run(source, provider, correct=True)
    assert source.read_bytes() == b"original source"
    assert not source.with_suffix(".txt").exists()
    assert all(not p.parent.exists() for p in paths)


@pytest.mark.parametrize("kind", ["file", "symlink", "dangling", "directory"])
def test_existing_transcript_never_overwritten(setup, provider, kind):
    source, paths = setup
    output = source.with_suffix(".txt")
    target = source.parent / "valuable.txt"
    if kind == "directory":
        output.mkdir()
    elif kind in {"symlink", "dangling"}:
        if kind == "symlink":
            target.write_text("valuable")
        output.symlink_to(target)
    else:
        output.write_text("valuable")
    with pytest.raises(FileExistsError):
        run(source, provider)
    assert not paths
    provider.transcribe.assert_not_called()
    if kind == "file":
        assert output.read_text() == "valuable"
    elif kind == "symlink":
        assert target.read_text() == "valuable"
    elif kind == "dangling":
        assert output.is_symlink() and not target.exists()
    else:
        assert output.is_dir()


def test_transcript_created_during_inference_is_preserved(setup, provider):
    source, paths = setup
    output = source.with_suffix(".txt")

    def transcribe(*a, **kw):
        output.write_text("other invocation")
        return TranscriptionResult(text="new transcript")

    provider.transcribe.side_effect = transcribe
    with pytest.raises(FileExistsError):
        run(source, provider)
    assert output.read_text() == "other invocation"
    assert all(not p.parent.exists() for p in paths)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_real_audio_conversion_preserves_wav_and_neighbor(
    tmp_path, provider, monkeypatch
):
    monkeypatch.setattr(
        processing, "spinner_context", lambda *a: nullcontext(MagicMock())
    )
    source = tmp_path / "real.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\x00\x00\x00\x00" * 4410)
    original = source.read_bytes()
    converted = []

    def transcribe(path, **kw):
        converted.append(Path(path))
        with wave.open(path, "rb") as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            assert wav.getnframes() > 0
        return TranscriptionResult(text="fake inference, real conversion")

    provider.transcribe.side_effect = transcribe
    run(source, provider)
    assert source.read_bytes() == original
    # A differently formatted file with the same stem must not reuse this WAV.
    mp3 = tmp_path / "second.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), str(mp3)], check=True)
    neighbor = mp3.with_suffix(".wav")
    neighbor.write_bytes(b"do not touch")
    run(mp3, provider)
    assert neighbor.read_bytes() == b"do not touch"
    assert all(not p.parent.exists() for p in converted)
