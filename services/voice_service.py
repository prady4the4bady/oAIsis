"""Voice command transcription helpers."""

from __future__ import annotations

import io

import speech_recognition as sr  # type: ignore


class VoiceCommandService:
    def __init__(self, language: str = "en-US") -> None:
        self.language = language
        self.recognizer = sr.Recognizer()

    def transcribe(self, wav_bytes: bytes) -> str:
        if not wav_bytes:
            return ""
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio = self.recognizer.record(source)
        try:
            return self.recognizer.recognize_google(audio, language=self.language)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:  # pragma: no cover - network errors
            raise RuntimeError("Speech service unavailable") from exc
