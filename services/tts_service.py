"""Hybrid text-to-speech helper with Edge-first, Google fallback."""

from __future__ import annotations

import asyncio
import io
import logging
import os
from typing import Any, Coroutine, Dict, Optional

try:  # pragma: no cover - optional dependency guard
    import edge_tts  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    edge_tts = None  # type: ignore
from google.cloud import texttospeech as google_texttospeech  # type: ignore

from services.google_auth import load_service_account_credentials

LOGGER = logging.getLogger(__name__)

EDGE_LANGUAGE_VOICE_MAP: Dict[str, str] = {
    "en": "en-US-AriaNeural",
    "ar": "ar-SA-ZariyahNeural",
    "hi": "hi-IN-SwaraNeural",
}

GOOGLE_VOICE_DEFAULTS: Dict[str, Dict[str, str]] = {
    "en": {"language_code": "en-US", "voice": "en-US-Neural2-F"},
    "ar": {"language_code": "ar-XA", "voice": "ar-XA-Wavenet-B"},
    "hi": {"language_code": "hi-IN", "voice": "hi-IN-Neural2-A"},
}


class TTSService:
    def __init__(self) -> None:
        self.rate = os.getenv("EDGE_TTS_RATE", "-10%")
        self.provider = os.getenv("GUIDELY_TTS_PROVIDER", "edge").lower()
        self._google_client: Optional[Any] = None

        if self.provider in {"google", "auto"}:
            self._google_client = self._build_google_client()
        elif self.provider == "edge":
            # Prepare Google fallback lazily when credentials exist
            self._google_client = self._build_google_client()
        else:
            LOGGER.warning("Unknown GUIDELY_TTS_PROVIDER '%s'; defaulting to edge", self.provider)
            self.provider = "edge"
            self._google_client = self._build_google_client()

    def synthesize(self, text: str, language: str) -> bytes:
        errors: list[str] = []

        if self.provider in {"edge", "auto"}:
            try:
                return self._synthesize_edge(text=text, language=language)
            except Exception as exc:  # pragma: no cover - network/runtime errors
                message = f"Edge TTS failed: {exc}"
                LOGGER.warning(message)
                errors.append(message)

        if self._ensure_google_client():
            try:
                return self._synthesize_google(text=text, language=language)
            except Exception as exc:  # pragma: no cover
                message = f"Google TTS failed: {exc}"
                LOGGER.error(message)
                errors.append(message)

        raise RuntimeError("; ".join(errors) or "No TTS providers are configured.")

    def _synthesize_edge(self, text: str, language: str) -> bytes:
        if edge_tts is None:
            raise RuntimeError("edge-tts package is not installed.")
        voice = EDGE_LANGUAGE_VOICE_MAP.get(language, EDGE_LANGUAGE_VOICE_MAP["en"])
        ssml = self._wrap_with_ssml(text)
        return self._run_async(self._edge_stream(ssml=ssml, voice=voice))

    async def _edge_stream(self, ssml: str, voice: str) -> bytes:
        assert edge_tts is not None
        communicator = edge_tts.Communicate(ssml, voice=voice, rate=self.rate)
        buffer = io.BytesIO()
        async for chunk in communicator.stream():
            if chunk.get("type") == "audio":  # type: ignore[union-attr]
                data = chunk.get("data")
                if isinstance(data, (bytes, bytearray)):
                    buffer.write(data)
        return buffer.getvalue()

    def _synthesize_google(self, text: str, language: str) -> bytes:
        assert self._google_client is not None
        voice_info = self._resolve_google_voice(language)
        audio_encoding = self._google_audio_encoding()

        synthesis_input = google_texttospeech.SynthesisInput(text=text)
        voice_params = google_texttospeech.VoiceSelectionParams(
            language_code=voice_info["language_code"],
            name=voice_info["voice"],
        )
        audio_config = google_texttospeech.AudioConfig(
            audio_encoding=audio_encoding,
            speaking_rate=self._google_float_env("GOOGLE_TTS_SPEAKING_RATE", 1.0),
            pitch=self._google_float_env("GOOGLE_TTS_PITCH", 0.0),
        )

        response = self._google_client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )
        return response.audio_content

    def _resolve_google_voice(self, language: str) -> Dict[str, str]:
        base = GOOGLE_VOICE_DEFAULTS.get(language, GOOGLE_VOICE_DEFAULTS["en"]).copy()
        env_override = os.getenv("GOOGLE_TTS_VOICE_OVERRIDE")
        if env_override:
            base["voice"] = env_override
        specific_override = os.getenv(f"GOOGLE_TTS_VOICE_{language.upper()}")
        if specific_override:
            base["voice"] = specific_override
        language_override = os.getenv(f"GOOGLE_TTS_LANGUAGE_CODE_{language.upper()}")
        if language_override:
            base["language_code"] = language_override
        return base

    def _google_audio_encoding(self) -> Any:
        encoding_name = os.getenv("GOOGLE_TTS_AUDIO_ENCODING", "MP3").upper()
        try:
            return google_texttospeech.AudioEncoding[encoding_name]
        except KeyError:
            LOGGER.warning(
                "Unknown GOOGLE_TTS_AUDIO_ENCODING '%s'; falling back to MP3",
                encoding_name,
            )
            return google_texttospeech.AudioEncoding.MP3

    def _google_float_env(self, name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            LOGGER.warning("Invalid value for %s=%s; using %s", name, raw, default)
            return default

    def _run_async(self, coro: Coroutine[Any, Any, bytes]) -> bytes:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            tmp_loop = asyncio.new_event_loop()
            try:
                return tmp_loop.run_until_complete(coro)
            finally:
                tmp_loop.close()

    def _wrap_with_ssml(self, text: str) -> str:
        cleaned = text.strip() or "No guidance available."
        return (
            "<speak><prosody rate='-10%'>"
            "<emphasis level='moderate'>"
            f"{cleaned}"
            "</emphasis></prosody></speak>"
        )

    def _build_google_client(self) -> Optional[Any]:
        creds = load_service_account_credentials()
        if not creds:
            LOGGER.info(
                "Google Text-to-Speech fallback unavailable until service-account credentials are provided."
            )
            return None
        try:
            return google_texttospeech.TextToSpeechClient(credentials=creds)
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Failed to initialize Google TTS client: %s", exc)
            return None

    def _ensure_google_client(self) -> bool:
        if self._google_client:
            return True
        self._google_client = self._build_google_client()
        return self._google_client is not None
