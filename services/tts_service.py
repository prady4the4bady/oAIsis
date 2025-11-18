"""Text-to-speech service with pluggable providers (Edge TTS or Google TTS)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from typing import Any, Coroutine, Dict, Optional, cast

import edge_tts

texttospeech: Any
try:  # pragma: no cover - import guard for optional dependency
    from google.cloud import texttospeech as google_texttospeech  # type: ignore
    texttospeech = cast(Any, google_texttospeech)
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    texttospeech = None  # type: ignore[assignment]

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

DEFAULT_RATE = "-10%"  # approx 0.9 speed


class TTSService:
    def __init__(self, rate: str = DEFAULT_RATE) -> None:
        self.rate = rate
        self.provider = os.getenv("GUIDELY_TTS_PROVIDER", "edge").lower()
        self._google_client: Optional[Any] = None
        if self.provider == "google":
            try:
                self._google_client = self._build_google_client()
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.error(
                    "Failed to initialize Google TTS client: %s", exc, exc_info=True
                )
                self._google_client = None

    def synthesize(self, text: str, language: str) -> bytes:
        if self.provider == "google":
            if not self._google_client:
                raise RuntimeError(
                    "Google TTS selected but credentials were not provided."
                )
            return self._synthesize_google(text=text, language=language)

        voice = EDGE_LANGUAGE_VOICE_MAP.get(language, EDGE_LANGUAGE_VOICE_MAP["en"])
        ssml = self._wrap_with_ssml(text)
        return self._run_async(self._synthesize_async(ssml=ssml, voice=voice))

    async def _synthesize_async(self, ssml: str, voice: str) -> bytes:
        communicator = edge_tts.Communicate(ssml, voice=voice, rate=self.rate)
        buffer = io.BytesIO()
        async for chunk in communicator.stream():
            if chunk.get("type") == "audio":  # type: ignore[union-attr]
                data = chunk.get("data")
                if isinstance(data, (bytes, bytearray)):
                    buffer.write(data)
        return buffer.getvalue()

    def _synthesize_google(self, text: str, language: str) -> bytes:
        if texttospeech is None:
            raise RuntimeError(
                "google-cloud-texttospeech is not installed but Google provider is active."
            )
        assert self._google_client is not None
        voice_info = self._resolve_google_voice(language)
        audio_encoding = self._google_audio_encoding()

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=voice_info["language_code"],
            name=voice_info["voice"],
        )
        audio_config = texttospeech.AudioConfig(
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
        if texttospeech is None:
            raise RuntimeError("Google TTS module is unavailable.")
        encoding_name = os.getenv("GOOGLE_TTS_AUDIO_ENCODING", "MP3").upper()
        try:
            return texttospeech.AudioEncoding[encoding_name]
        except KeyError:
            LOGGER.warning(
                "Unknown GOOGLE_TTS_AUDIO_ENCODING '%s'; falling back to MP3",
                encoding_name,
            )
            return texttospeech.AudioEncoding.MP3

    def _google_float_env(self, name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            LOGGER.warning("Invalid value for %s=%s; using %s", name, raw, default)
            return default

    def _build_google_client(self) -> Optional[Any]:
        if texttospeech is None:
            LOGGER.error(
                "google-cloud-texttospeech is not installed but Google TTS provider is selected."
            )
            return None
        credentials_json = os.getenv("GOOGLE_TTS_CREDENTIALS_JSON")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        try:
            if credentials_json:
                info = json.loads(credentials_json)
                return texttospeech.TextToSpeechClient.from_service_account_info(info)
            if credentials_path:
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(credentials_path)
                return texttospeech.TextToSpeechClient.from_service_account_file(
                    credentials_path
                )
        except json.JSONDecodeError:
            LOGGER.error(
                "Invalid JSON found in GOOGLE_TTS_CREDENTIALS_JSON; Google TTS disabled."
            )
            return None
        except FileNotFoundError:
            LOGGER.error(
                "Google credentials file '%s' could not be found; Google TTS disabled.",
                credentials_path,
            )
            return None

        LOGGER.warning(
            "Google TTS provider selected but no credentials env vars were supplied."
        )
        return None

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
        cleaned = text.strip()
        if not cleaned:
            cleaned = "No guidance available."
        return (
            "<speak><prosody rate='-10%'>"
            "<emphasis level='moderate'>"
            f"{cleaned}"
            "</emphasis>"
            "</prosody></speak>"
        )
