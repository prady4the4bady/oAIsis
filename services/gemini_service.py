"""Gemini API helper utilities for Guidely AI."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    api_key: str
    model: str = "gemini-2.0-flash-exp"
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta"
    max_image_size: tuple[int, int] = (1280, 720)
    jpeg_quality: int = 85


class GeminiService:
    """Tiny wrapper over the Gemini REST APIs."""

    def __init__(self, api_key: str, model: str | None = None) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing. Set it in your environment.")
        self.config = GeminiConfig(api_key=api_key, model=model or GeminiConfig.model)

    def analyze_image(
        self,
        image: Image.Image,
        user_prompt: str,
        response_mime_type: str | None = None,
    ) -> str:
        """Send a combined text + image prompt to Gemini and return the response text."""

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_prompt},
                        self._pil_to_content_part(image),
                    ],
                }
            ]
        }

        if response_mime_type:
            payload["generationConfig"] = {"responseMimeType": response_mime_type}

        response = self._post(
            path=f"/models/{self.config.model}:generateContent",
            json=payload,
        )
        return self._extract_text(response)

    def _pil_to_content_part(self, image: Image.Image) -> Dict[str, Any]:
        resized = image.copy()
        resized.thumbnail(self.config.max_image_size)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=self.config.jpeg_quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": encoded,
            }
        }

    def _post(self, path: str, json: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.config.endpoint}{path}?key={self.config.api_key}"
        try:
            resp = requests.post(url, json=json, timeout=45)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as err:
            detail = err.response.text if err.response else str(err)
            LOGGER.exception("Gemini HTTP error: %s", detail)
            raise RuntimeError(f"Gemini request failed: {detail}") from err
        except requests.RequestException as err:
            LOGGER.exception("Gemini network error")
            raise RuntimeError("Unable to reach Gemini service") from err

    def _extract_text(self, response: Dict[str, Any]) -> str:
        candidates: List[Dict[str, Any]] = response.get("candidates") or []
        for candidate in candidates:
            content: Dict[str, Any] = candidate.get("content") or {}
            parts: List[Dict[str, Any]] = content.get("parts") or []
            for part in parts:
                text_value = part.get("text")
                if isinstance(text_value, str):
                    cleaned = text_value.strip()
                    if cleaned:
                        return cleaned

        raise ValueError("Gemini returned no text content")
