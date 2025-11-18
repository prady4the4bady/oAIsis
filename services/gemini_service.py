"""Gemini API helper utilities for Guidely AI."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from services.google_auth import build_authorized_session

LOGGER = logging.getLogger(__name__)


@dataclass
class GeminiConfig:
    model: str = "gemini-2.0-flash-exp"
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta"
    project_id: Optional[str] = None
    location: str = "us-central1"
    max_image_size: tuple[int, int] = (1280, 720)
    jpeg_quality: int = 85


class GeminiService:
    """Wrapper over Gemini via Vertex AI when available, falling back to API keys."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str | None = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
    ) -> None:
        if not api_key and not project_id:
            raise ValueError(
                "Configure GEMINI_API_KEY or set GOOGLE_PROJECT_ID plus service-account credentials."
            )

        self.config = GeminiConfig(
            model=model or GeminiConfig.model,
            project_id=project_id,
            location=location,
        )
        self._api_key = api_key
        self._session = None
        self._use_vertex = False

        if project_id:
            self._session = build_authorized_session()
            if self._session:
                self._use_vertex = True
            elif not api_key:
                raise ValueError(
                    "Google Vertex requested but service-account credentials are missing."
                )
            else:
                LOGGER.warning(
                    "Vertex credentials missing; falling back to API-key Gemini endpoint."
                )

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

        response = self._post(generate_path="generateContent", payload=payload)
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

    def _post(self, generate_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._use_vertex:
            url = (
                f"https://{self.config.location}-aiplatform.googleapis.com/v1/projects/"
                f"{self.config.project_id}/locations/{self.config.location}/publishers/google/models/"
                f"{self.config.model}:{generate_path}"
            )
            try:
                assert self._session is not None
                resp = self._session.post(url, json=payload, timeout=45)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as err:
                detail = err.response.text if err.response else str(err)
                LOGGER.exception("Vertex Gemini HTTP error: %s", detail)
                raise RuntimeError(f"Vertex Gemini request failed: {detail}") from err
            except requests.RequestException as err:
                LOGGER.exception("Vertex Gemini network error")
                raise RuntimeError("Unable to reach Vertex Gemini service") from err

        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY missing for API key mode.")
        url = (
            f"{self.config.endpoint}/models/{self.config.model}:{generate_path}?key={self._api_key}"
        )
        try:
            resp = requests.post(url, json=payload, timeout=45)
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
