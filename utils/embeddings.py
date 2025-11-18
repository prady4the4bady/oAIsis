"""Utilities for generating embeddings via Gemini."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

import numpy as np
import numpy.typing as npt
import requests

LOGGER = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY missing. Set it in your environment.")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def embed_text(self, text: str) -> Optional[npt.NDArray[np.float32]]:
        if not text:
            return None
        payload: Dict[str, Any] = {
            "model": self.model,
            "content": {
                "parts": [{"text": text[:2000]}],
            },
        }
        url = f"{self.endpoint}/models/{self.model}:embedContent?key={self.api_key}"
        try:
            resp = requests.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            vector = self._extract_vector(data)
            return np.array(vector, dtype=np.float32)
        except requests.HTTPError as err:
            LOGGER.warning("Embedding error: %s", err.response.text if err.response else err)
        except requests.RequestException as err:
            LOGGER.warning("Embedding service unreachable: %s", err)
        except ValueError as err:
            LOGGER.warning("Embedding parse failure: %s", err)
        return None

    def _extract_vector(self, data: Dict[str, Any]) -> List[float]:
        embedding = data.get("embedding")
        if not embedding:
            raise ValueError("No embedding returned")
        values = embedding.get("values")
        if not isinstance(values, list):
            raise ValueError("Embedding values missing")
        values_list = cast(List[Any], values)
        float_values: List[float] = []
        for item in values_list:
            try:
                float_values.append(float(item))
            except (TypeError, ValueError) as err:
                raise ValueError("Invalid embedding value") from err
        return float_values
