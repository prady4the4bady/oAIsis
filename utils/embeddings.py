"""Embedding utilities with local sentence-transformers before Gemini fallback."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, cast

import numpy as np
import numpy.typing as npt
import requests

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

from services.google_auth import build_authorized_session

LOGGER = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(
        self,
        api_key: Optional[str],
        model: str = "text-embedding-004",
        endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
        project_id: Optional[str] = None,
        location: str = "us-central1",
        local_model: Optional[str] = None,
        local_device: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.project_id = project_id
        self.location = location
        self._session = None
        self._use_vertex = False
        self.provider_mode = os.getenv("GUIDELY_EMBEDDING_PROVIDER", "local-first").lower()
        self.local_model_name = local_model or os.getenv(
            "GUIDELY_LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.local_device = local_device or os.getenv("GUIDELY_LOCAL_EMBED_DEVICE", "cpu")
        self._local_model: Any | None = None
        self.output_dimension = self._resolve_output_dimension()

        if project_id:
            self._session = build_authorized_session()
            if self._session:
                self._use_vertex = True
            elif not api_key:
                raise ValueError(
                    "Vertex embeddings requested but Google credentials are missing."
                )
            else:
                LOGGER.warning(
                    "Vertex credentials missing; falling back to API-key embedding endpoint."
                )
        self._remote_enabled = bool(api_key or self._use_vertex)
        if self._remote_enabled and not api_key and not self._use_vertex:
            raise ValueError("Embedding fallback requires GEMINI_API_KEY or Vertex credentials.")

        if not self._remote_enabled and SentenceTransformer is None:
            raise ValueError(
                "Install sentence-transformers or configure Gemini credentials for embeddings."
            )

    def embed_text(self, text: str) -> Optional[npt.NDArray[np.float32]]:
        if not text:
            return None
        provider_order = self._provider_sequence()
        errors: List[str] = []
        for provider in provider_order:
            if provider == "local" and SentenceTransformer is not None:
                try:
                    return self._embed_local(text)
                except Exception as exc:  # pragma: no cover - model runtime errors
                    errors.append(f"local embedding failed: {exc}")
                    LOGGER.warning(errors[-1])
            if provider == "remote" and self._remote_enabled:
                try:
                    return self._embed_remote(text)
                except Exception as exc:
                    errors.append(f"remote embedding failed: {exc}")
                    LOGGER.warning(errors[-1])
        if not errors:
            if SentenceTransformer is None:
                errors.append("sentence-transformers not installed")
            if not self._remote_enabled:
                errors.append("Gemini embeddings not configured")
        raise RuntimeError("; ".join(errors))

    def _provider_sequence(self) -> List[str]:
        if self.provider_mode == "remote-first":
            return ["remote", "local"]
        if self.provider_mode in {"google-only", "remote-only"}:
            return ["remote"]
        if self.provider_mode == "local-only":
            return ["local"]
        return ["local", "remote"]

    def _embed_local(self, text: str) -> npt.NDArray[np.float32]:
        if not self._ensure_local_model():
            raise RuntimeError("Local embedding model unavailable")
        assert self._local_model is not None
        vector = self._local_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self._normalize_dimension(np.asarray(vector, dtype=np.float32))

    def _ensure_local_model(self) -> bool:
        if self._local_model is not None:
            return True
        if SentenceTransformer is None:
            return False
        try:
            self._local_model = SentenceTransformer(
                self.local_model_name,
                device=self.local_device,
            )
            return True
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Unable to load local embedding model %s: %s", self.local_model_name, exc)
            return False

    def _embed_remote(self, text: str) -> npt.NDArray[np.float32]:
        payload = self._build_payload(text)
        try:
            if self._use_vertex:
                url = (
                    f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/"
                    f"locations/{self.location}/publishers/google/models/{self.model}:predict"
                )
                assert self._session is not None
                resp = self._session.post(url, json=payload, timeout=20)
            else:
                if not self.api_key:
                    raise RuntimeError("GEMINI_API_KEY missing for embedding fallback mode.")
                url = f"{self.endpoint}/models/{self.model}:embedContent?key={self.api_key}"
                resp = requests.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            vector = self._extract_vector(data)
            return self._normalize_dimension(np.array(vector, dtype=np.float32))
        except requests.HTTPError as err:
            message = err.response.text if err.response else str(err)
            raise RuntimeError(f"Embedding error: {message}")
        except requests.RequestException as err:
            raise RuntimeError(f"Embedding service unreachable: {err}")
        except ValueError as err:
            raise RuntimeError(f"Embedding parse failure: {err}")

    def _normalize_dimension(self, vector: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        target = self.output_dimension
        dim = int(vector.shape[-1])
        if dim == target:
            return vector.astype(np.float32, copy=False)
        if dim > target:
            return vector[..., :target].astype(np.float32, copy=False)
        pad = target - dim
        padded = np.pad(vector, (0, pad), mode="constant")
        return padded.astype(np.float32, copy=False)

    def _resolve_output_dimension(self) -> int:
        raw_candidates = [os.getenv("QDRANT_VECTOR_SIZE"), os.getenv("GUIDELY_EMBEDDING_DIM")]
        for raw in raw_candidates:
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError:
                LOGGER.warning("Invalid embedding dimension '%s'", raw)
                continue
            if value > 0:
                return value
        return 384

    def _build_payload(self, text: str) -> Dict[str, Any]:
        trimmed = text[:2000]
        if self._use_vertex:
            return {"instances": [{"content": trimmed}]}
        return {
            "model": self.model,
            "content": {"parts": [{"text": trimmed}]},
        }

    def _extract_vector(self, data: Dict[str, Any]) -> List[float]:
        if self._use_vertex:
            predictions: List[Any] = data.get("predictions") or []
            if not predictions:
                raise ValueError("No predictions returned")
            embeddings = predictions[0].get("embeddings")
            values: Optional[List[float]] = None
            if isinstance(embeddings, list) and embeddings:
                first_embedding: Any = embeddings[0]
                if isinstance(first_embedding, dict):
                    first_embedding_dict = cast(Dict[str, Any], first_embedding)
                    raw_values = first_embedding_dict.get("values")
                    if isinstance(raw_values, list):
                        values = cast(List[float], raw_values)
            if values is None:
                raw_values = predictions[0].get("values")
                if isinstance(raw_values, list):
                    values = cast(List[float], raw_values)
            if values is None:
                raise ValueError("Vertex embedding values missing")
        else:
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
