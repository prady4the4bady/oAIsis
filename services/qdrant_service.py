"""Qdrant vector memory helper."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import numpy as np
import numpy.typing as npt
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

LOGGER = logging.getLogger(__name__)

VectorArray = npt.NDArray[np.floating[Any]]


@dataclass
class SceneMatch:
    location_name: str
    score: float
    payload: Dict[str, Any]


class QdrantMemory:
    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        collection_name: str,
        vector_size: Optional[int] = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.collection_name = collection_name
        self.threshold = similarity_threshold
        self.vector_size = vector_size or self._resolve_vector_size()
        self.client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection()

    def _resolve_vector_size(self) -> int:
        env_value = os.getenv("QDRANT_VECTOR_SIZE")
        if env_value:
            try:
                parsed = int(env_value)
                if parsed > 0:
                    return parsed
            except ValueError:
                LOGGER.warning(
                    "Invalid QDRANT_VECTOR_SIZE '%s'; defaulting to 3072.", env_value
                )
        return 3072

    def _ensure_collection(self) -> None:
        try:
            description = self.client.get_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            LOGGER.info("Creating Qdrant collection %s", self.collection_name)
            self._create_collection()
            return

        current_size = self._extract_vector_size(description)
        if current_size and current_size != self.vector_size:
            LOGGER.warning(
                "Qdrant collection %s has vector size %s but %s is required; recreating.",
                self.collection_name,
                current_size,
                self.vector_size,
            )
            self._create_collection()

    def _extract_vector_size(
        self,
        description: rest.CollectionDescription | rest.CollectionInfo,
    ) -> Optional[int]:
        config = getattr(description, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if isinstance(vectors, rest.VectorParams):
            return vectors.size
        if isinstance(vectors, dict):
            vector_map = cast(Dict[str, rest.VectorParams], vectors)
            first = next(iter(vector_map.values()), None)
            if isinstance(first, rest.VectorParams):
                return first.size
        return None

    def _create_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(
                size=self.vector_size,
                distance=rest.Distance.COSINE,
            ),
        )

    def save_scene(
        self,
        vector: VectorArray,
        location_name: str,
        description: str,
    ) -> bool:
        if vector.shape[-1] != self.vector_size:
            LOGGER.error(
                "Vector length %s does not match configured Qdrant size %s",
                vector.shape[-1],
                self.vector_size,
            )
            return False
        payload = {
            "location": location_name,
            "description": description,
        }
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    rest.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector.astype(float).tolist(),
                        payload=payload,
                    )
                ],
            )
            return True
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to save scene to Qdrant: %s", exc)
            return False

    def search_scene(self, vector: VectorArray) -> Optional[SceneMatch]:
        if vector.shape[-1] != self.vector_size:
            LOGGER.error(
                "Vector length %s does not match configured Qdrant size %s",
                vector.shape[-1],
                self.vector_size,
            )
            return None
        try:
            result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector.astype(float).tolist(),
                limit=1,
                with_payload=True,
                score_threshold=self.threshold,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Qdrant search failed: %s", exc)
            return None

        if not result:
            return None

        top = result[0]
        payload = top.payload or {}
        location_name = payload.get("location") or payload.get("location_name")
        if not location_name:
            return None

        return SceneMatch(
            location_name=location_name,
            score=top.score or 0.0,
            payload=payload,
        )
