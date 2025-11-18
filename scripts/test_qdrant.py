"""Quick CLI to verify Qdrant connectivity, collection schema, and search flow."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import Any, List, Tuple, cast

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

COLLECTION_ENV = "QDRANT_COLLECTION"
VECTOR_ENV = "QDRANT_VECTOR_SIZE"
URL_ENV = "QDRANT_URL"
KEY_ENV = "QDRANT_API_KEY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test Qdrant health, collection config, and vector similarity",
    )
    parser.add_argument(
        "--collection",
        help="Override collection name (defaults to QDRANT_COLLECTION)",
    )
    parser.add_argument(
        "--vector-size",
        type=int,
        help="Override vector size (defaults to QDRANT_VECTOR_SIZE)",
    )
    parser.add_argument(
        "--skip-insert",
        action="store_true",
        help="Skip inserting/searching a test vector—only check health + schema.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Tuple[str, str, str, int]:
    load_dotenv()
    url = os.getenv(URL_ENV)
    api_key = os.getenv(KEY_ENV)
    collection = args.collection or os.getenv(COLLECTION_ENV)
    vector_size = args.vector_size or os.getenv(VECTOR_ENV)

    if not url or not api_key or not collection or not vector_size:
        missing = [
            name
            for name, value in [
                (URL_ENV, url),
                (KEY_ENV, api_key),
                (COLLECTION_ENV, collection),
                (VECTOR_ENV, vector_size),
            ]
            if not value
        ]
        raise SystemExit(
            "Missing required env vars: " + ", ".join(missing)
        )

    try:
        vector_size_int = int(vector_size)
    except ValueError as exc:  # noqa: BLE001
        raise SystemExit(
            f"QDRANT_VECTOR_SIZE must be an integer, got: {vector_size}"
        ) from exc

    return url, api_key, collection, vector_size_int


def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int,
) -> None:
    try:
        info = client.get_collection(collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()))
        else:
            first = vectors
        size = first.size if first else None
        if size != vector_size:
            print(
                f"Collection '{collection}' uses size {size}; recreating for {vector_size} dims.",
            )
            raise ValueError("vector size mismatch")
        print(f"Collection '{collection}' already exists with correct schema.")
    except Exception:  # noqa: BLE001
        print(f"Creating collection '{collection}' with size {vector_size} and COSINE distance…")
        client.recreate_collection(
            collection_name=collection,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )


def run_insert_search(
    client: QdrantClient,
    collection: str,
    vector_size: int,
) -> None:
    vector = np.random.rand(vector_size).astype(float)
    payload = {
        "location": "guidely_test_point",
        "description": "Temporary smoke-test vector",
    }
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=collection,
        wait=True,
        points=[
            rest.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            )
        ],
    )
    print(f"Inserted test point {point_id} with payload {payload}.")

    search = client.search(
        collection_name=collection,
        query_vector=vector.tolist(),
        limit=1,
        with_payload=True,
    )
    if not search:
        raise RuntimeError("Search returned no results.")
    top = search[0]
    print(
        "Search successful: score={:.4f}, payload={}".format(
            top.score,
            top.payload,
        )
    )

    client.delete(
        collection_name=collection,
        points_selector=rest.PointIdsList(points=[point_id]),
    )
    print("Cleaned up test point.")


def main() -> None:
    args = parse_args()
    url, api_key, collection, vector_size = load_config(args)

    client = QdrantClient(url=url, api_key=api_key)
    try:
        collections_obj = cast(Any, client.get_collections())  # type: ignore[attr-defined]
        maybe_list = getattr(collections_obj, "collections", None)
        count = len(cast(List[Any], maybe_list)) if isinstance(maybe_list, list) else "unknown"
        print(f"Connected to Qdrant cluster (collections: {count}).")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to reach Qdrant cluster: {exc}") from exc

    ensure_collection(client, collection, vector_size)

    if not args.skip_insert:
        run_insert_search(client, collection, vector_size)
    else:
        print("Skip-insert flag set; leaving after schema validation.")

    print("Qdrant workflow test completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Qdrant test failed: {exc}", file=sys.stderr)
        raise
