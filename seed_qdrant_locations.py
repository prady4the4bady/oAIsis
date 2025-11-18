import os
from typing import TypedDict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

class Location(TypedDict):
    id: int
    name: str
    description: str

# 1. Load env or hardcode your Qdrant config
QDRANT_URL = os.getenv("QDRANT_URL", "https://af8d015a-0fb5-446a-81c7-767126b38284.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "oAIsis")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY not found in environment variables!")

# 2. Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# 3. Make sure collection exists with 384-dim vectors
try:
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"Collection '{QDRANT_COLLECTION}' recreated successfully.")
except Exception as e:
    print(f"Warning: Could not recreate collection: {e}")
    print("Attempting to use existing collection...")

# 4. Load embedding model
print("Loading embedding model (this may take a moment)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully.")

# 5. Define sample locations
locations: list[Location] = [
    {
        "id": 1,
        "name": "Kitchen counter",
        "description": "A bright kitchen with a sink, cabinets, and a countertop.",
    },
    {
        "id": 2,
        "name": "Bedroom door",
        "description": "A bedroom entrance with a bed visible and a bedside table.",
    },
    {
        "id": 3,
        "name": "Study desk",
        "description": "A desk with a laptop, books, and a chair for studying.",
    },
    {
        "id": 4,
        "name": "Front hallway",
        "description": "A hallway with a shoe rack, coat hanger, and main door.",
    },
    {
        "id": 5,
        "name": "Living room sofa",
        "description": "A living room with a comfortable sofa, coffee table, and TV.",
    },
    {
        "id": 6,
        "name": "Bathroom sink",
        "description": "A bathroom with a sink, mirror, towels, and toiletries.",
    },
]

# 6. Embed descriptions
print(f"Embedding {len(locations)} location descriptions...")
texts = [loc["description"] for loc in locations]
embeddings: list[list[float]] = model.encode(texts, normalize_embeddings=True).tolist()  # type: ignore

# 7. Build PointStructs for Qdrant
points: list[PointStruct] = []
for loc, vector in zip(locations, embeddings):
    points.append(
        PointStruct(
            id=loc["id"],
            vector=vector,
            payload={
                "name": loc["name"],
                "description": loc["description"],
                "type": "seed_location",
            },
        )
    )

# 8. Upsert into Qdrant
print(f"Upserting {len(points)} points into collection '{QDRANT_COLLECTION}'...")
operation_info = client.upsert(
    collection_name=QDRANT_COLLECTION,
    points=points,
)

print("\n✅ Upsert result:", operation_info)
print(f"✅ Successfully seeded {len(points)} locations into collection '{QDRANT_COLLECTION}'.")
print("\nYou can now verify these locations in Qdrant Cloud UI or by running your Streamlit app with memory enabled.")
