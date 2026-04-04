"""Configuration module."""

from pathlib import Path
import os

VECTOR_STORE_LOCATION: str | None = os.environ.get("VECTOR_STORE_LOCATION")
EMBEDDING_MODEL_NAME: str | None = os.environ.get(
    "EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m"
)

REPO_PATH = Path(__file__).parents[2]
