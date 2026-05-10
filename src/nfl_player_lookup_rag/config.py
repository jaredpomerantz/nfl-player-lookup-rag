"""Configuration module."""

from pathlib import Path
import os

VECTOR_STORE_LOCATION: str | None = os.environ.get("VECTOR_STORE_LOCATION")

REPO_PATH = Path(__file__).parents[2]

MODEL_LOCATION = Path(REPO_PATH / "resources" / "all-MiniLM-L6-v2.pt")
