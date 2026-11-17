"""Paths and runtime options; launch commands should run from project root."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
VECTORSTORE = ROOT / "data" / "vectorstore"
ARTIFACTS = ROOT / "artifacts"
REVIEWS_CSV = PROCESSED / "reviews_clean.csv"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "olist_reviews")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
TOP_K = max(1, int(os.getenv("TOP_K", "8")))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.76"))
