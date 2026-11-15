"""rag_engine/config.py : paramètres partagés (surchargeables par variables d'environnement)."""
import os
from pathlib import Path

try:                                    # lit un fichier .env à la racine du projet s'il existe
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent

DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", ROOT / "docs"))            # PDF des guidelines ESC
CHUNKS_PATH = Path(os.getenv("RAG_CHUNKS", ROOT / "data" / "chunks.jsonl"))

# Qdrant : serveur (service 2, Docker) ou mode embarqué (un dossier local, sans Docker)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.getenv("QDRANT_PATH", "")                           # ex. rag_engine/data/qdrant_local
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")                     # Qdrant Cloud : clé API (jamais dans le code ni dans git)
COLLECTION = os.getenv("QDRANT_COLLECTION", "esc_guidelines")

# Embeddings : multilingue (FR <-> EN), petit (384 dim), tourne sur CPU
EMBEDDER = os.getenv("EMBEDDER", "st")                               # "st" (sentence-transformers) | "hash" (test hors ligne)
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")

# Chunking
MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "220"))                 # ~300 tokens (limite e5 : 512)
OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))
MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "30"))
