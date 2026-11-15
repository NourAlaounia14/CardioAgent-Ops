"""rag_engine/embedder.py : transforme du texte en vecteurs (« empreintes de sens »).

- STEmbedder  : sentence-transformers (par défaut intfloat/multilingual-e5-small).
                Une question en français retrouve un passage en anglais.
- HashEmbedder: bag-of-words haché, SANS téléchargement. Uniquement pour tester le pipeline hors ligne
                (recherche par mots-clés, pas de vraie compréhension du sens).
"""
from __future__ import annotations

import re
import unicodedata
import zlib
from functools import lru_cache

import numpy as np

from rag_engine import config


class STEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.e5 = "e5" in model_name.lower()       # les modèles e5 attendent des préfixes

    def _enc(self, texts, prefix):
        texts = [prefix + t for t in texts] if self.e5 else list(texts)
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float32)

    def embed_passages(self, texts):
        return self._enc(texts, "passage: ")

    def embed_query(self, text):
        return self._enc([text], "query: ")[0]


class HashEmbedder:
    dim = 512

    @staticmethod
    def _tokens(text):
        text = unicodedata.normalize("NFKD", text.lower())
        text = "".join(c for c in text if not unicodedata.combining(c))
        return re.findall(r"[a-z0-9]{3,}", text)

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokens(text):
            h = zlib.crc32(tok.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            # hachage signé : limite l'effet des collisions
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@lru_cache(maxsize=1)
def get_embedder():
    if config.EMBEDDER == "hash":
        return HashEmbedder()
    return STEmbedder(config.EMBED_MODEL)
