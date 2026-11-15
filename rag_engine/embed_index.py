"""rag_engine/embed_index.py : chunks.jsonl -> embeddings -> collection Qdrant.

Relancer le script est sans danger : l'identifiant d'un point est déterministe (uuid5 du chunk_id),
donc un chunk existant est mis à jour et non dupliqué.

Usage :
    python -m rag_engine.embed_index                 # serveur Qdrant (QDRANT_URL)
    QDRANT_PATH=rag_engine/data/qdrant_local python -m rag_engine.embed_index   # sans Docker
    python -m rag_engine.embed_index --recreate      # repart d'une collection vide
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from rag_engine import config
from rag_engine.embedder import get_embedder


def get_client() -> QdrantClient:
    if config.QDRANT_PATH:
        return QdrantClient(path=config.QDRANT_PATH)
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None, timeout=30)


def embed_text(c: dict) -> str:
    """Texte réellement embeddé : le titre de section aide la recherche."""
    return f"{c['heading']}. {c['text']}" if c.get("heading") else c["text"]


def ensure_collection(client, dim: int, recreate: bool):
    exists = client.collection_exists(config.COLLECTION)
    if exists and recreate:
        client.delete_collection(config.COLLECTION)
        exists = False
    if exists:
        size = client.get_collection(config.COLLECTION).config.params.vectors.size
        if size != dim:
            raise SystemExit(f"La collection existante a des vecteurs de taille {size}, "
                             f"le modèle en produit {dim}. Relance avec --recreate.")
    else:
        client.create_collection(config.COLLECTION,
                                 vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        try:
            client.create_payload_index(config.COLLECTION, "source", PayloadSchemaType.KEYWORD)
        except Exception:
            pass          # le mode embarqué n'en a pas besoin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(config.CHUNKS_PATH))
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    chunks = [json.loads(l) for l in Path(a.chunks).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not chunks:
        raise SystemExit("chunks.jsonl est vide : lance d'abord le chunking.")
    emb = get_embedder()
    client = get_client()
    ensure_collection(client, emb.dim, a.recreate)

    for i in range(0, len(chunks), a.batch):
        batch = chunks[i:i + a.batch]
        vecs = emb.embed_passages([embed_text(c) for c in batch])
        client.upsert(config.COLLECTION, points=[
            PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])), vector=v.tolist(), payload=c)
            for c, v in zip(batch, vecs)])
        print(f"  indexé {min(i + a.batch, len(chunks))}/{len(chunks)}")
    n = client.count(config.COLLECTION, exact=True).count
    print(f"Collection '{config.COLLECTION}' : {n} points (dim {emb.dim}, cosinus)")


if __name__ == "__main__":
    main()
