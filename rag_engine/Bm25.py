"""rag_engine/bm25.py : recherche par mots-clés (BM25 Okapi), sans dépendance externe.

L'index est construit en mémoire à partir de chunks.jsonl (les mêmes chunks que ceux de Qdrant).
    from rag_engine.bm25 import bm25_search
    bm25_search("ETT sténose aortique", k=5)
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rag_engine import config

K1, B = 1.5, 0.75      # réglages standard de BM25

STOPWORDS = set("""
le la les l un une des du de d et ou a au aux en dans sur sous par pour avec sans ce cet cette ces
se sa son ses leur leurs qui que quoi dont ou est sont etre ete avoir ont il elle ils elles on nous
vous je tu ne pas plus ni si mais donc car comme tres aussi peut peuvent fait faire
the of and or to in on for with by is are be as at an it its this that from
""".split())


def normalize(text: str) -> str:
    """Minuscules + suppression des accents : « échocardiographie » == « echocardiographie »."""
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    toks = []
    for w in re.findall(r"[a-z0-9]+", normalize(text)):
        if w in STOPWORDS or (len(w) < 2 and not w.isdigit()):
            continue
        if len(w) > 4 and w.endswith(("s", "x")):      # pluriel simple : souffles -> souffle
            w = w[:-1]
        toks.append(w)
    return toks


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        for i, c in enumerate(chunks):
            toks = tokenize(index_text(c))
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.n = len(chunks)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def scores(self, query: str, allowed: set[int] | None = None) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                if allowed is not None and i not in allowed:
                    continue
                norm = tf + K1 * (1 - B + B * self.doc_len[i] / self.avgdl)
                out[i] += idf * tf * (K1 + 1) / norm
        return out

    def search(self, query: str, k: int = 10, source: str | None = None) -> list[dict]:
        allowed = {i for i, c in enumerate(self.chunks) if c.get("source") == source} if source else None
        ranked = sorted(self.scores(query, allowed).items(), key=lambda x: -x[1])[:k]
        return [dict(score=round(s, 4), **hit_fields(self.chunks[i])) for i, s in ranked]


def index_text(c: dict) -> str:
    """Même texte que celui embeddé dans Qdrant (titre + corps)."""
    return f"{c['heading']}. {c['text']}" if c.get("heading") else c["text"]


def hit_fields(c: dict) -> dict:
    return dict(text=c["text"], heading=c.get("heading", ""), source=c["source"],
                page=c.get("page_start"), chunk_id=c["chunk_id"])


@lru_cache(maxsize=1)
def get_index(path: str | None = None) -> BM25Index:
    p = Path(path or config.CHUNKS_PATH)
    if not p.exists():
        raise FileNotFoundError(f"{p} introuvable : lance d'abord le chunking (BM25 lit chunks.jsonl).")
    chunks = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return BM25Index(chunks)


def bm25_search(query: str, k: int = 10, source: str | None = None) -> list[dict]:
    return get_index().search(query, k, source)