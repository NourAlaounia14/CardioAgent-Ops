# rag_engine : base de connaissances ESC (service 2, vector-db)

```
rag_engine/
├── docs/                  # PDF des guidelines ESC (à placer ici)
├── data/                  # généré : chunks.jsonl, rag_eval_results.csv (ne pas versionner)
├── config.py              # paramètres (chunking, modèle, Qdrant)
├── chunking.py            # PDF -> chunks.jsonl
├── embedder.py            # texte -> vecteurs
├── embed_index.py         # chunks -> Qdrant
├── search.py              # recherche utilisée par le KnowledgeRAG-Agent
├── evaluate_rag.py        # Recall@k et MRR (section « évaluation du RAG » du rapport)
└── eval_questions.example.json
```

## Utilisation (depuis la racine du projet)

```bash
pip install -r rag_engine/requirements.txt

# 1. Lancer Qdrant (ou utiliser le mode sans Docker, voir plus bas)
docker run -d --name vector-db -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

# 2. Découper les PDF, puis indexer
python -m rag_engine.chunking
python -m rag_engine.embed_index --recreate

# 3. Tester une recherche
python -m rag_engine.search "souffle systolique, conduite à tenir" -k 3

# 4. Évaluer (après avoir écrit rag_engine/eval_questions.json)
python -m rag_engine.evaluate_rag --questions rag_engine/eval_questions.json --mlflow
```

Sans Docker (Colab, test rapide) : `QDRANT_PATH=rag_engine/data/qdrant_local python -m rag_engine.embed_index`
(la même variable doit être définie pour `search` et `evaluate_rag`).

## Dans docker-compose.yml

```yaml
  vector-db:
    image: qdrant/qdrant
    container_name: vector-db
    ports: ["6333:6333"]
    volumes: [qdrant_data:/qdrant/storage]
    networks: [docker-network]
# volumes: { qdrant_data: {} }
```
Depuis les autres conteneurs : `QDRANT_URL=http://vector-db:6333`.
Pense à fixer un tag de version précis dans le rendu final plutôt que `latest`.

## Utilisation par l'agent

```python
from rag_engine.search import search, format_context
hits = search(f"{classe_audio} {prompt_client}", k=4)
contexte = format_context(hits)      # à insérer dans le prompt du ReasoningLLM-Agent
```
Si `hits` est vide ou si tous les scores sont faibles, l'agent doit le signaler au LLM plutôt que de laisser inventer.

## Choix à justifier dans le rapport

- **Modèle** `intfloat/multilingual-e5-small` : multilingue (question FR, guidelines EN), 384 dimensions, tourne sur CPU.
- **Chunks** de ~220 mots avec chevauchement de 40 : assez petits pour être précis, assez grands pour garder le contexte.
- **Titre de section** ajouté au texte embeddé : améliore la recherche.
- **Cosinus** : standard pour des embeddings normalisés.
- Variables à tester pour l'évaluation : `CHUNK_MAX_WORDS` (150 / 220 / 350), `EMBED_MODEL`, `k`.
