"""Multilingual E5 embeddings and persistent cosine similarity search in ChromaDB."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import COLLECTION_NAME, EMBEDDING_MODEL, REVIEWS_CSV, VECTORSTORE


@dataclass
class Evidence:
    code: str
    document_id: str
    text: str
    score: float
    review_score: int
    review_id: str
    order_id: str
    category: str
    purchase_month: str
    is_late: bool


class ReviewStore:
    def __init__(self):
        # Lazy imports keep unit tests and data preparation lightweight.
        import chromadb
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        VECTORSTORE.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(VECTORSTORE))
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def index(self, rebuild: bool = False, batch_size: int = 64, limit: int = 0):
        if not REVIEWS_CSV.exists():
            raise FileNotFoundError("Base processada ausente: execute python scripts/01_prepare.py")
        data = pd.read_csv(REVIEWS_CSV).fillna({"category": "nao_informada", "purchase_month": "nao_informado"})
        if limit > 0:
            data = data.head(limit)
        if rebuild:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
        total = len(data)
        for start in range(0, total, batch_size):
            batch = data.iloc[start:start + batch_size]
            texts = batch["text"].fillna("").astype(str).tolist()
            # E5 expects "passage:" for indexed texts and "query:" for questions.
            vectors = self.encoder.encode(
                ["passage: " + t[:1500] for t in texts],
                normalize_embeddings=True, show_progress_bar=False,
            ).tolist()
            metadata = []
            for row in batch.itertuples(index=False):
                metadata.append({
                    "review_score": int(row.review_score),
                    "review_id": str(row.review_id),
                    "order_id": str(row.order_id),
                    "category": str(row.category),
                    "purchase_month": str(row.purchase_month),
                    "year": int(row.year),
                    "is_late": bool(row.is_late),
                })
            self.collection.upsert(
                ids=batch["document_id"].astype(str).tolist(),
                documents=texts,
                metadatas=metadata,
                embeddings=vectors,
            )
            print(f"Indexados {min(start + batch_size, total):,}/{total:,} comentários", flush=True)
        return self.collection.count()

    def search(self, question: str, top_k: int = 8, score_max: int | None = None,
               category: str | None = None, year: int | None = None,
               late_only: bool = False) -> list[Evidence]:
        if self.collection.count() == 0:
            raise RuntimeError("Índice Chroma vazio. Execute python scripts/02_index.py --rebuild")
        clauses = []
        if score_max is not None:
            clauses.append({"review_score": {"$lte": int(score_max)}})
        if category and category != "Todas":
            clauses.append({"category": {"$eq": category}})
        if year:
            clauses.append({"year": {"$eq": int(year)}})
        if late_only:
            clauses.append({"is_late": {"$eq": True}})
        where = None
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}
        vector = self.encoder.encode(["query: " + question], normalize_embeddings=True).tolist()
        found = self.collection.query(
            query_embeddings=vector, n_results=min(top_k, self.collection.count()),
            where=where, include=["documents", "metadatas", "distances"],
        )
        evidences = []
        for i, doc_id in enumerate(found["ids"][0]):
            meta = found["metadatas"][0][i]
            distance = float(found["distances"][0][i])
            evidences.append(Evidence(
                code=f"E{i + 1}", document_id=doc_id,
                text=str(found["documents"][0][i]), score=1 - distance,
                review_score=int(meta["review_score"]), review_id=str(meta["review_id"]),
                order_id=str(meta["order_id"]), category=str(meta["category"]),
                purchase_month=str(meta["purchase_month"]),
                is_late=bool(meta["is_late"]),
            ))
        return evidences
