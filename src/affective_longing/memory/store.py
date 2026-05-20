"""
Memory store — save & retrieve conversations by embedding similarity.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Memory:
    """A stored memory entry."""

    text: str
    tags: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)


class MemoryStore:
    """
    Store and search memories by semantic similarity.

    Usage:
        store = MemoryStore()
        store.add("你喜欢下雨天", tags=["weather"])
        store.add("我们第一次见面在咖啡店", tags=["place"])

        hits = store.search("今天下雨了", top_k=1)
        # → [Memory(text="你喜欢下雨天", ...)]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._memories: list[Memory] = []
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy-load sentence transformer."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def add(self, text: str, tags: list[str] = None, metadata: dict = None) -> Memory:
        """Store a memory."""
        model = self._get_model()
        embedding = model.encode(text).tolist()
        mem = Memory(
            text=text,
            tags=tags or [],
            embedding=embedding,
            metadata=metadata or {},
        )
        self._memories.append(mem)
        return mem

    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.3) -> list[tuple[Memory, float]]:
        """
        Search memories by semantic similarity.

        Returns list of (memory, similarity_score) tuples.
        """
        if not self._memories:
            return []

        model = self._get_model()
        import numpy as np

        query_emb = model.encode(query)
        results = []

        for mem in self._memories:
            if mem.embedding is None:
                continue
            sim = np.dot(query_emb, mem.embedding) / (
                np.linalg.norm(query_emb) * np.linalg.norm(mem.embedding)
            )
            if sim >= min_similarity:
                results.append((mem, float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._memories)
