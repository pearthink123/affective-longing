"""
MemoryStore — ChromaDB-backed conversation memory with embedding similarity search.
"""
from __future__ import annotations
import time
from typing import Any, Optional

import chromadb
from chromadb.utils import embedding_functions


class MemoryStore:
    """
    Store and search conversation memories by semantic similarity.

    Usage:
        store = MemoryStore()
        store.add("你喜欢下雨天", metadata={"type": "preference"})
        hits = store.query("今天下雨了", top_k=1)
    """

    def __init__(
        self,
        collection_name: str = "companion_memories",
        persist_dir: str = "./companion_memory_db",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, text: str, metadata: Optional[dict[str, Any]] = None) -> str:
        """Add a memory. Returns memory_id."""
        if metadata is None:
            metadata = {}

        metadata["timestamp"] = time.time()
        memory_id = f"mem_{int(time.time() * 1000)}"

        self.collection.add(documents=[text], metadatas=[metadata], ids=[memory_id])
        return memory_id

    def delete_by_id(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self.collection.delete(ids=[memory_id])

    def delete_by_ids(self, memory_ids: list[str]) -> None:
        """Delete multiple memories by IDs."""
        self.collection.delete(ids=memory_ids)

    def clear(self) -> None:
        """Delete all memories. Use with caution!"""
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)

    def get_all(self) -> list[dict]:
        """Get all stored memories."""
        results = self.collection.get(include=["documents", "metadatas"])
        return [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]

    def query(
        self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.65
    ) -> list[dict]:
        """
        Search similar memories.

        Returns list of {text, metadata, similarity} dicts above threshold.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            similarity = 1 - dist  # Chroma returns distance, convert to similarity
            if similarity >= similarity_threshold:
                memories.append({"text": doc, "metadata": meta, "similarity": similarity})
        return memories

    def count(self) -> int:
        """Number of stored memories."""
        return self.collection.count()
