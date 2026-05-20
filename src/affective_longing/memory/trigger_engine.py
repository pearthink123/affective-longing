"""
TriggerEngine — compute memory-triggered longing boost.
"""
from __future__ import annotations
import time
from typing import Any

import numpy as np

from .memory_store import MemoryStore


class TriggerEngine:
    """
    Evaluates current context against stored memories to boost longing.

    Usage:
        store = MemoryStore()
        trigger = TriggerEngine(store)
        boosted = trigger.boost_longing(base_prob=0.15, context="今天下雨了")
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        similarity_threshold: float = 0.68,
        boost_factor: float = 2.5,
    ):
        self.memory = memory_store
        self.similarity_threshold = similarity_threshold
        self.boost_factor = boost_factor

    def compute_trigger_score(self, current_context: str) -> dict[str, Any]:
        """
        Check if current context triggers any memories.

        Returns {trigger_score, memories, max_similarity}.
        """
        memories = self.memory.query(
            current_context, top_k=3, similarity_threshold=self.similarity_threshold
        )

        if not memories:
            return {"trigger_score": 0.0, "memories": [], "max_similarity": 0.0}

        max_sim = max(m["similarity"] for m in memories)

        # Weighted score: recent memories get slight boost
        now = time.time()
        weighted_sims = []
        for m in memories:
            age_days = (now - m["metadata"].get("timestamp", now)) / 86400
            recency_weight = 1.0 + 0.3 * max(0, 1 - age_days / 30)  # Boost within 30 days
            weighted_sims.append(m["similarity"] * recency_weight)

        weighted_score = float(np.mean(weighted_sims))
        trigger_score = min(1.0, weighted_score * self.boost_factor)

        return {
            "trigger_score": trigger_score,
            "memories": memories[:2],
            "max_similarity": max_sim,
        }

    def boost_longing(self, base_longing_prob: float, context: str) -> float:
        """
        Core method: boost base longing probability with memory triggers.

        Args:
            base_longing_prob: Current longing from Poisson/Bayesian.
            context: Current situation text for memory matching.

        Returns:
            Boosted probability (capped at 1.0).
        """
        trigger = self.compute_trigger_score(context)
        boosted = base_longing_prob * (1 + trigger["trigger_score"])
        return min(1.0, boosted)
