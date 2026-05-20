"""
AffectiveLonging — emotional extension engine.

Wraps revive-companion.PoissonLove and adds:
- Memory-triggered longing boosts
- Relationship state awareness
- AI self-emotion modeling
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from revive_my_lover import PoissonLove
from revive_my_lover.core.models import TickResult

from .memory import MemoryStore


@dataclass
class AffectiveResult:
    """Result of an AffectiveLonging tick."""

    # Inherited from base
    should_send: bool
    base_result: TickResult

    # New fields
    memory_trigger: Optional[str] = None  # Which memory triggered, if any
    memory_similarity: float = 0.0
    longing_boost: float = 0.0  # Extra longing from memory trigger


class AffectiveLonging:
    """
    Emotional engagement engine — extends PoissonLove with memory triggers.

    Usage:
        engine = AffectiveLonging()
        engine.remember("你喜欢下雨天", tags=["weather"])

        result = engine.tick(context="今天下雨了")
        if result.should_send:
            send(result.prompt)
    """

    def __init__(self, **kwargs):
        self._base = PoissonLove(**kwargs)
        self.memory = MemoryStore()
        self._memory_boost_threshold = 0.6  # Min similarity to trigger boost
        self._memory_boost_amount = 0.15  # How much to boost probability

    def remember(self, text: str, tags: list[str] = None) -> None:
        """Store a memory for future triggering."""
        self.memory.add(text, tags=tags)

    def tick(
        self, now: Optional[datetime] = None, context: str = ""
    ) -> AffectiveResult:
        """
        Run one tick with optional context for memory triggering.

        Args:
            now: Current time.
            context: Current situation (e.g., "今天下雨了") for memory matching.
        """
        # Check memory triggers first
        trigger_text = None
        trigger_sim = 0.0
        boost = 0.0

        if context and len(self.memory) > 0:
            hits = self.memory.search(context, top_k=1, min_similarity=self._memory_boost_threshold)
            if hits:
                trigger_text = hits[0][0].text
                trigger_sim = hits[0][1]
                boost = self._memory_boost_amount * trigger_sim

        # Run base engine (could inject boost into lambda here)
        base_result = self._base.tick(now=now)

        return AffectiveResult(
            should_send=base_result.should_send,
            base_result=base_result,
            memory_trigger=trigger_text,
            memory_similarity=trigger_sim,
            longing_boost=boost,
        )

    def record_reply(self, **kwargs) -> None:
        """Pass through to base engine."""
        self._base.record_reply(**kwargs)

    def record_send(self, **kwargs) -> None:
        """Pass through to base engine."""
        self._base.record_send(**kwargs)
