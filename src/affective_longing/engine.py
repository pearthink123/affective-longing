"""
AffectiveLonging — emotional extension engine.

Wraps revive-companion.PoissonLove and adds:
- Memory-triggered longing boosts
- Relationship state awareness (placeholder)
- AI self-emotion modeling (placeholder)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from revive_my_lover import PoissonLove
from revive_my_lover.core.models import TickResult

from .memory import MemoryStore, TriggerEngine


@dataclass
class AffectiveResult:
    """Result of an AffectiveLonging tick."""

    should_send: bool
    base_result: TickResult

    # Memory trigger fields
    memory_trigger: Optional[str] = None
    memory_similarity: float = 0.0
    longing_boost: float = 0.0
    boosted_probability: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)


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

    def __init__(self, memory_persist_dir: str = "./companion_memory_db", **kwargs):
        self._base = PoissonLove(**kwargs)
        self.memory = MemoryStore(persist_dir=memory_persist_dir)
        self.trigger = TriggerEngine(self.memory)

    def remember(self, text: str, tags: list[str] = None, **metadata) -> str:
        """Store a memory for future triggering. Returns memory_id."""
        meta = dict(metadata)
        if tags:
            meta["tags"] = ",".join(tags)
        return self.memory.add(text, metadata=meta)

    def tick(
        self, now: Optional[datetime] = None, context: str = ""
    ) -> AffectiveResult:
        """
        Run one tick with optional context for memory triggering.

        Args:
            now: Current time.
            context: Current situation for memory matching (e.g., "今天下雨了").
        """
        base_result = self._base.tick(now=now)

        trigger_text = None
        trigger_sim = 0.0
        boost = 0.0
        boosted_prob = base_result.probability

        if context and self.memory.count() > 0:
            trigger_info = self.trigger.compute_trigger_score(context)
            if trigger_info["memories"]:
                trigger_text = trigger_info["memories"][0]["text"]
                trigger_sim = trigger_info["max_similarity"]
                boost = trigger_info["trigger_score"]
                boosted_prob = self.trigger.boost_longing(base_result.probability, context)

        return AffectiveResult(
            should_send=base_result.should_send,
            base_result=base_result,
            memory_trigger=trigger_text,
            memory_similarity=trigger_sim,
            longing_boost=boost,
            boosted_probability=boosted_prob,
        )

    def record_reply(self, **kwargs) -> None:
        """Pass through to base engine."""
        self._base.record_reply(**kwargs)

    def record_send(self, **kwargs) -> None:
        """Pass through to base engine."""
        self._base.record_send(**kwargs)
