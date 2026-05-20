"""
AffectiveLonging — unified emotional extension engine.

Integrates three modules:
1. Memory triggers (embedding similarity → longing boost)
2. Relationship state machine (HMM + OU → stage transitions)
3. AI self-emotion (VAD model → emotional state)

Usage:
    engine = AffectiveLonging()

    # Store memories
    engine.remember("你喜欢下雨天", tags=["weather"])

    # Main loop
    result = engine.tick(context="今天下雨了")
    if result.should_send:
        send(result.prompt)

    # Observe events
    engine.observe("reply_fast")

    # Time passes
    engine.step_time(hours=12)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from revive_my_lover import PoissonLove

from .emotion import EmotionEngine, EmotionalState
from .memory import MemoryStore, TriggerEngine
from .relationship import Event, RelationshipStateMachine, RelationshipState, Stage

logger = logging.getLogger(__name__)


@dataclass
class AffectiveResult:
    """Full result of an AffectiveLonging tick."""

    # Base decision
    should_send: bool
    base_probability: float

    # Memory trigger
    memory_trigger: str | None = None
    memory_similarity: float = 0.0
    longing_boost: float = 0.0
    boosted_probability: float = 0.0

    # Relationship
    relationship_stage: Stage = Stage.COURTING
    intimacy: float = 0.5
    conflict: float = 0.1

    # Emotion
    emotional_state: EmotionalState | None = None

    # Prompt
    prompt: str = ""
    reason: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


class AffectiveLonging:
    """
    Unified emotional engagement engine.

    Combines three layers:
    1. Memory — past memories trigger present longing
    2. Relationship — 6-stage lifecycle with OU decay
    3. Emotion — VAD model for AI's internal state

    Args:
        memory_persist_dir: Where to store memory embeddings.
        relationship_seed: Seed for relationship state machine.
        emotion_seed: Seed for emotion engine.
        **kwargs: Passed to PoissonLove (config, seed, etc.)

    Example:
        >>> engine = AffectiveLonging()
        >>> engine.remember("你喜欢下雨天")
        >>> engine.observe("reply_fast")
        >>> result = engine.tick(context="今天下雨了")
        >>> result.emotional_state
        EmotionalState(😊 joy, V=+0.45, A=0.52, D=0.55)
    """

    def __init__(
        self,
        memory_persist_dir: str = "./companion_memory_db",
        relationship_seed: int | None = None,
        emotion_seed: int | None = None,
        **kwargs,
    ):
        # Base engine (revive-companion)
        self._base = PoissonLove(**kwargs)

        # Memory layer
        self.memory = MemoryStore(persist_dir=memory_persist_dir)
        self.trigger = TriggerEngine(self.memory)

        # Relationship layer
        self.relationship = RelationshipStateMachine(seed=relationship_seed)

        # Emotion layer
        self.emotion = EmotionEngine(seed=emotion_seed)

        logger.info("AffectiveLonging initialized with 3 layers")

    # ── Memory API ──

    def remember(
        self, text: str, tags: list[str] | None = None, **metadata
    ) -> str:
        """
        Store a memory for future triggering.

        Args:
            text: The memory content (e.g., "你喜欢下雨天").
            tags: Optional tags for filtering.
            **metadata: Additional metadata.

        Returns:
            Memory ID.
        """
        meta = dict(metadata)
        if tags:
            meta["tags"] = ",".join(tags)
        return self.memory.add(text, metadata=meta)

    # ── Event API ──

    def observe(self, event: str) -> None:
        """
        Observe an event and update relationship + emotion.

        Maps string event names to:
        - Relationship Event enum
        - Emotion engine bumps

        Args:
            event: One of: reply_fast, reply_slow, no_reply, long_silence,
                   affection, fight, apology, initiate, reject, long_message
        """
        # Update relationship state
        try:
            rel_event = Event(event)
            self.relationship.observe(rel_event)
        except ValueError:
            logger.warning("Unknown relationship event: %s", event)

        # Update emotion engine
        self.emotion.observe(event)

        # Sync emotion baselines with relationship stage
        self.emotion.update_relationship_stage(self.relationship.stage)

        logger.debug(
            "Observed event=%s, stage=%s, emotion=%s",
            event,
            self.relationship.stage.value,
            self.emotion.current_state.emotion.value,
        )

    def step_time(self, hours: float = 1.0) -> None:
        """
        Let time pass — decays relationship intimacy/conflict and emotion.

        Args:
            hours: Hours to advance.
        """
        self.relationship.step_time(hours)
        self.emotion.step_time(hours)

        # Keep emotion baselines in sync
        self.emotion.update_relationship_stage(self.relationship.stage)

    # ── Tick API ──

    def tick(
        self,
        now: datetime | None = None,
        context: str = "",
    ) -> AffectiveResult:
        """
        Run one tick of the full pipeline.

        Flow:
        1. Base Poisson tick (timing decision)
        2. Memory trigger (if context provided)
        3. Package result with relationship + emotion state

        Args:
            now: Current time.
            context: Situation text for memory matching.

        Returns:
            AffectiveResult with all layers' state.
        """
        # 1. Base engine
        base_result = self._base.tick(now=now)

        # 2. Memory trigger
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
                boosted_prob = self.trigger.boost_longing(
                    base_result.probability, context
                )

        # 3. Build result
        rel_state = self.relationship.current_state
        emo_state = self.emotion.current_state

        return AffectiveResult(
            should_send=base_result.should_send,
            base_probability=base_result.probability,
            memory_trigger=trigger_text,
            memory_similarity=trigger_sim,
            longing_boost=boost,
            boosted_probability=boosted_prob,
            relationship_stage=rel_state.stage,
            intimacy=rel_state.intimacy,
            conflict=rel_state.conflict,
            emotional_state=emo_state,
            prompt=base_result.prompt,
            reason=base_result.reason,
        )

    # ── Pass-through API ──

    def record_reply(self, **kwargs) -> None:
        """Record user reply — passes to base engine."""
        self._base.record_reply(**kwargs)

    def record_send(self, **kwargs) -> None:
        """Record that we sent — passes to base engine."""
        self._base.record_send(**kwargs)

    # ── State API ──

    def get_state(self) -> dict:
        """Get full system state snapshot."""
        rel = self.relationship.current_state
        emo = self.emotion.current_state
        return {
            "relationship": {
                "stage": rel.stage.value,
                "stage_display": rel.stage.display_name,
                "intimacy": rel.intimacy,
                "conflict": rel.conflict,
                "confidence": rel.confidence,
            },
            "emotion": {
                "emotion": emo.emotion.value,
                "emoji": emo.emoji,
                "valence": emo.valence,
                "arousal": emo.arousal,
                "dominance": emo.dominance,
            },
            "memory_count": self.memory.count(),
        }

    def __repr__(self) -> str:
        rel = self.relationship.current_state
        emo = self.emotion.current_state
        return (
            f"AffectiveLonging("
            f"stage={rel.stage.display_name}, "
            f"{emo.emoji} {emo.emotion.value}, "
            f"memories={self.memory.count()})"
        )
