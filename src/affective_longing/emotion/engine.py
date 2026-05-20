"""
Emotion engine — manages the AI companion's emotional state.

Combines:
1. OU processes (one per VAD dimension) for time decay
2. Event-driven bumps (user actions affect emotions)
3. Relationship state influence (cold war → low valence)

Design:
- Each VAD dimension has its own OU process
- Events bump specific dimensions
- Relationship state modulates baselines (e.g., COLD → lower valence baseline)
- Emotions decay toward dynamic baselines over time
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..relationship.ou_process import OUProcess
from ..relationship.stages import Stage
from .vad import Emotion, EmotionalState

logger = logging.getLogger(__name__)


# Event → (valence_bump, arousal_bump, dominance_bump)
EVENT_BUMPS: dict[str, tuple[float, float, float]] = {
    # Positive events
    "reply_fast": (0.08, 0.05, 0.03),
    "affection": (0.12, 0.08, 0.02),
    "initiate": (0.10, 0.06, 0.05),
    "long_message": (0.06, 0.04, 0.01),

    # Negative events
    "no_reply": (-0.06, 0.08, -0.05),
    "reply_slow": (-0.03, 0.02, -0.02),
    "long_silence": (-0.10, 0.10, -0.08),
    "reject": (-0.12, 0.10, -0.10),
    "fight": (-0.15, 0.15, -0.05),

    # Recovery
    "apology": (0.05, -0.05, 0.03),
}

# Relationship stage → baseline adjustments (V, A, D)
STAGE_BASELINES: dict[Stage, tuple[float, float, float]] = {
    Stage.COURTING: (0.2, 0.6, 0.3),    # Slightly positive, anxious, uncertain
    Stage.SWEET: (0.7, 0.5, 0.5),       # Happy, moderate arousal, balanced
    Stage.PASSIONATE: (0.8, 0.7, 0.6),  # Very happy, excited, confident
    Stage.STABLE: (0.5, 0.3, 0.5),      # Content, calm, balanced
    Stage.COLD: (-0.5, 0.5, 0.3),       # Unhappy, tense, low control
    Stage.REPAIRING: (0.0, 0.5, 0.4),   # Neutral, attentive, cautious
}


@dataclass
class EmotionEngineConfig:
    """Configuration for emotion engine."""

    # Default baselines (overridden by relationship stage)
    valence_baseline: float = 0.3
    arousal_baseline: float = 0.4
    dominance_baseline: float = 0.5

    # OU parameters
    valence_reversion: float = 0.06   # Slow decay (emotions are sticky)
    arousal_reversion: float = 0.10   # Faster decay (excitement fades)
    dominance_reversion: float = 0.08

    valence_volatility: float = 0.02
    arousal_volatility: float = 0.03
    dominance_volatility: float = 0.02


class EmotionEngine:
    """
    Manages the AI companion's emotional state.

    Example:
        >>> engine = EmotionEngine(seed=42)
        >>> engine.current_state
        EmotionalState(😐 neutral, V=+0.30, A=0.40, D=0.50)

        >>> engine.observe("affection")
        >>> engine.current_state
        EmotionalState(😊 joy, V=+0.42, A=0.48, D=0.52)

        >>> engine.update_relationship_stage(Stage.COLD)
        >>> engine.step_time(24)
        >>> engine.current_state
        EmotionalState(😢 sadness, V=-0.20, ...)
    """

    def __init__(
        self,
        config: EmotionEngineConfig | None = None,
        seed: int | None = None,
    ):
        self.config = config or EmotionEngineConfig()

        # OU processes for each dimension
        self._valence = OUProcess(
            baseline=self.config.valence_baseline,
            reversion_speed=self.config.valence_reversion,
            volatility=self.config.valence_volatility,
            seed=seed,
        )
        self._arousal = OUProcess(
            baseline=self.config.arousal_baseline,
            reversion_speed=self.config.arousal_reversion,
            volatility=self.config.arousal_volatility,
            seed=(seed + 1) if seed is not None else None,
        )
        self._dominance = OUProcess(
            baseline=self.config.dominance_baseline,
            reversion_speed=self.config.dominance_reversion,
            volatility=self.config.dominance_volatility,
            seed=(seed + 2) if seed is not None else None,
        )

        # Valence is [-1, 1], not [0, 1], so we need special handling
        self._valence.value = self.config.valence_baseline

        self._current_stage: Stage | None = None

        logger.info(
            "EmotionEngine initialized: %s",
            self.current_state,
        )

    @property
    def current_state(self) -> EmotionalState:
        """Current emotional snapshot."""
        return EmotionalState(
            valence=self._valence.value,
            arousal=self._arousal.value,
            dominance=self._dominance.value,
        )

    @property
    def valence(self) -> float:
        return self._valence.value

    @property
    def arousal(self) -> float:
        return self._arousal.value

    @property
    def dominance(self) -> float:
        return self._dominance.value

    def observe(self, event: str) -> EmotionalState:
        """
        React to an event.

        Args:
            event: Event name (e.g., "reply_fast", "fight", "affection")

        Returns:
            New emotional state.
        """
        bumps = EVENT_BUMPS.get(event, (0.0, 0.0, 0.0))
        v_bump, a_bump, d_bump = bumps

        self._valence.bump(v_bump)
        self._arousal.bump(a_bump)
        self._dominance.bump(d_bump)

        logger.debug(
            "Emotion event=%s: bumps=(V%+.2f, A%+.2f, D%+.2f) → %s",
            event,
            v_bump,
            a_bump,
            d_bump,
            self.current_state,
        )

        return self.current_state

    def update_relationship_stage(self, stage: Stage) -> None:
        """
        Update emotional baselines based on relationship stage.

        Call this when the relationship state machine transitions.
        """
        if stage == self._current_stage:
            return

        self._current_stage = stage
        v_base, a_base, d_base = STAGE_BASELINES.get(
            stage,
            (self.config.valence_baseline,
             self.config.arousal_baseline,
             self.config.dominance_baseline),
        )

        self._valence.baseline = v_base
        self._arousal.baseline = a_base
        self._dominance.baseline = d_base

        logger.info(
            "Emotion baselines updated for stage=%s: (V=%.2f, A=%.2f, D=%.2f)",
            stage.value,
            v_base,
            a_base,
            d_base,
        )

    def step_time(self, dt: float = 1.0) -> EmotionalState:
        """
        Advance time — emotions decay toward baselines.

        Args:
            dt: Time step in hours.

        Returns:
            New emotional state.
        """
        self._valence.step(dt)
        self._arousal.step(dt)
        self._dominance.step(dt)

        return self.current_state

    def reset(self) -> None:
        """Reset to default emotional state."""
        self._valence.reset()
        self._arousal.reset()
        self._dominance.reset()
        self._current_stage = None

    def __repr__(self) -> str:
        return f"EmotionEngine({self.current_state})"
