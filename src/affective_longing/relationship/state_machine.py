"""
Relationship state machine — HMM + OU process.

Architecture:
    HMM (discrete): Which stage are we in?
        - Transition probabilities depend on events + continuous values
        - Events: reply, fight, long_silence, affection, apology

    OU (continuous): How intimate/conflicted are we?
        - Intimacy drifts toward baseline (0.5) without interaction
        - Conflict decays toward 0.1 over time
        - Events bump these values, which in turn affect HMM transitions

Flow:
    1. Event arrives (reply, fight, etc.)
    2. OU values are bumped
    3. HMM transition probabilities are computed from (stage, intimacy, conflict, event)
    4. New stage is sampled (argmax for deterministic, or sample for stochastic)
    5. Between events, OU processes decay toward baseline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np

from .ou_process import OUProcess
from .stages import RelationshipState, Stage

logger = logging.getLogger(__name__)


class Event(Enum):
    """Events that can trigger stage transitions."""

    REPLY_FAST = "reply_fast"         # User replied quickly (< 5 min)
    REPLY_SLOW = "reply_slow"         # User replied slowly (> 1 hour)
    NO_REPLY = "no_reply"             # User didn't reply to our message
    LONG_SILENCE = "long_silence"     # No contact for > 24 hours
    AFFECTION = "affection"           # User showed warmth (long msg, emoji, etc.)
    FIGHT = "fight"                   # Conflict, harsh words
    APOLOGY = "apology"               # Someone apologized
    INITIATE = "initiate"             # User initiated contact unprompted
    REJECT = "reject"                 # User rejected our message/call


@dataclass
class TransitionConfig:
    """Configures HMM transition behavior."""

    # Base transition probabilities (stage -> stage)
    # These are priors; actual probs depend on events + OU values
    transition_matrix: dict[Stage, dict[Stage, float]] = field(
        default_factory=lambda: {
            Stage.COURTING: {
                Stage.COURTING: 0.6,
                Stage.SWEET: 0.3,
                Stage.COLD: 0.1,
                Stage.PASSIONATE: 0.0,
                Stage.STABLE: 0.0,
                Stage.REPAIRING: 0.0,
            },
            Stage.SWEET: {
                Stage.COURTING: 0.05,
                Stage.SWEET: 0.5,
                Stage.PASSIONATE: 0.35,
                Stage.COLD: 0.1,
                Stage.STABLE: 0.0,
                Stage.REPAIRING: 0.0,
            },
            Stage.PASSIONATE: {
                Stage.COURTING: 0.0,
                Stage.SWEET: 0.1,
                Stage.PASSIONATE: 0.4,
                Stage.STABLE: 0.4,
                Stage.COLD: 0.1,
                Stage.REPAIRING: 0.0,
            },
            Stage.STABLE: {
                Stage.COURTING: 0.0,
                Stage.SWEET: 0.05,
                Stage.PASSIONATE: 0.05,
                Stage.STABLE: 0.7,
                Stage.COLD: 0.15,
                Stage.REPAIRING: 0.05,
            },
            Stage.COLD: {
                Stage.COURTING: 0.0,
                Stage.SWEET: 0.0,
                Stage.PASSIONATE: 0.0,
                Stage.STABLE: 0.1,
                Stage.COLD: 0.6,
                Stage.REPAIRING: 0.3,
            },
            Stage.REPAIRING: {
                Stage.COURTING: 0.0,
                Stage.SWEET: 0.15,
                Stage.PASSIONATE: 0.0,
                Stage.STABLE: 0.35,
                Stage.COLD: 0.2,
                Stage.REPAIRING: 0.3,
            },
        }
    )

    # OU process parameters
    intimacy_baseline: float = 0.5
    intimacy_reversion: float = 0.05    # slow decay (bonds are sticky)
    intimacy_volatility: float = 0.02

    conflict_baseline: float = 0.1
    conflict_reversion: float = 0.08    # faster decay (conflicts fade)
    conflict_volatility: float = 0.03

    # Event impact on OU values
    event_bumps: dict[Event, tuple[float, float]] = field(
        default_factory=lambda: {
            # (intimacy_bump, conflict_bump)
            Event.REPLY_FAST: (0.05, -0.02),
            Event.REPLY_SLOW: (-0.02, 0.01),
            Event.NO_REPLY: (-0.05, 0.03),
            Event.LONG_SILENCE: (-0.10, 0.05),
            Event.AFFECTION: (0.10, -0.05),
            Event.FIGHT: (-0.15, 0.20),
            Event.APOLOGY: (0.05, -0.15),
            Event.INITIATE: (0.08, -0.02),
            Event.REJECT: (-0.12, 0.10),
        }
    )


class RelationshipStateMachine:
    """
    HMM + OU relationship state machine.

    Example:
        >>> sm = RelationshipStateMachine()
        >>> state = sm.current_state
        >>> state.stage
        <Stage.COURTING: 'courting'>

        >>> sm.observe(Event.REPLY_FAST)
        >>> sm.observe(Event.AFFECTION)
        >>> sm.current_state.stage
        <Stage.SWEET: 'sweet'>

        >>> sm.step_time(dt=24)  # 24 hours of silence
        >>> sm.current_state.intimacy  # decayed toward baseline
    """

    def __init__(
        self,
        config: TransitionConfig | None = None,
        initial_stage: Stage = Stage.COURTING,
        seed: int | None = None,
    ):
        self.config = config or TransitionConfig()
        self._stage = initial_stage
        self._ticks_in_stage = 0
        self._seed = seed

        # OU processes for continuous emotional dimensions
        self._intimacy = OUProcess(
            baseline=self.config.intimacy_baseline,
            reversion_speed=self.config.intimacy_reversion,
            volatility=self.config.intimacy_volatility,
            seed=seed,
        )
        self._conflict = OUProcess(
            baseline=self.config.conflict_baseline,
            reversion_speed=self.config.conflict_reversion,
            volatility=self.config.conflict_volatility,
            seed=(seed + 1) if seed is not None else None,
        )

        # HMM belief state (probability over stages)
        self._belief = np.zeros(len(Stage))
        stage_list = list(Stage)
        self._belief[stage_list.index(initial_stage)] = 1.0

        logger.info(
            "RelationshipStateMachine initialized: stage=%s, intimacy=%.2f, conflict=%.2f",
            initial_stage.value,
            self._intimacy.value,
            self._conflict.value,
        )

    @property
    def current_state(self) -> RelationshipState:
        """Full snapshot of the relationship."""
        stage_list = list(Stage)
        confidence = float(self._belief[stage_list.index(self._stage)])
        return RelationshipState(
            stage=self._stage,
            intimacy=self._intimacy.value,
            conflict=self._conflict.value,
            confidence=confidence,
            ticks_in_stage=self._ticks_in_stage,
        )

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def intimacy(self) -> float:
        return self._intimacy.value

    @property
    def conflict(self) -> float:
        return self._conflict.value

    def observe(self, event: Event) -> Stage:
        """
        Process an event and potentially transition to a new stage.

        This is the core method — call it when something happens:
        - User replies -> Event.REPLY_FAST or Event.REPLY_SLOW
        - User doesn't reply -> Event.NO_REPLY
        - Conflict -> Event.FIGHT
        - etc.

        Returns:
            New stage after the transition.
        """
        # 1. Bump OU values based on event
        intimacy_bump, conflict_bump = self.config.event_bumps.get(
            event, (0.0, 0.0)
        )
        self._intimacy.bump(intimacy_bump)
        self._conflict.bump(conflict_bump)

        # 2. Compute transition probabilities
        trans_probs = self._compute_transition_probs(event)

        # 3. Update HMM belief state
        stage_list = list(Stage)
        old_idx = stage_list.index(self._stage)
        new_belief = np.zeros(len(Stage))
        for j, stage in enumerate(Stage):
            new_belief[j] = self._belief[old_idx] * trans_probs[stage]

        # Normalize
        total = new_belief.sum()
        if total > 0:
            new_belief /= total

        self._belief = new_belief

        # 4. Sample new stage (argmax = deterministic)
        new_idx = int(np.argmax(new_belief))
        new_stage = stage_list[new_idx]

        if new_stage != self._stage:
            logger.info(
                "Stage transition: %s -> %s (event=%s, intimacy=%.2f, conflict=%.2f)",
                self._stage.value,
                new_stage.value,
                event.value,
                self._intimacy.value,
                self._conflict.value,
            )
            self._stage = new_stage
            self._ticks_in_stage = 0
        else:
            self._ticks_in_stage += 1

        return self._stage

    def step_time(self, dt: float = 1.0) -> None:
        """
        Advance time without an event (OU decay).

        Call this periodically (e.g., every hour) to let intimacy/conflict
        drift toward their baselines.

        Args:
            dt: Time step in hours.
        """
        self._intimacy.step(dt)
        self._conflict.step(dt)
        self._ticks_in_stage += 1

        # Check if OU values suggest a stage transition
        self._check_ou_driven_transition()

    def _compute_transition_probs(
        self, event: Event
    ) -> dict[Stage, float]:
        """
        Compute P(next_stage | current_stage, event, intimacy, conflict).

        Starts with base transition matrix, then modulates by:
        - Event type (some events make certain transitions more likely)
        - Intimacy level (high intimacy -> more likely to stay in positive stages)
        - Conflict level (high conflict -> more likely to go to COLD)
        """
        base_probs = dict(self.config.transition_matrix[self._stage])

        # Modulate by intimacy
        intimacy = self._intimacy.value
        if intimacy > 0.7:
            # Boost positive stages
            base_probs[Stage.SWEET] *= 2.0
            base_probs[Stage.PASSIONATE] *= 2.0
            base_probs[Stage.COLD] *= 0.3
        elif intimacy < 0.3:
            # Boost negative stages
            base_probs[Stage.COLD] *= 2.0
            base_probs[Stage.SWEET] *= 0.3
            base_probs[Stage.PASSIONATE] *= 0.2

        # Modulate by conflict
        conflict = self._conflict.value
        if conflict > 0.5:
            base_probs[Stage.COLD] *= 3.0
            base_probs[Stage.REPAIRING] *= 2.0
            base_probs[Stage.SWEET] *= 0.2
            base_probs[Stage.PASSIONATE] *= 0.1
        elif conflict < 0.2:
            base_probs[Stage.COLD] *= 0.2

        # Event-specific overrides
        if event == Event.FIGHT:
            base_probs[Stage.COLD] *= 5.0
            base_probs[self._stage] *= 0.5  # Less likely to stay
        elif event == Event.APOLOGY:
            base_probs[Stage.REPAIRING] *= 3.0
            base_probs[Stage.COLD] *= 0.3
        elif event == Event.AFFECTION and intimacy > 0.6:
            base_probs[Stage.PASSIONATE] *= 3.0
            base_probs[Stage.SWEET] *= 2.0
            base_probs[self._stage] *= 0.5  # Less likely to stay
        elif event == Event.LONG_SILENCE:
            base_probs[Stage.COLD] *= 2.0

        # Normalize
        total = sum(base_probs.values())
        if total > 0:
            for stage in base_probs:
                base_probs[stage] /= total

        return base_probs

    def _check_ou_driven_transition(self) -> None:
        """
        Check if OU values suggest a stage transition even without an event.

        E.g., if intimacy has decayed very low in SWEET stage,
        maybe we should transition to STABLE or COLD.
        """
        intimacy = self._intimacy.value
        conflict = self._conflict.value

        # High conflict in a positive stage -> likely should be in COLD
        if conflict > 0.7 and self._stage in (Stage.SWEET, Stage.PASSIONATE, Stage.STABLE):
            # Probabilistic: don't always transition
            if np.random.random() < conflict:
                self._stage = Stage.COLD
                self._ticks_in_stage = 0
                logger.info(
                    "OU-driven transition: -> COLD (conflict=%.2f)", conflict
                )

        # Low conflict + high intimacy in COLD -> maybe recovering
        elif conflict < 0.3 and intimacy > 0.5 and self._stage == Stage.COLD:
            if np.random.random() < 0.3:
                self._stage = Stage.REPAIRING
                self._ticks_in_stage = 0
                logger.info(
                    "OU-driven transition: -> REPAIRING (intimacy=%.2f, conflict=%.2f)",
                    intimacy,
                    conflict,
                )

    def reset(self, stage: Stage = Stage.COURTING) -> None:
        """Reset to initial state."""
        self._stage = stage
        self._ticks_in_stage = 0
        self._intimacy.reset()
        self._conflict.reset()

        stage_list = list(Stage)
        self._belief = np.zeros(len(Stage))
        self._belief[stage_list.index(stage)] = 1.0

    def __repr__(self) -> str:
        return (
            f"RelationshipStateMachine("
            f"stage={self._stage.display_name}, "
            f"intimacy={self._intimacy.value:.2f}, "
            f"conflict={self._conflict.value:.2f})"
        )
