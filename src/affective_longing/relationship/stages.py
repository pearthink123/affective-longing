"""
Relationship stages — the 6 phases of a relationship.

Design notes:
- Stages are discrete (HMM states)
- Each stage has a "valence" (positive/negative) and "intensity" (how strong)
- Transitions are event-driven (reply, fight, silence) + time-decay (OU process)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Stage(Enum):
    """Six relationship phases."""

    COURTING = "courting"       # 追求 — initial, uncertain, high effort
    SWEET = "sweet"             # 甜蜜 — reciprocated, warm, honeymoon
    PASSIONATE = "passionate"   # 热恋 — intense, frequent contact, deep
    STABLE = "stable"           # 平稳 — comfortable, routine, low drama
    COLD = "cold"               # 冷战 — conflict, silence, distance
    REPAIRING = "repairing"     # 修复 — recovering, rebuilding trust

    @property
    def valence(self) -> float:
        """Emotional valence: -1 (negative) to +1 (positive)."""
        return {
            Stage.COURTING: 0.3,
            Stage.SWEET: 0.8,
            Stage.PASSIONATE: 1.0,
            Stage.STABLE: 0.5,
            Stage.COLD: -0.7,
            Stage.REPAIRING: -0.2,
        }[self]

    @property
    def intensity(self) -> float:
        """Emotional intensity: 0 (calm) to 1 (extreme)."""
        return {
            Stage.COURTING: 0.6,
            Stage.SWEET: 0.7,
            Stage.PASSIONATE: 1.0,
            Stage.STABLE: 0.3,
            Stage.COLD: 0.8,
            Stage.REPAIRING: 0.5,
        }[self]

    @property
    def display_name(self) -> str:
        return {
            Stage.COURTING: "追求",
            Stage.SWEET: "甜蜜",
            Stage.PASSIONATE: "热恋",
            Stage.STABLE: "平稳",
            Stage.COLD: "冷战",
            Stage.REPAIRING: "修复",
        }[self]


@dataclass
class RelationshipState:
    """Full snapshot of the relationship."""

    stage: Stage
    intimacy: float       # 0-1: how close
    conflict: float       # 0-1: how much tension
    confidence: float     # 0-1: how sure we are about this stage
    ticks_in_stage: int   # how long in current stage

    def __repr__(self) -> str:
        return (
            f"RelationshipState("
            f"stage={self.stage.display_name}, "
            f"intimacy={self.intimacy:.2f}, "
            f"conflict={self.conflict:.2f}, "
            f"confidence={self.confidence:.2f})"
        )
