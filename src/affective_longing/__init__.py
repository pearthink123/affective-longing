"""
affective-longing: Emotional extension for AI companions.

Built on revive-companion, adds:
- Memory triggers (embedding similarity)
- Relationship state machine (HMM + OU process)
- AI self-emotion modeling (placeholder)
"""

from .engine import AffectiveLonging, AffectiveResult
from .memory import MemoryStore, TriggerEngine
from .relationship import (
    Event,
    OUProcess,
    RelationshipStateMachine,
    RelationshipState,
    Stage,
    TransitionConfig,
)

__all__ = [
    # Core engine
    "AffectiveLonging",
    "AffectiveResult",
    # Memory
    "MemoryStore",
    "TriggerEngine",
    # Relationship
    "Stage",
    "RelationshipState",
    "Event",
    "RelationshipStateMachine",
    "TransitionConfig",
    "OUProcess",
]
__version__ = "0.1.0"
