"""
affective-longing: Emotional extension for AI companions.

Built on revive-companion, adds:
- Memory triggers (embedding similarity)
- Relationship state machine (HMM + OU process)
- AI self-emotion modeling (VAD model)

Three layers, one decision:
1. Memory — past memories trigger present longing
2. Relationship — 6-stage lifecycle with OU decay
3. Emotion — VAD model for AI's internal state
"""

from .emotion import Emotion, EmotionEngine, EmotionEngineConfig, EmotionalState
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
    # Emotion
    "Emotion",
    "EmotionalState",
    "EmotionEngine",
    "EmotionEngineConfig",
]
__version__ = "0.1.0"
