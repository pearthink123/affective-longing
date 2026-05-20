"""
Emotion module — VAD model for AI companion's internal state.

Components:
- Emotion: Discrete emotion labels (joy, sadness, anger, etc.)
- EmotionalState: VAD snapshot with emotion label
- EmotionEngine: Manages emotional state with OU decay + events
"""

from .engine import EmotionEngine, EmotionEngineConfig
from .vad import Emotion, EmotionalState

__all__ = [
    "Emotion",
    "EmotionalState",
    "EmotionEngine",
    "EmotionEngineConfig",
]
