"""
affective-longing: Emotional extension for AI companions.

Built on revive-companion, adds:
- Memory triggers (embedding similarity)
- Relationship state machine (placeholder)
- AI self-emotion modeling (placeholder)
"""

from .engine import AffectiveLonging, AffectiveResult
from .memory import MemoryStore, TriggerEngine

__all__ = ["AffectiveLonging", "AffectiveResult", "MemoryStore", "TriggerEngine"]
__version__ = "0.1.0"
