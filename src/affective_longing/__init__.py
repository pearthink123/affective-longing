"""
affective-longing: Emotional extension for AI companions.

Built on revive-companion, adds:
- Memory triggers (embedding similarity)
- Relationship state machine
- AI self-emotion modeling
"""

from .engine import AffectiveLonging

__all__ = ["AffectiveLonging"]
__version__ = "0.1.0"
