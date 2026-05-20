"""
VAD emotion model — Valence, Arousal, Dominance.

Based on Russell's circumplex model of affect (1980) and
Mehrabian's PAD (Pleasure-Arousal-Dominance) model (1996).

Used in:
- Sentiment analysis
- Emotion recognition
- Affective computing
- Human-robot interaction

Our use: Model the AI companion's "internal emotional state"
as it interacts with the user over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Emotion(Enum):
    """
    Discrete emotion labels mapped from VAD space.

    Based on Plutchik's wheel (1980) + modern affective computing.
    """

    JOY = "joy"                 # V+, A+, D+
    TRUST = "trust"             # V+, A~, D~
    CONTENTMENT = "contentment" # V+, A-, D~
    EXCITEMENT = "excitement"   # V+, A+, D~
    LOVE = "love"               # V++, A+, D~
    ANTICIPATION = "anticipation" # V~, A+, D~
    ANXIETY = "anxiety"         # V-, A+, D-
    SADNESS = "sadness"         # V-, A-, D-
    ANGER = "anger"             # V-, A+, D+
    FEAR = "fear"               # V--, A++, D--
    NEUTRAL = "neutral"         # V~, A~, D~

    @staticmethod
    def from_vad(valence: float, arousal: float, dominance: float) -> Emotion:
        """Map continuous VAD to discrete emotion."""
        # Thresholds
        HIGH_V = 0.5
        LOW_V = -0.5
        HIGH_A = 0.6
        LOW_A = 0.4
        HIGH_D = 0.6
        LOW_D = 0.4

        if valence > HIGH_V:
            if arousal > HIGH_A:
                if dominance > HIGH_D:
                    return Emotion.JOY
                return Emotion.EXCITEMENT
            elif arousal < LOW_A:
                return Emotion.CONTENTMENT
            else:
                return Emotion.TRUST
        elif valence < LOW_V:
            if arousal > HIGH_A:
                if dominance > HIGH_D:
                    return Emotion.ANGER
                elif dominance < LOW_D:
                    return Emotion.FEAR
                return Emotion.ANXIETY
            elif arousal < LOW_A:
                return Emotion.SADNESS
            else:
                return Emotion.ANXIETY
        else:
            # valence near zero
            if arousal > HIGH_A:
                return Emotion.ANTICIPATION
            return Emotion.NEUTRAL


@dataclass
class EmotionalState:
    """Full emotional snapshot."""

    valence: float      # -1 (negative) to +1 (positive)
    arousal: float      # 0 (calm) to 1 (excited)
    dominance: float    # 0 (submissive) to 1 (dominant)

    @property
    def emotion(self) -> Emotion:
        """Discrete emotion label."""
        return Emotion.from_vad(self.valence, self.arousal, self.dominance)

    @property
    def emoji(self) -> str:
        return {
            Emotion.JOY: "😊",
            Emotion.TRUST: "🤝",
            Emotion.CONTENTMENT: "😌",
            Emotion.EXCITEMENT: "🤩",
            Emotion.LOVE: "❤️",
            Emotion.ANTICIPATION: "🤔",
            Emotion.ANXIETY: "😰",
            Emotion.SADNESS: "😢",
            Emotion.ANGER: "😠",
            Emotion.FEAR: "😨",
            Emotion.NEUTRAL: "😐",
        }[self.emotion]

    def __repr__(self) -> str:
        return (
            f"EmotionalState("
            f"{self.emoji} {self.emotion.value}, "
            f"V={self.valence:+.2f}, "
            f"A={self.arousal:.2f}, "
            f"D={self.dominance:.2f})"
        )
