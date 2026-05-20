"""Tests for emotion engine."""

import pytest

from affective_longing.emotion import Emotion, EmotionEngine, EmotionalState
from affective_longing.relationship import Stage


# ── Emotion Mapping Tests ──


class TestEmotion:
    def test_joy_mapping(self):
        e = Emotion.from_vad(0.7, 0.7, 0.7)
        assert e == Emotion.JOY

    def test_sadness_mapping(self):
        e = Emotion.from_vad(-0.6, 0.3, 0.5)
        assert e == Emotion.SADNESS

    def test_anger_mapping(self):
        e = Emotion.from_vad(-0.6, 0.7, 0.7)
        assert e == Emotion.ANGER

    def test_fear_mapping(self):
        e = Emotion.from_vad(-0.7, 0.8, 0.3)
        assert e == Emotion.FEAR

    def test_neutral_mapping(self):
        e = Emotion.from_vad(0.0, 0.4, 0.5)
        assert e == Emotion.NEUTRAL


# ── EmotionalState Tests ──


class TestEmotionalState:
    def test_emoji(self):
        state = EmotionalState(valence=0.7, arousal=0.7, dominance=0.7)
        assert state.emoji == "😊"

    def test_repr(self):
        state = EmotionalState(valence=0.7, arousal=0.7, dominance=0.7)
        assert "joy" in repr(state)


# ── EmotionEngine Tests ──


class TestEmotionEngine:
    def test_initial_state(self):
        engine = EmotionEngine(seed=42)
        assert engine.valence == pytest.approx(0.3)
        assert engine.arousal == pytest.approx(0.4)
        assert engine.dominance == pytest.approx(0.5)

    def test_observe_affection(self):
        engine = EmotionEngine(seed=42)
        state = engine.observe("affection")
        assert state.valence > 0.3  # Should increase
        assert state.arousal > 0.4  # Should increase

    def test_observe_fight(self):
        engine = EmotionEngine(seed=42)
        state = engine.observe("fight")
        assert state.valence < 0.3  # Should decrease
        assert state.arousal > 0.4  # Should increase (agitated)

    def test_observe_no_reply(self):
        engine = EmotionEngine(seed=42)
        state = engine.observe("no_reply")
        assert state.valence < 0.3
        assert state.dominance < 0.5

    def test_update_relationship_stage(self):
        engine = EmotionEngine(seed=42)
        engine.update_relationship_stage(Stage.COLD)
        # Baseline should shift negative
        assert engine._valence.baseline < 0

    def test_step_time_decays(self):
        engine = EmotionEngine(seed=42)
        engine.observe("affection")  # Bump up
        high_valence = engine.valence
        engine.step_time(dt=48)  # 48 hours
        assert engine.valence < high_valence

    def test_cold_stage_emotion(self):
        """In COLD stage, emotions should drift negative."""
        engine = EmotionEngine(seed=42)
        engine.update_relationship_stage(Stage.COLD)
        for _ in range(30):
            engine.step_time(dt=12)
        assert engine.valence < 0.1  # Should be near or below baseline (-0.5)

    def test_sweet_stage_emotion(self):
        """In SWEET stage, emotions should drift positive."""
        engine = EmotionEngine(seed=42)
        engine.update_relationship_stage(Stage.SWEET)
        for _ in range(20):
            engine.step_time(dt=12)
        assert engine.valence > 0.3

    def test_reset(self):
        engine = EmotionEngine(seed=42)
        engine.observe("fight")
        engine.reset()
        assert engine.valence == pytest.approx(0.3)

    def test_current_state_returns_dataclass(self):
        engine = EmotionEngine(seed=42)
        state = engine.current_state
        assert isinstance(state, EmotionalState)
        assert state.emotion == Emotion.NEUTRAL

    def test_repr(self):
        engine = EmotionEngine(seed=42)
        assert "EmotionEngine" in repr(engine)
        assert "neutral" in repr(engine)
