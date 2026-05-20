"""Tests for integrated AffectiveLonging engine."""

import os
import shutil

import pytest

from affective_longing import AffectiveLonging, AffectiveResult, Stage


@pytest.fixture
def engine(tmp_path):
    """Create a fresh engine for each test."""
    db_dir = str(tmp_path / "test_memory_db")
    eng = AffectiveLonging(memory_persist_dir=db_dir, seed=42)
    yield eng
    # ChromaDB may hold file handles on Windows, ignore cleanup errors
    try:
        del eng.memory  # Release ChromaDB client
    except Exception:
        pass


class TestAffectiveLonging:
    def test_initialization(self, engine):
        state = engine.get_state()
        assert state["relationship"]["stage"] == "courting"
        assert state["emotion"]["emotion"] == "neutral"
        assert state["memory_count"] == 0

    def test_remember(self, engine):
        mid = engine.remember("你喜欢下雨天", tags=["weather"])
        assert mid is not None
        assert engine.memory.count() == 1

    def test_observe_updates_relationship(self, engine):
        engine.observe("affection")
        assert engine.relationship.intimacy > 0.5

    def test_observe_updates_emotion(self, engine):
        engine.observe("affection")
        assert engine.emotion.valence > 0.3

    def test_step_time(self, engine):
        engine.observe("affection")  # Bump up
        intimacy_before = engine.relationship.intimacy
        engine.step_time(hours=48)
        # Should decay toward baseline
        assert engine.relationship.intimacy < intimacy_before

    def test_tick_without_context(self, engine):
        result = engine.tick()
        assert isinstance(result, AffectiveResult)
        assert result.memory_trigger is None
        assert result.emotional_state is not None

    def test_tick_with_context_no_memory(self, engine):
        result = engine.tick(context="今天天气不错")
        assert result.memory_trigger is None

    def test_tick_with_context_and_memory(self, engine):
        engine.remember("你喜欢下雨天")
        result = engine.tick(context="今天下雨了")
        # Memory should be triggered (or at least searched)
        assert result.base_probability > 0

    def test_observe_multiple_events(self, engine):
        for _ in range(5):
            engine.observe("affection")
            engine.observe("reply_fast")
        # Should be past courting
        assert engine.relationship.stage in (Stage.SWEET, Stage.PASSIONATE)

    def test_fight_changes_stage(self, engine):
        for _ in range(5):
            engine.observe("fight")
        assert engine.relationship.stage == Stage.COLD
        assert engine.emotion.valence < 0.3

    def test_get_state(self, engine):
        engine.remember("test memory")
        engine.observe("affection")
        state = engine.get_state()

        assert "relationship" in state
        assert "emotion" in state
        assert "memory_count" in state
        assert state["memory_count"] == 1
        assert state["relationship"]["stage"] in ("courting", "sweet")
        assert state["emotion"]["emoji"] is not None

    def test_repr(self, engine):
        engine.remember("test")
        r = repr(engine)
        assert "AffectiveLonging" in r
        assert "memories=1" in r

    def test_record_reply_passthrough(self, engine):
        # Should not raise
        engine.record_reply(reply_speed=0.8, reply_length=0.7)

    def test_record_send_passthrough(self, engine):
        # Should not raise
        engine.record_send()
