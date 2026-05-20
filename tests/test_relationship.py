"""Tests for relationship state machine."""

import pytest

from affective_longing.relationship import (
    Event,
    OUProcess,
    RelationshipStateMachine,
    RelationshipState,
    Stage,
    TransitionConfig,
)


# ── OU Process Tests ──


class TestOUProcess:
    def test_init_at_baseline(self):
        ou = OUProcess(baseline=0.5)
        assert ou.value == 0.5

    def test_bump(self):
        ou = OUProcess(baseline=0.5)
        ou.bump(0.3)
        assert ou.value == pytest.approx(0.8)

    def test_step_decays_toward_baseline(self):
        ou = OUProcess(baseline=0.5, reversion_speed=0.5, volatility=0.0, seed=42)
        ou.bump(0.3)  # 0.8
        ou.step(dt=1.0)
        # With no noise, should decay: 0.8 + 0.5 * (0.5 - 0.8) * 1.0 = 0.65
        assert ou.value == pytest.approx(0.65, abs=0.01)

    def test_clamp_to_zero_one(self):
        ou = OUProcess(baseline=0.5, volatility=0.0)
        ou.bump(10.0)
        ou.step(dt=1.0)
        assert 0.0 <= ou.value <= 1.0

    def test_reset(self):
        ou = OUProcess(baseline=0.5)
        ou.bump(0.3)
        ou.reset()
        assert ou.value == 0.5

    def test_reset_to_value(self):
        ou = OUProcess(baseline=0.5)
        ou.reset(value=0.8)
        assert ou.value == 0.8


# ── Stage Tests ──


class TestStage:
    def test_valence_positive_stages(self):
        assert Stage.SWEET.valence > 0
        assert Stage.PASSIONATE.valence > 0

    def test_valence_negative_stages(self):
        assert Stage.COLD.valence < 0

    def test_intensity_passionate_is_highest(self):
        assert Stage.PASSIONATE.intensity == 1.0

    def test_display_name(self):
        assert Stage.COURTING.display_name == "追求"
        assert Stage.COLD.display_name == "冷战"


# ── State Machine Tests ──


class TestRelationshipStateMachine:
    def test_initial_state(self):
        sm = RelationshipStateMachine(seed=42)
        assert sm.stage == Stage.COURTING
        assert sm.intimacy == pytest.approx(0.5)
        assert sm.conflict == pytest.approx(0.1)

    def test_custom_initial_stage(self):
        sm = RelationshipStateMachine(initial_stage=Stage.STABLE, seed=42)
        assert sm.stage == Stage.STABLE

    def test_observe_reply_fast(self):
        sm = RelationshipStateMachine(seed=42)
        sm.observe(Event.REPLY_FAST)
        # intimacy should increase
        assert sm.intimacy > 0.5

    def test_observe_fight(self):
        sm = RelationshipStateMachine(seed=42)
        sm.observe(Event.FIGHT)
        # conflict should increase
        assert sm.conflict > 0.1

    def test_courting_to_sweet_with_affection(self):
        """Multiple affection events should push toward SWEET."""
        sm = RelationshipStateMachine(seed=42)
        for _ in range(5):
            sm.observe(Event.AFFECTION)
            sm.observe(Event.REPLY_FAST)
        # Should have moved past COURTING
        assert sm.stage in (Stage.SWEET, Stage.PASSIONATE)

    def test_fight_causes_cold(self):
        """Multiple fights should push toward COLD."""
        sm = RelationshipStateMachine(initial_stage=Stage.STABLE, seed=42)
        for _ in range(5):
            sm.observe(Event.FIGHT)
        assert sm.stage == Stage.COLD

    def test_apology_after_fight(self):
        """After fight + apology, should move toward REPAIRING."""
        sm = RelationshipStateMachine(initial_stage=Stage.STABLE, seed=42)
        sm.observe(Event.FIGHT)
        sm.observe(Event.FIGHT)
        sm.observe(Event.APOLOGY)
        # Should be in COLD or REPAIRING
        assert sm.stage in (Stage.COLD, Stage.REPAIRING)

    def test_step_time_decays_intimacy(self):
        """Without events, intimacy should drift toward baseline."""
        sm = RelationshipStateMachine(seed=42)
        sm.observe(Event.AFFECTION)  # bump intimacy up
        high_intimacy = sm.intimacy
        sm.step_time(dt=48)  # 48 hours
        assert sm.intimacy < high_intimacy

    def test_current_state_returns_dataclass(self):
        sm = RelationshipStateMachine(seed=42)
        state = sm.current_state
        assert isinstance(state, RelationshipState)
        assert state.stage == Stage.COURTING

    def test_reset(self):
        sm = RelationshipStateMachine(seed=42)
        sm.observe(Event.FIGHT)
        sm.observe(Event.FIGHT)
        sm.reset()
        assert sm.stage == Stage.COURTING
        assert sm.intimacy == pytest.approx(0.5)
        assert sm.conflict == pytest.approx(0.1)

    def test_repr(self):
        sm = RelationshipStateMachine(seed=42)
        assert "追求" in repr(sm)
        assert "0.50" in repr(sm)
