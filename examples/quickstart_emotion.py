"""
Quickstart: Emotion engine (VAD model).

Run:
    python examples/quickstart_emotion.py

Shows:
1. How events drive emotional changes
2. How relationship stages shift baselines
3. How emotions decay over time
4. VAD → discrete emotion mapping
"""

from affective_longing import EmotionEngine, Stage


def main():
    print("=== VAD Emotion Engine Demo ===\n")

    engine = EmotionEngine(seed=42)
    print(f"Initial: {engine.current_state}\n")

    # Phase 1: Happy events
    print("--- Phase 1: Positive events ---")
    events = ["reply_fast", "affection", "initiate", "long_message"]
    for event in events:
        engine.observe(event)
        print(f"  {event:15s} → {engine.current_state}")

    print()

    # Phase 2: Relationship shifts to SWEET
    print("--- Phase 2: Relationship → SWEET ---")
    engine.update_relationship_stage(Stage.SWEET)
    print(f"  Baselines updated: {engine.current_state}")
    for i in range(5):
        engine.step_time(dt=6)  # 6 hours
        print(f"  +{6*(i+1):2d}h: {engine.current_state}")

    print()

    # Phase 3: Conflict
    print("--- Phase 3: Conflict ---")
    engine.observe("fight")
    print(f"  fight           → {engine.current_state}")
    engine.observe("no_reply")
    print(f"  no_reply        → {engine.current_state}")
    engine.observe("long_silence")
    print(f"  long_silence    → {engine.current_state}")

    print()

    # Phase 4: Relationship shifts to COLD
    print("--- Phase 4: Relationship → COLD ---")
    engine.update_relationship_stage(Stage.COLD)
    print(f"  Baselines updated: {engine.current_state}")
    for i in range(5):
        engine.step_time(dt=12)
        print(f"  +{12*(i+1):2d}h: {engine.current_state}")

    print()

    # Phase 5: Recovery
    print("--- Phase 5: Recovery ---")
    engine.observe("apology")
    print(f"  apology         → {engine.current_state}")
    engine.update_relationship_stage(Stage.REPAIRING)
    print(f"  → REPAIRING     → {engine.current_state}")
    engine.observe("reply_fast")
    print(f"  reply_fast      → {engine.current_state}")
    engine.observe("affection")
    print(f"  affection       → {engine.current_state}")

    print()

    # Summary
    state = engine.current_state
    print("=== Final State ===")
    print(f"  {state.emoji} {state.emotion.value}")
    print(f"  Valence:   {state.valence:+.2f} (愉快 ↔ 不愉快)")
    print(f"  Arousal:   {state.arousal:.2f}  (兴奋 ↔ 平静)")
    print(f"  Dominance: {state.dominance:.2f}  (控制 ↔ 被控)")


if __name__ == "__main__":
    main()
