"""
Quickstart: Relationship state machine.

Run:
    pip install -e ".[memory]"
    python examples/quickstart_relationship.py

Shows:
1. How events drive stage transitions
2. How OU process decays intimacy/conflict over time
3. The full lifecycle: courting → sweet → passionate → stable → cold → repairing
"""

from affective_longing import Event, RelationshipStateMachine


def main():
    print("=== Relationship State Machine Demo ===\n")

    sm = RelationshipStateMachine(seed=42)
    print(f"Initial: {sm.current_state}\n")

    # Phase 1: Courting — lots of affection, fast replies
    print("--- Phase 1: Courting → Sweet ---")
    for i in range(5):
        sm.observe(Event.REPLY_FAST)
        sm.observe(Event.AFFECTION)
        print(f"  Tick {i+1}: {sm.current_state}")

    print()

    # Phase 2: Sweet → Passionate — deepening connection
    print("--- Phase 2: Sweet → Passionate ---")
    for i in range(5):
        sm.observe(Event.INITIATE)  # User starts reaching out
        sm.observe(Event.AFFECTION)
        print(f"  Tick {i+1}: {sm.current_state}")

    print()

    # Phase 3: Passionate → Stable — comfortable routine
    print("--- Phase 3: Passionate → Stable (time decay) ---")
    for i in range(10):
        sm.step_time(dt=24)  # 24 hours pass
        if i % 3 == 0:
            sm.observe(Event.REPLY_SLOW)  # Occasional slow replies
        print(f"  Day {i+1}: {sm.current_state}")

    print()

    # Phase 4: Conflict!
    print("--- Phase 4: Conflict → Cold ---")
    sm.observe(Event.FIGHT)
    print(f"  After fight: {sm.current_state}")
    sm.observe(Event.NO_REPLY)
    print(f"  No reply: {sm.current_state}")
    sm.observe(Event.LONG_SILENCE)
    print(f"  Long silence: {sm.current_state}")

    print()

    # Phase 5: Repair
    print("--- Phase 5: Repair ---")
    sm.observe(Event.APOLOGY)
    print(f"  After apology: {sm.current_state}")
    sm.observe(Event.REPLY_FAST)
    print(f"  Fast reply: {sm.current_state}")
    sm.observe(Event.AFFECTION)
    print(f"  Affection: {sm.current_state}")

    print()

    # Summary
    state = sm.current_state
    print("=== Final State ===")
    print(f"  Stage: {state.stage.display_name}")
    print(f"  Intimacy: {state.intimacy:.2f}")
    print(f"  Conflict: {state.conflict:.2f}")
    print(f"  Confidence: {state.confidence:.2f}")
    print(f"  Ticks in stage: {state.ticks_in_stage}")


if __name__ == "__main__":
    main()
