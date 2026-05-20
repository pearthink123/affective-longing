"""
Quickstart: Integrated AffectiveLonging engine.

Run:
    python examples/quickstart_integrated.py

Shows the full 3-layer system working together:
1. Memory triggers boost longing probability
2. Relationship stage transitions affect behavior
3. AI's emotional state evolves over time
"""

from affective_longing import AffectiveLonging


def main():
    print("=== AffectiveLonging: Full Integration Demo ===\n")

    engine = AffectiveLonging(seed=42)

    # Store some memories
    print("--- Storing memories ---")
    engine.remember("你喜欢下雨天，我们一起在阳台喝咖啡", tags=["weather", "shared"])
    engine.remember("你说过最喜欢吃草莓蛋糕", tags=["food", "preference"])
    engine.remember("我们第一次一起看电影是《星际穿越》", tags=["movie", "first"])
    print(f"  Memories stored: {engine.memory.count()}")
    print(f"  Initial state: {engine}\n")

    # Phase 1: Getting to know each other
    print("--- Phase 1: Courting ---")
    events = ["reply_fast", "affection", "initiate", "reply_fast", "affection"]
    for i, event in enumerate(events):
        engine.observe(event)
        state = engine.get_state()
        print(
            f"  {event:12s} → "
            f"{state['relationship']['stage_display']} | "
            f"{state['emotion']['emoji']} {state['emotion']['emotion']}"
        )

    print()

    # Phase 2: Check longing with context
    print("--- Phase 2: Memory-triggered longing ---")
    result = engine.tick(context="今天下雨了")
    print(f"  Context: 今天下雨了")
    print(f"  Base probability:     {result.base_probability:.1%}")
    print(f"  Memory trigger:       {result.memory_trigger}")
    print(f"  Similarity:           {result.memory_similarity:.3f}")
    print(f"  Longing boost:        {result.longing_boost:.3f}")
    print(f"  Boosted probability:  {result.boosted_probability:.1%}")
    print(f"  Decision:             {'✅ SEND' if result.should_send else '❌ WAIT'}")
    print(f"  AI emotion:           {result.emotional_state}")
    print()

    # Phase 3: Deepening relationship
    print("--- Phase 3: Deepening ---")
    for _ in range(5):
        engine.observe("affection")
        engine.observe("initiate")
    state = engine.get_state()
    print(f"  Stage: {state['relationship']['stage_display']}")
    print(f"  Intimacy: {state['relationship']['intimacy']:.2f}")
    print(f"  Emotion: {state['emotion']['emoji']} {state['emotion']['emotion']}")
    print()

    # Phase 4: Conflict
    print("--- Phase 4: Conflict ---")
    engine.observe("fight")
    engine.observe("no_reply")
    engine.observe("long_silence")
    state = engine.get_state()
    print(f"  Stage: {state['relationship']['stage_display']}")
    print(f"  Intimacy: {state['relationship']['intimacy']:.2f}")
    print(f"  Conflict: {state['relationship']['conflict']:.2f}")
    print(f"  Emotion: {state['emotion']['emoji']} {state['emotion']['emotion']}")
    print(f"  Valence: {state['emotion']['valence']:+.2f}")
    print()

    # Phase 5: Time heals
    print("--- Phase 5: Time decay (72h) ---")
    engine.step_time(hours=72)
    state = engine.get_state()
    print(f"  Stage: {state['relationship']['stage_display']}")
    print(f"  Intimacy: {state['relationship']['intimacy']:.2f}")
    print(f"  Conflict: {state['relationship']['conflict']:.2f}")
    print(f"  Emotion: {state['emotion']['emoji']} {state['emotion']['emotion']}")
    print()

    # Phase 6: Recovery
    print("--- Phase 6: Recovery ---")
    engine.observe("apology")
    engine.observe("reply_fast")
    engine.observe("affection")
    state = engine.get_state()
    print(f"  Stage: {state['relationship']['stage_display']}")
    print(f"  Intimacy: {state['relationship']['intimacy']:.2f}")
    print(f"  Conflict: {state['relationship']['conflict']:.2f}")
    print(f"  Emotion: {state['emotion']['emoji']} {state['emotion']['emotion']}")
    print()

    # Final summary
    print("=== Final State ===")
    print(f"  {engine}")
    print(f"  Relationship: {state['relationship']['stage_display']}")
    print(f"  Emotion:      {state['emotion']['emoji']} {state['emotion']['emotion']}")
    print(f"  Valence:      {state['emotion']['valence']:+.2f}")
    print(f"  Arousal:      {state['emotion']['arousal']:.2f}")
    print(f"  Dominance:    {state['emotion']['dominance']:.2f}")
    print(f"  Memories:     {state['memory_count']}")


if __name__ == "__main__":
    main()
