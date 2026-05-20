"""
Quickstart — memory-triggered longing demo.
"""
from affective_longing import AffectiveLonging

# Initialize (loads sentence-transformers on first use)
engine = AffectiveLonging(memory_persist_dir="./demo_memory_db")

# Store some memories
engine.remember("你说你喜欢下雨天，我们一起在阳台喝咖啡", tags=["weather", "romantic"])
engine.remember("我们第一次见面是在那家咖啡店", tags=["place", "milestone"])
engine.remember("你上次说最近工作压力很大", tags=["mood", "work"])

# Simulate a tick with context
from datetime import datetime

# Normal tick — no memory trigger
result1 = engine.tick(now=datetime(2026, 5, 20, 10, 0))
print(f"Normal: prob={result1.base_result.probability:.3f}, boost={result1.longing_boost:.3f}")

# Tick with memory-triggering context
result2 = engine.tick(now=datetime(2026, 5, 20, 10, 0), context="今天下雨了")
print(f"Rain! : prob={result2.base_result.probability:.3f}, boost={result2.longing_boost:.3f}")
print(f"  → triggered memory: {result2.memory_trigger}")
print(f"  → similarity: {result2.memory_similarity:.3f}")
print(f"  → boosted prob: {result2.boosted_probability:.3f}")
