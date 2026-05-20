# affective-longing 🧠💫

**Emotional extension for AI companions — beyond timing, into feeling.**

Built on [revive-companion](https://github.com/pearthink123/revive-companion) (Poisson timing + Bayesian inference).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AffectiveLonging                         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Memory     │  │ Relationship │  │   Emotion    │      │
│  │              │  │              │  │              │      │
│  │ ChromaDB     │  │ HMM          │  │ VAD Model    │      │
│  │ Embeddings   │  │ OU Process   │  │ Valence      │      │
│  │ Similarity   │  │ 6 Stages     │  │ Arousal      │      │
│  │              │  │              │  │ Dominance    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                 │
│                    tick() → AffectiveResult                 │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              revive-companion (base)                  │  │
│  │  Poisson Process → InfoGain → Bayesian → Decision   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Three Layers

### 1. Memory — Past triggers present

Store conversations as embeddings. When current context matches past memories, longing probability gets a boost.

```python
engine.remember("你喜欢下雨天", tags=["weather"])
engine.remember("我们第一次看电影是《星际穿越》", tags=["movie"])

# Later...
result = engine.tick(context="今天下雨了")
# → Triggers "你喜欢下雨天" with similarity 0.988
# → Longing boost: +15%
```

**Theory:** [Sentence embeddings](https://arxiv.org/abs/1708.00055) (all-MiniLM-L6-v2) + cosine similarity. Memory decay via [Ebbinghaus forgetting curve](https://en.wikipedia.org/wiki/Forgetting_curve).

### 2. Relationship — 6-stage lifecycle

Models relationship dynamics through discrete state transitions (HMM) + continuous emotional drift (Ornstein-Uhlenbeck process).

```
追求 → 甜蜜 → 热恋 → 平稳
  ↑                    ↓
  └──── 修复 ← 冷战 ←──┘
```

```python
engine.observe("affection")   # intimacy +0.10
engine.observe("fight")       # conflict +0.20, may → 冷战
engine.step_time(hours=24)    # OU decay toward baseline
```

**Theory:**
- **HMM:** Hidden states (relationship stages), observed events (user actions). Transition matrix modulated by intimacy/conflict levels.
- **Ornstein-Uhlenbeck:** Mean-reverting stochastic process. `dX = θ(μ - X)dt + σdW`. Models how intimacy/conflict drift toward baselines over time.

### 3. Emotion — VAD model

AI companion's internal emotional state modeled as 3D vector (Valence, Arousal, Dominance). Mapped to 11 discrete emotions.

```python
state = engine.emotion.current_state
# EmotionalState(😊 joy, V=+0.65, A=0.58, D=0.55)
```

| Dimension | Range | Meaning |
|-----------|-------|---------|
| Valence | -1 to +1 | Unhappy ↔ Happy |
| Arousal | 0 to 1 | Calm ↔ Excited |
| Dominance | 0 to 1 | Submissive ↔ Dominant |

**Theory:** [Russell's Circumplex Model (1980)](https://psycnet.apa.org/record/1981-25703-001) + [Mehrabian's PAD model (1996)](https://psycnet.apa.org/record/1996-97463-000). OU process for time decay, event-driven bumps for state changes.

## Install

```bash
# Base (Poisson + Bayesian from revive-companion)
pip install affective-longing

# With memory support (sentence-transformers + chromadb)
pip install affective-longing[memory]
```

## Quick Start

```python
from affective_longing import AffectiveLonging

engine = AffectiveLonging(seed=42)

# 1. Store memories
engine.remember("你喜欢下雨天", tags=["weather"])
engine.remember("你说过最喜欢吃草莓蛋糕", tags=["food"])

# 2. Observe events
engine.observe("reply_fast")
engine.observe("affection")

# 3. Let time pass
engine.step_time(hours=12)

# 4. Tick with context
result = engine.tick(context="今天下雨了")

print(f"Base probability:    {result.base_probability:.1%}")
print(f"Memory trigger:      {result.memory_trigger}")
print(f"Similarity:          {result.memory_similarity:.3f}")
print(f"Boosted probability: {result.boosted_probability:.1%}")
print(f"Relationship:        {result.relationship_stage.value}")
print(f"Emotion:             {result.emotional_state.emoji} {result.emotional_state.emotion.value}")

if result.should_send:
    send_message(result.prompt)
    engine.record_send()
```

## API Reference

### AffectiveLonging

```python
engine = AffectiveLonging(
    memory_persist_dir="./companion_memory_db",  # Where to store embeddings
    relationship_seed=None,                       # For reproducibility
    emotion_seed=None,
    **kwargs                                      # Passed to PoissonLove
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `remember(text, tags, **metadata)` | Store a memory. Returns memory ID. |
| `observe(event)` | Update relationship + emotion. Events: `reply_fast`, `reply_slow`, `no_reply`, `long_silence`, `affection`, `fight`, `apology`, `initiate`, `reject`, `long_message` |
| `step_time(hours)` | Advance time — OU decay on all dimensions |
| `tick(now, context)` | Full pipeline. Returns `AffectiveResult`. |
| `record_reply(**kwargs)` | Record user reply (passthrough to base) |
| `record_send()` | Record that we sent (passthrough to base) |
| `get_state()` | Snapshot of all 3 layers |

### AffectiveResult

```python
@dataclass
class AffectiveResult:
    # Decision
    should_send: bool
    base_probability: float

    # Memory
    memory_trigger: str | None
    memory_similarity: float
    longing_boost: float
    boosted_probability: float

    # Relationship
    relationship_stage: Stage
    intimacy: float
    conflict: float

    # Emotion
    emotional_state: EmotionalState

    # Output
    prompt: str
    reason: str
```

### Events

| Event | Description | Intimacy | Conflict |
|-------|-------------|----------|----------|
| `reply_fast` | User replied quickly | +0.05 | -0.02 |
| `reply_slow` | User replied slowly | -0.02 | +0.01 |
| `no_reply` | User didn't reply | -0.05 | +0.03 |
| `long_silence` | No contact >24h | -0.10 | +0.05 |
| `affection` | User showed warmth | +0.10 | -0.05 |
| `fight` | Conflict | -0.15 | +0.20 |
| `apology` | Someone apologized | +0.05 | -0.15 |
| `initiate` | User initiated contact | +0.08 | -0.02 |
| `reject` | User rejected us | -0.12 | +0.10 |

## Examples

Run the quickstarts:

```bash
# Memory triggers
python examples/quickstart.py

# Relationship state machine
python examples/quickstart_relationship.py

# VAD emotion engine
python examples/quickstart_emotion.py

# Full integration
python examples/quickstart_integrated.py
```

## Tests

```bash
pip install -e ".[memory,test]"
pytest tests/ -v
```

62 tests covering all modules.

## Theoretical Foundations

| Module | Theory | Reference |
|--------|--------|-----------|
| Memory | Sentence embeddings | [Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084) |
| Memory | Forgetting curve | [Ebbinghaus, 1885](https://en.wikipedia.org/wiki/Forgetting_curve) |
| Relationship | Hidden Markov Model | [Rabiner, 1989](https://ieeexplore.ieee.org/document/18626) |
| Relationship | Ornstein-Uhlenbeck | [Uhlenbeck & Ornstein, 1930](https://journals.aps.org/pr/abstract/10.1103/PhysRev.36.823) |
| Emotion | Circumplex Model | [Russell, 1980](https://psycnet.apa.org/record/1981-25703-001) |
| Emotion | PAD Model | [Mehrabian, 1996](https://psycnet.apa.org/record/1996-97463-000) |
| Base | Poisson Process | [Poisson, 1837](https://en.wikipedia.org/wiki/Poisson_point_process) |
| Base | Bayesian Inference | [Bayes, 1763](https://en.wikipedia.org/wiki/Bayesian_inference) |
| Base | Information Gain | [Shannon, 1948](https://en.wikipedia.org/wiki/Information_gain) |

## License

MIT
