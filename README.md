# affective-longing 🧠💫

**Emotional extension for AI companions — beyond timing, into feeling.**

Built on top of [revive-companion](https://github.com/pearthink123/revive-companion).

## What's new (vs base)

| Feature | revive-companion | affective-longing |
|---------|-----------------|-------------------|
| Poisson timing | ✅ | ✅ (inherited) |
| Bayesian state | ✅ | ✅ (inherited) |
| Memory triggers | ❌ | ✅ |
| Relationship states | ❌ | ✅ |
| AI self-emotion | ❌ | ✅ |
| Cost/restraint | ❌ | ✅ |

## Install

```bash
pip install affective-longing
```

## Quick Start

```python
from affective_longing import AffectiveLonging

engine = AffectiveLonging()

# Store memories from conversations
engine.remember("你说你喜欢下雨天", tags=["weather", "preference"])
engine.remember("我们第一次见面是在咖啡店", tags=["place", "milestone"])

# Tick — now longing can be triggered by memory similarity
result = engine.tick(
    context="今天下雨了"  # triggers memory match → boost longing
)

if result.should_send:
    send(result.prompt)
```

## Modules

- **memory/** — Store & retrieve conversations, embedding similarity search
- **emotion/** — AI self-emotion vector (loneliness, happiness, anxiety)
- **relationship/** — State machine (honeymoon / cold war / stable / reunited)

## Status

🚧 Early development. API will change.

## License

MIT
