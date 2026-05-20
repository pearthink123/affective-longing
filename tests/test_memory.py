"""
Tests for MemoryStore and TriggerEngine.
"""
import pytest
import tempfile
import shutil
import os

from affective_longing.memory import MemoryStore, TriggerEngine


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(temp_dir):
    return MemoryStore(collection_name="test_memories", persist_dir=temp_dir)


class TestMemoryStore:
    def test_add_and_count(self, store):
        assert store.count() == 0
        store.add("我喜欢下雨天")
        assert store.count() == 1

    def test_add_with_metadata(self, store):
        mid = store.add("测试记忆", metadata={"type": "test", "score": 0.9})
        assert mid.startswith("mem_")

    def test_query_returns_similar(self, store):
        store.add("你喜欢下雨天", metadata={"type": "weather"})
        store.add("我们去咖啡店了", metadata={"type": "place"})
        store.add("今天天气真好", metadata={"type": "weather"})

        results = store.query("外面在下雨", top_k=2, similarity_threshold=0.3)
        assert len(results) > 0
        # The rain-related memory should be most similar
        assert "雨" in results[0]["text"]

    def test_query_respects_threshold(self, store):
        store.add("完全无关的内容关于量子物理")
        results = store.query("今天吃什么", top_k=5, similarity_threshold=0.9)
        # Should return nothing due to high threshold
        assert len(results) == 0

    def test_query_empty_store(self, store):
        results = store.query("任何内容")
        assert results == []


class TestTriggerEngine:
    def test_no_trigger_without_memories(self, store):
        engine = TriggerEngine(store)
        result = engine.compute_trigger_score("今天下雨了")
        assert result["trigger_score"] == 0.0
        assert result["memories"] == []

    def test_trigger_with_similar_memory(self, store):
        store.add("你喜欢下雨天")
        engine = TriggerEngine(store, similarity_threshold=0.5)
        result = engine.compute_trigger_score("今天下雨了")
        assert result["trigger_score"] > 0
        assert len(result["memories"]) > 0

    def test_boost_longing(self, store):
        store.add("你喜欢下雨天")
        engine = TriggerEngine(store, similarity_threshold=0.5)
        base_prob = 0.15
        boosted = engine.boost_longing(base_prob, "今天下雨了")
        assert boosted > base_prob
        assert boosted <= 1.0

    def test_no_boost_without_trigger(self, store):
        store.add("完全无关的量子物理内容")
        engine = TriggerEngine(store, similarity_threshold=0.5)
        base_prob = 0.15
        boosted = engine.boost_longing(base_prob, "今天吃什么")
        assert boosted == base_prob  # No boost
