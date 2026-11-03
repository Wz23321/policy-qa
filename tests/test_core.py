"""核心模块的单元测试。

跑法:
    python -m unittest discover -s tests -v
"""

import sys
import pathlib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core import cache, chunker, session  # noqa: E402


class TestChunker(unittest.TestCase):
    def test_split_by_article(self):
        """按「一、」「1.」切，而不是按字数切"""
        text = (
            "一、总则\n"
            "为做好高校毕业生就业工作，根据国家有关规定，制定本通知，各地遵照执行。\n"
            "1.第一条补贴\n"
            "政策内容：对吸纳离校两年内未就业高校毕业生参加见习的单位，给予就业见习补贴。"
        )
        parts = chunker.split(text)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].startswith("一、总则"))
        self.assertTrue(parts[1].startswith("1.第一条补贴"))
        self.assertIn("政策内容", parts[1])

    def test_split_keeps_paragraph_intact(self):
        """一条政策不能被切到两半"""
        body = ("1.就业见习补贴\n"
                "政策内容：对吸纳离校2年内未就业高校毕业生参加见习的单位，给予就业见习补贴，"
                "用于支付见习人员基本生活费、办理人身意外伤害保险。")
        parts = chunker.split(body)
        self.assertEqual(len(parts), 1)
        self.assertIn("基本生活费", parts[0])
        self.assertIn("人身意外伤害保险", parts[0])

    def test_split_long_block_falls_back(self):
        """超长块按长度兜底切分"""
        long_text = "一、长段落\n" + "这是一句话。" * 200
        parts = chunker.split(long_text, size=200, overlap=20)
        self.assertGreater(len(parts), 1)
        for p in parts:
            self.assertLess(len(p), 400)

    def test_short_chunks_dropped(self):
        """太碎的块丢掉"""
        parts = chunker.split("一、题\n短", size=100)
        self.assertEqual(parts, [])


class TestLRUCache(unittest.TestCase):
    def test_hit_and_miss(self):
        c = cache.LRUCache(maxsize=10, ttl_sec=60)
        self.assertIsNone(c.get("a"))
        c.put("a", 1)
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(c.stats()["hits"], 1)
        self.assertEqual(c.stats()["misses"], 1)

    def test_eviction(self):
        c = cache.LRUCache(maxsize=2, ttl_sec=60)
        c.put("a", 1)
        c.put("b", 2)
        c.put("c", 3)          # 淘汰最久未用的 a
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.get("b"), 2)
        self.assertEqual(c.get("c"), 3)

    def test_lru_order_updated_on_get(self):
        c = cache.LRUCache(maxsize=2, ttl_sec=60)
        c.put("a", 1)
        c.put("b", 2)
        c.get("a")             # a 变成最近使用
        c.put("c", 3)          # 应该淘汰 b
        self.assertEqual(c.get("a"), 1)
        self.assertIsNone(c.get("b"))

    def test_ttl_expiry(self):
        c = cache.LRUCache(maxsize=10, ttl_sec=0)
        c.put("a", 1)
        self.assertIsNone(c.get("a"))


class TestSession(unittest.TestCase):
    def test_history_accumulates(self):
        s = session.SessionStore()
        sid = s.new_id()
        s.append(sid, "问题1", "回答1")
        s.append(sid, "问题2", "回答2")
        h = s.get(sid)
        self.assertEqual(len(h), 4)          # 两轮 = 4 条消息
        self.assertEqual(h[0]["role"], "user")
        self.assertEqual(h[1]["role"], "assistant")

    def test_history_truncated(self):
        s = session.SessionStore()
        sid = s.new_id()
        for i in range(20):
            s.append(sid, f"问题{i}", f"回答{i}")
        h = s.get(sid)
        self.assertLessEqual(len(h), session.config.MAX_TURNS * 2)

    def test_unknown_session_returns_empty(self):
        s = session.SessionStore()
        self.assertEqual(s.get("不存在的id"), [])

    def test_clear(self):
        s = session.SessionStore()
        sid = s.new_id()
        s.append(sid, "q", "a")
        s.clear(sid)
        self.assertEqual(s.get(sid), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)