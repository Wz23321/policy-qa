"""新加模块的测试：重试、会话持久化、上下文压缩。"""

import sys
import pathlib
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core import config, llm, session  # noqa: E402


class TestRetry(unittest.TestCase):
    def test_backoff_grows(self):
        """退避时间要随尝试次数增长"""
        samples = [llm._backoff(0), llm._backoff(1), llm._backoff(2)]
        self.assertLess(samples[0], samples[1])
        self.assertLess(samples[1], samples[2])

    def test_backoff_has_jitter(self):
        """同一个次数多次取值不应该完全一样"""
        values = {round(llm._backoff(1), 6) for _ in range(10)}
        self.assertGreater(len(values), 1)

    def test_backoff_capped(self):
        """再多次也不能超过上限太多"""
        self.assertLessEqual(llm._backoff(20), config.RETRY_MAX_DELAY * 1.5)

    @patch("core.llm.time.sleep", return_value=None)
    @patch("core.llm._is_retryable", return_value=True)
    def test_succeeds_after_retries(self, _retryable, _sleep):
        calls = {"n": 0}

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        calls["n"] += 1
                        if calls["n"] < 3:
                            raise TimeoutError("boom")
                        return "ok"

        with patch("core.llm.get_client", return_value=FakeClient()):
            out = llm.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(out, "ok")
        self.assertEqual(calls["n"], 3)

    @patch("core.llm.time.sleep", return_value=None)
    @patch("core.llm._is_retryable", return_value=False)
    def test_no_retry_on_fatal(self, _retryable, _sleep):
        """不可重试的错误要立刻抛出，不能白等"""
        calls = {"n": 0}

        class FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        calls["n"] += 1
                        raise ValueError("bad request")

        with patch("core.llm.get_client", return_value=FakeClient()):
            with self.assertRaises(ValueError):
                llm.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(calls["n"], 1)


class TestSessionPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp()) / "sessions.json"

    def test_survives_restart(self):
        s1 = session.SessionStore(path=self.tmp)
        sid = s1.new_id()
        s1.append(sid, "问题一", "回答一")

        # 模拟进程重启：新建一个 store 指向同一个文件
        s2 = session.SessionStore(path=self.tmp)
        h = s2.get(sid)
        self.assertEqual(len(h), 2)
        self.assertEqual(h[0]["content"], "问题一")

    def test_summary_persists(self):
        s1 = session.SessionStore(path=self.tmp)
        sid = s1.new_id()
        s1.append(sid, "q", "a")
        s1.set_summary(sid, "这是摘要")

        s2 = session.SessionStore(path=self.tmp)
        self.assertEqual(s2.get_summary(sid), "这是摘要")

    def test_corrupt_file_does_not_crash(self):
        """文件坏了要能正常起来，不能把服务带崩"""
        self.tmp.parent.mkdir(parents=True, exist_ok=True)
        self.tmp.write_text("{ this is not json", encoding="utf-8")
        s = session.SessionStore(path=self.tmp)
        self.assertEqual(s.count(), 0)

    def test_needs_compress_threshold(self):
        s = session.SessionStore(path=self.tmp)
        sid = s.new_id()
        for i in range(config.COMPRESS_AFTER):
            s.append(sid, f"q{i}", f"a{i}")
        self.assertFalse(s.needs_compress(sid))
        s.append(sid, "one more", "ok")
        self.assertTrue(s.needs_compress(sid))


class TestCompress(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp()) / "sessions.json"
        self.store = session.SessionStore(path=self.tmp)

    def test_compress_reduces_history(self):
        from core import compress

        sid = self.store.new_id()
        for i in range(config.COMPRESS_AFTER + 2):
            self.store.append(sid, f"问题{i}", f"回答{i}")

        before = len(self.store.history_ref(sid))

        class FakeResp:
            class choices:
                @staticmethod
                def __len__():
                    return 1

        # 用假的 llm.chat 返回固定摘要
        class FakeMsg:
            content = "这是压缩后的摘要"

        class FakeChoice:
            message = FakeMsg()

        class FakeResp2:
            choices = [FakeChoice()]

        with patch("core.compress.llm.chat", return_value=FakeResp2()):
            ok = compress.maybe_compress(self.store, sid)

        self.assertTrue(ok)
        after = len(self.store.history_ref(sid))
        self.assertLess(after, before)
        self.assertEqual(after, config.MAX_TURNS * 2)
        self.assertIn("压缩", self.store.get_summary(sid))

    def test_compress_failure_falls_back_to_truncate(self):
        from core import compress

        sid = self.store.new_id()
        for i in range(config.COMPRESS_AFTER + 2):
            self.store.append(sid, f"问题{i}", f"回答{i}")

        with patch("core.compress.llm.chat", side_effect=RuntimeError("模型挂了")):
            ok = compress.maybe_compress(self.store, sid)

        self.assertFalse(ok)
        # 即使压缩失败，历史也不能无限增长
        self.assertEqual(len(self.store.history_ref(sid)), config.MAX_TURNS * 2)

    def test_build_history_includes_summary(self):
        from core import compress

        sid = self.store.new_id()
        self.store.append(sid, "q", "a")
        self.store.set_summary(sid, "用户是河南应届生")

        msgs = compress.build_history(self.store, sid)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("河南应届生", msgs[0]["content"])
        self.assertEqual(msgs[-1]["content"], "a")


if __name__ == "__main__":
    unittest.main(verbosity=2)