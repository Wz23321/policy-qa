"""会话管理：记住多轮对话的上下文，并落盘保存。

为什么要限制轮数：模型上下文有限，聊太久会超长且变慢。
策略是只保留最近若干轮，超出的由 compress 模块压缩成摘要。

为什么要落盘：进程重启后会话还在，用户刷新页面不会断。
"""

import json
import time
import uuid
from collections import OrderedDict

from . import config


class SessionStore:
    def __init__(self, max_sessions=200, ttl_sec=3600, path=None):
        self._sessions = OrderedDict()   # sid -> {"history": [...], "ts": float}
        self.max_sessions = max_sessions
        self.ttl = ttl_sec
        self.path = path if path is not None else config.SESSION_FILE
        self._load()

    # ---------- 落盘 ----------

    def _load(self):
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return            # 文件坏了就当没有，不能让服务起不来

        now = time.time()
        for sid, s in raw.items():
            if now - s.get("ts", 0) <= self.ttl:
                self._sessions[sid] = {
                    "history": s.get("history", []),
                    "summary": s.get("summary", ""),
                    "ts": s.get("ts", now),
                }

    def _save(self):
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {sid: s for sid, s in self._sessions.items()}
            tmp = self.path.with_suffix(".tmp")
            # 先写临时文件再改名，避免写到一半进程挂了留下坏文件
            tmp.write_text(json.dumps(data, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass          # 存不下不影响本次对话

    # ---------- 接口 ----------

    def new_id(self):
        return uuid.uuid4().hex[:16]

    def get(self, sid):
        """取会话历史；不存在或过期返回空列表。"""
        s = self._sessions.get(sid)
        if not s:
            return []
        if time.time() - s["ts"] > self.ttl:
            self._sessions.pop(sid, None)
            self._save()
            return []
        s["ts"] = time.time()
        self._sessions.move_to_end(sid)
        return s["history"]

    def set_history(self, sid, history):
        """整体替换历史（压缩后用）。"""
        s = self._sessions.get(sid)
        if s is None:
            s = {"history": [], "ts": time.time()}
            self._sessions[sid] = s
        s["history"] = history
        s["ts"] = time.time()
        self._save()

    def append(self, sid, user_msg, assistant_msg):
        """追加一轮对话。超长时由 compress 模块处理，这里只做兜底截断。"""
        s = self._sessions.get(sid)
        if s is None:
            s = {"history": [], "summary": "", "ts": time.time()}
            self._sessions[sid] = s

        s["history"].append({"role": "user", "content": user_msg})
        s["history"].append({"role": "assistant", "content": assistant_msg})

        # 兜底：万一压缩失败，也不能让 history 无限涨
        hard_limit = config.COMPRESS_AFTER * 2 + 8
        if len(s["history"]) > hard_limit:
            s["history"] = s["history"][-hard_limit:]

        s["ts"] = time.time()
        self._sessions.move_to_end(sid)

        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

        self._save()

    def needs_compress(self, sid):
        s = self._sessions.get(sid)
        return bool(s) and len(s["history"]) > config.COMPRESS_AFTER * 2

    def get_summary(self, sid):
        s = self._sessions.get(sid)
        return s.get("summary", "") if s else ""

    def set_summary(self, sid, summary):
        s = self._sessions.get(sid)
        if s is not None:
            s["summary"] = summary
            self._save()

    def history_ref(self, sid):
        """拿到内部 history 列表的引用，供压缩模块原地修改。"""
        s = self._sessions.get(sid)
        return s["history"] if s else None

    def clear(self, sid):
        self._sessions.pop(sid, None)
        self._save()

    def count(self):
        return len(self._sessions)

    def purge_expired(self):
        """清掉过期会话，返回清掉的数量。"""
        now = time.time()
        dead = [sid for sid, s in self._sessions.items()
                if now - s["ts"] > self.ttl]
        for sid in dead:
            self._sessions.pop(sid, None)
        if dead:
            self._save()
        return len(dead)


STORE = SessionStore()