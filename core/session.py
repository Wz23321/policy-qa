"""会话管理：记住多轮对话的上下文。

为什么要限制轮数：模型上下文有限，聊太久会超长且变慢。
策略是只保留最近若干轮，超出的丢掉。
"""

import time
import uuid
from collections import OrderedDict

from . import config


class SessionStore:
    def __init__(self, max_sessions=200, ttl_sec=3600):
        self._sessions = OrderedDict()   # sid -> {"history": [...], "ts": float}
        self.max_sessions = max_sessions
        self.ttl = ttl_sec

    def new_id(self):
        return uuid.uuid4().hex[:16]

    def get(self, sid):
        """取会话历史；不存在或过期返回空列表。"""
        s = self._sessions.get(sid)
        if not s:
            return []
        if time.time() - s["ts"] > self.ttl:
            self._sessions.pop(sid, None)
            return []
        s["ts"] = time.time()
        self._sessions.move_to_end(sid)
        return s["history"]

    def append(self, sid, user_msg, assistant_msg):
        """追加一轮对话，并裁剪到最近 N 轮。"""
        s = self._sessions.get(sid)
        if s is None:
            s = {"history": [], "ts": time.time()}
            self._sessions[sid] = s

        s["history"].append({"role": "user", "content": user_msg})
        s["history"].append({"role": "assistant", "content": assistant_msg})

        # 一轮 = 2 条消息
        limit = config.MAX_TURNS * 2
        if len(s["history"]) > limit:
            s["history"] = s["history"][-limit:]

        s["ts"] = time.time()
        self._sessions.move_to_end(sid)

        # 会话太多时淘汰最久没用的
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def clear(self, sid):
        self._sessions.pop(sid, None)

    def count(self):
        return len(self._sessions)


STORE = SessionStore()