"""简易 LRU 缓存：同一个问题问第二次时直接返回，不再调模型。

只在单轮问答时启用——多轮对话答案依赖上文，缓存会出错。
"""

import time
from collections import OrderedDict


class LRUCache:
    def __init__(self, maxsize=256, ttl_sec=3600):
        self.data = OrderedDict()    # key -> (value, expire_ts)
        self.maxsize = maxsize
        self.ttl = ttl_sec
        self.hits = 0
        self.misses = 0

    def get(self, key):
        item = self.data.get(key)
        if item is None:
            self.misses += 1
            return None
        value, expire = item
        if time.time() > expire:
            self.data.pop(key, None)
            self.misses += 1
            return None
        self.data.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key, value):
        self.data[key] = (value, time.time() + self.ttl)
        self.data.move_to_end(key)
        while len(self.data) > self.maxsize:
            self.data.popitem(last=False)

    def clear(self):
        self.data.clear()
        self.hits = self.misses = 0

    def stats(self):
        total = self.hits + self.misses
        return {
            "size": len(self.data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


QUERY_CACHE = LRUCache()