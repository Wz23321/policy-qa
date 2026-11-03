"""政策问答服务的 Python 客户端。

用法:
    from sdk import PolicyClient
    c = PolicyClient("http://127.0.0.1:8100")

    # 单轮
    print(c.ask("就业见习补贴是给谁的？").answer)

    # 多轮
    sid = c.new_session()
    c.ask("创业担保贷款能贷多少？", session_id=sid)
    c.ask("那期限呢？", session_id=sid)      # 会记得上一句在说什么
"""

from dataclasses import dataclass, field

import requests


@dataclass
class Source:
    doc: str
    doc_no: str
    org: str
    date: str
    score: float
    text: str
    region: str = "全国"
    invalid_date: str = ""


@dataclass
class Answer:
    answer: str
    sources: list = field(default_factory=list)
    used_tokens: int = 0
    cached: bool = False
    search_query: str = ""

    def __str__(self):
        return self.answer

    def refs(self):
        """把来源整理成"文号 | 标题"，方便打印。"""
        return [f"{s.doc_no or '无文号'} | {s.doc}" for s in self.sources]


@dataclass
class AgentAnswer:
    answer: str
    steps: list = field(default_factory=list)
    used_tokens: int = 0

    def __str__(self):
        return self.answer

    def tools_used(self):
        return [s["tool"] for s in self.steps]


class PolicyClient:
    def __init__(self, base_url="http://127.0.0.1:8100", timeout=90):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def health(self):
        return requests.get(f"{self.base}/health", timeout=10).json()

    def new_session(self):
        """开一个新会话，返回 session_id。"""
        r = requests.post(f"{self.base}/session/new", timeout=10)
        r.raise_for_status()
        return r.json()["session_id"]

    def ask(self, question, top_k=5, session_id=None):
        payload = {"question": question, "top_k": top_k}
        if session_id:
            payload["session_id"] = session_id

        r = requests.post(f"{self.base}/ask", json=payload, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return Answer(
            answer=d["answer"],
            sources=[Source(**s) for s in d["sources"]],
            used_tokens=d.get("used_tokens", 0),
            cached=d.get("cached", False),
            search_query=d.get("search_query", ""),
        )

    def agent(self, question, session_id=None):
        """Agent 模式：模型自己决定调检索、读全文还是算数。"""
        payload = {"question": question}
        if session_id:
            payload["session_id"] = session_id

        r = requests.post(f"{self.base}/agent", json=payload, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return AgentAnswer(
            answer=d["answer"],
            steps=d.get("steps", []),
            used_tokens=d.get("used_tokens", 0),
        )

    def ingest(self):
        """让服务重新扫描 data/raw 并重建索引。"""
        r = requests.post(f"{self.base}/ingest", timeout=300)
        r.raise_for_status()
        return r.json()