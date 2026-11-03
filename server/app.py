"""政策问答服务：对外提供 HTTP 接口，别的程序可以直接调。

启动:
    python -m uvicorn server.app:app --port 8100
"""

import contextlib
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from core import agent, cache, chunker, config, rag, retriever, session  # noqa: E402

STATIC = pathlib.Path(__file__).resolve().parent / "static"

app = FastAPI(title="政策问答服务", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_state = {"index": None}


def get_index():
    if _state["index"] is None:
        if not config.INDEX.exists():
            raise HTTPException(503, "索引还没建，先跑 python scripts/build_index.py")
        _state["index"] = retriever.load()
    return _state["index"]


class AskIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    session_id: str | None = Field(default=None, max_length=64,
                                   description="传了就是多轮对话，不传就是单轮")


class Source(BaseModel):
    doc: str
    doc_no: str
    org: str
    date: str
    region: str = "全国"
    invalid_date: str = ""
    score: float
    text: str


class AskOut(BaseModel):
    answer: str
    sources: list[Source]
    used_tokens: int
    session_id: str | None = None
    cached: bool = False
    search_query: str = ""


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    idx = _state["index"]
    return {
        "status": "ok",
        "chunks": len(idx["chunks"]) if idx else 0,
        "docs": len({c["doc"] for c in idx["chunks"]}) if idx else 0,
        "sessions": session.STORE.count(),
        "cache": cache.QUERY_CACHE.stats(),
    }


@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    # 传了 session_id 就取出上文，实现多轮
    sid = body.session_id
    history = session.STORE.get(sid) if sid else None

    result = rag.ask(body.question, get_index(), k=body.top_k, history=history)

    if sid:
        session.STORE.append(sid, body.question, result["answer"])

    return AskOut(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        used_tokens=result["used"],
        session_id=sid,
        cached=result.get("cached", False),
        search_query=result.get("search_query", ""),
    )


class AgentOut(BaseModel):
    answer: str
    steps: list[dict]
    used_tokens: int
    session_id: str | None = None


@app.post("/agent", response_model=AgentOut)
def agent_ask(body: AskIn):
    """Agent 模式：模型自己决定调哪些工具。"""
    sid = body.session_id
    history = session.STORE.get(sid) if sid else None

    result = agent.ask(body.question, get_index(), history=history)

    if sid:
        session.STORE.append(sid, body.question, result["answer"])

    return AgentOut(
        answer=result["answer"],
        steps=result["steps"],
        used_tokens=result["used"],
        session_id=sid,
    )


@app.post("/session/new")
def new_session():
    """新建一个会话，返回 session_id。多轮对话时把它带上。"""
    return {"session_id": session.STORE.new_id()}


@app.delete("/session/{sid}")
def clear_session(sid: str):
    session.STORE.clear(sid)
    return {"cleared": sid}


@app.post("/ingest")
def ingest():
    """重新扫描 data/raw 并重建索引。"""
    chunks = chunker.build()
    if not chunks:
        raise HTTPException(400, "data/raw 里没有 txt 文件")
    index = retriever.build_index(chunks)
    retriever.save(index)
    _state["index"] = index
    return {
        "chunks": len(chunks),
        "docs": len({c["doc"] for c in chunks}),
        "vocab": len(index["idf"]),
    }