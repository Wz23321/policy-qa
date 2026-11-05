"""RAG 主流程：检索 → 拼材料 → 让模型照着答。"""

from . import cache, config, llm, retriever

SYSTEM = """你是大学生就业创业政策助手。规则：

1. 只能依据下面提供的政策原文回答，不要用你自己的知识补充。
2. 每条结论后面用 [编号] 标注出处，编号对应材料的序号。
3. 如果材料里没有能回答这个问题的内容，直接说"提供的政策文件里没有找到相关内容"，不要猜。
4. 回答要具体：涉及金额、期限、条件、办理部门的，照原文写清楚。
5. 如果材料给的是**计算方式**而不是固定数额（例如"当地最低工资标准的70%"
   "不超过实际缴费的2/3"），这仍然是有效答案，必须照实说明计算方式，
   不能因为"没有具体数字"就说找不到。
6. 注意区分国家级政策和地方政策。如果用户问的是某个地区，而材料里只有国家级政策、
   没有该地区的政策，必须明确说明"没有该地区的具体政策，以下是国家层面的规定"，
   不要把国家政策当成地方政策回答。
7. 用中文回答，简洁分点。"""


def get_client():
    """保留这个入口，内部走 llm 模块（带重试）。"""
    return llm.get_client()


REWRITE_PROMPT = """把用户最新的问题改写成一个不依赖上文、能独立检索的句子。

要求：
- 补全代词和省略的主语。比如上文在聊"创业担保贷款"，新问题"期限多久"要改写成"创业担保贷款的期限是多久"
- 只输出改写后的句子，不要解释，不要加引号
- 如果本来就是独立问题，原样输出

对话历史：
{history}

最新问题：{question}

改写后："""


def rewrite_query(question, history):
    """把带指代的追问改写成能独立检索的问题。

    没有历史时直接返回原问题，不多花一次调用。
    """
    if not history:
        return question

    lines = []
    for m in history[-6:]:
        who = "用户" if m["role"] == "user" else "助手"
        lines.append(f"{who}：{m['content'][:200]}")

    try:
        resp = llm.chat(
            [{"role": "user",
              "content": REWRITE_PROMPT.format(
                  history="\n".join(lines), question=question)}],
            temperature=0,
            max_tokens=100,
            attempts=2,          # 改写失败不重试太多次，直接用原问题
        )
        new_q = resp.choices[0].message.content.strip().strip('"“”')
        return new_q or question
    except Exception:
        return question     # 改写失败就用原问题，不影响主流程


def format_materials(hits):
    parts = []
    for i, (fused, vsim, c) in enumerate(hits, 1):
        head = f"[{i}] 《{c['doc']}》"
        if c.get("region") and c["region"] != "全国":
            head += f"（{c['region']}）"
        if c["doc_no"]:
            head += f" {c['doc_no']}"
        if c["date"]:
            head += f"（{c['date']} 发布"
            if c.get("invalid_date") and c["invalid_date"] != "未标注":
                head += f"，{c['invalid_date']} 失效"
            head += "）"
        parts.append(f"{head}\n{c['text']}")
    return "\n\n".join(parts)


def diversify(hits, per_doc=3):
    """同一份文件最多取 per_doc 块，避免来源全是同一篇。

    这个上限不能太小：政策文件里同一份的相邻切块常常都排在前几名，
    上限设成 2 时，排名第 3 但内容关键的那块会被挤掉，模型就会说"没找到"。
    """
    seen, out = {}, []
    for fused, vsim, c in hits:
        n = seen.get(c["doc_no"] or c["doc"], 0)
        if n >= per_doc:
            continue
        seen[c["doc_no"] or c["doc"]] = n + 1
        out.append((fused, vsim, c))
    return out


def region_rerank(hits, question, regions):
    """按地区重排：问了某个省就偏向该省，没问就偏向国家级。

    不做直接过滤——万一地方政策才是唯一相关的，过滤掉就什么都查不到了。
    只调分数，让相关性接近时地区优先。
    """
    target = next((r for r in regions if r != "全国" and r in question), None)

    out = []
    for fused, vsim, c in hits:
        reg = c.get("region", "全国")
        if target:
            factor = 1.30 if reg == target else (1.10 if reg == "全国" else 1.0)
        else:
            factor = 1.15 if reg == "全国" else 1.0
        out.append((fused * factor, vsim, c))

    out.sort(key=lambda x: -x[0])
    return out



def _source(vsim, c):
    """把检索块转成对外的来源结构。"""
    return {
        "doc": c["doc"],
        "doc_no": c["doc_no"],
        "org": c["org"],
        "date": c["date"],
        "region": c.get("region", "全国"),
        "invalid_date": c.get("invalid_date", ""),
        "score": float(round(float(vsim), 4)),   # numpy 的 float32 不能直接 json 序列化
        "text": c["text"][:300],
    }


def _retrieve(question, index, k, history, mode):
    """检索并整理来源。ask 和 ask_stream 共用。"""
    search_query = rewrite_query(question, history)

    hits = retriever.search(index, search_query, top=k * 3, mode=mode)
    # 关键词模式的分数不是余弦相似度，量纲不同，不套用向量阈值
    floor = 0.05 if mode == "keyword" else config.MIN_VEC_SIM
    hits = [h for h in hits if h[1] >= floor]

    regions = {c.get("region", "全国") for _, _, c in hits}
    hits = region_rerank(hits, search_query, regions)

    # 可选：用 cross-encoder 精排。默认关闭，原因见 core/rerank.py
    if config.ENABLE_RERANK and hits:
        from . import rerank as rerank_mod
        hits = rerank_mod.rerank(search_query,
                                 hits[:config.RERANK_CANDIDATES], top=k)
    else:
        hits = diversify(hits)[:k]

    sources = [_source(vsim, c) for _, vsim, c in hits]
    return search_query, hits, sources


REFUSAL_WORDS = ("没有找到", "未找到", "找不到", "无法回答", "不在政策库")


def _is_refusal(text):
    return any(k in text[:60] for k in REFUSAL_WORDS)


def _build_messages(question, hits, history):
    messages = [{"role": "system", "content": SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"政策材料：\n{format_materials(hits)}\n\n问题：{question}",
    })
    return messages


EMPTY_ANSWER = "提供的政策文件里没有找到相关内容。"


def ask(question, index, k=config.TOP_K, history=None, mode="vector"):
    """问一个问题，返回 {answer, sources, used, search_query, cached}"""
    # 单轮问答可以按问题缓存；多轮依赖上文，不能缓存
    cache_key = (mode, question.strip())
    if not history:
        hit = cache.QUERY_CACHE.get(cache_key)
        if hit is not None:
            # 命中缓存没有新的 token 消耗
            return {**hit, "cached": True, "used": 0}

    search_query, hits, sources = _retrieve(question, index, k, history, mode)

    if not hits:
        result = {"answer": EMPTY_ANSWER, "sources": [], "used": 0,
                  "search_query": search_query, "cached": False}
        if not history:
            cache.QUERY_CACHE.put(cache_key, result)
        return result

    resp = llm.chat(_build_messages(question, hits, history), temperature=0.2)
    answer = resp.choices[0].message.content.strip()

    # 模型说没找到时，别再把来源列出来，否则界面上自相矛盾
    if _is_refusal(answer):
        result = {"answer": answer, "sources": [],
                  "used": resp.usage.total_tokens,
                  "search_query": search_query, "cached": False}
        if not history:
            cache.QUERY_CACHE.put(cache_key, result)
        return result

    result = {
        "answer": answer,
        "sources": sources,
        "used": resp.usage.total_tokens,
        "search_query": search_query,
        "cached": False,
    }
    if not history:
        cache.QUERY_CACHE.put(cache_key, result)
    return result


def ask_stream(question, index, k=config.TOP_K, history=None, mode="vector"):
    """流式版本。逐段 yield：

    {"type": "meta",  "search_query": ...}
    {"type": "delta", "text": "..."}       多次
    {"type": "done",  "sources": [...], "used": n, "cached": bool}
    """
    cache_key = (mode, question.strip())
    if not history:
        hit = cache.QUERY_CACHE.get(cache_key)
        if hit is not None:
            yield {"type": "meta", "search_query": hit.get("search_query", "")}
            yield {"type": "delta", "text": hit["answer"]}
            yield {"type": "done", "sources": hit["sources"],
                   "used": 0, "cached": True}
            return

    search_query, hits, sources = _retrieve(question, index, k, history, mode)
    yield {"type": "meta", "search_query": search_query}

    if not hits:
        yield {"type": "delta", "text": EMPTY_ANSWER}
        yield {"type": "done", "sources": [], "used": 0, "cached": False}
        return

    stream = llm.chat(_build_messages(question, hits, history),
                      temperature=0.2, stream=True)

    parts = []
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content
        if piece:
            parts.append(piece)
            yield {"type": "delta", "text": piece}

    answer = "".join(parts).strip()
    refused = _is_refusal(answer)
    final_sources = [] if refused else sources

    if not history:
        cache.QUERY_CACHE.put(cache_key, {
            "answer": answer, "sources": final_sources,
            "used": 0, "search_query": search_query, "cached": False,
        })

    yield {"type": "done", "sources": final_sources,
           "used": 0, "cached": False}
