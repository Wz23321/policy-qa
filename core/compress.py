"""上下文压缩：对话变长后把旧轮次压成摘要，只保留最近几轮原文。

为什么不是直接丢弃：用户第一句说的"我是河南的应届生"这种约束，
丢了之后后面几轮模型就不知道了。压成摘要能保住关键信息，
又不会让提示词无限变长。
"""

from . import config, llm

SUMMARY_PROMPT = """把下面的对话压缩成不超过 200 字的摘要。

必须保留：
- 用户问过的具体问题，尤其是政策名称、地区、金额、时间
- 已经给出的关键结论和文号
- 用户提到过的自身情况（学历、毕业年份、所在地等）

不要写"用户询问了…""助手回答了…"这种套话，直接写事实。

已有摘要（可能为空）：
{summary}

新增对话：
{history}

合并后的摘要："""


def _format(msgs):
    lines = []
    for m in msgs:
        who = "用户" if m["role"] == "user" else "助手"
        content = m["content"].replace("\n", " ")[:300]
        lines.append(f"{who}：{content}")
    return "\n".join(lines)


def maybe_compress(store, sid):
    """需要压缩就压缩，返回是否真的压了。压缩失败不影响主流程。"""
    if not store.needs_compress(sid):
        return False

    history = store.history_ref(sid)
    keep = config.MAX_TURNS * 2
    old = list(history[:-keep])
    if not old:
        return False

    try:
        resp = llm.chat(
            [{"role": "user",
              "content": SUMMARY_PROMPT.format(
                  summary=store.get_summary(sid) or "（无）",
                  history=_format(old))}],
            temperature=0,
            max_tokens=400,
            attempts=2,
        )
        summary = resp.choices[0].message.content.strip()
    except Exception:
        # 压缩失败就退化成直接截断，至少不会越堆越长
        store.set_history(sid, history[-keep:])
        return False

    store.set_summary(sid, summary)
    store.set_history(sid, history[-keep:])
    return True


def build_history(store, sid):
    """把摘要和最近几轮拼成给模型的消息列表。"""
    if not sid:
        return []

    messages = []
    summary = store.get_summary(sid)
    if summary:
        messages.append({
            "role": "system",
            "content": "以下是本次对话早些时候的内容摘要，供参考：\n" + summary,
        })
    messages.extend(store.get(sid))
    return messages