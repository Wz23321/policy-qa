"""Agent：让模型自己决定查政策、查原文、还是算数。

一次问答不再是"检索一次然后回答"，而是一个循环：
    想 → 调工具 → 看结果 → 还想再调就继续 → 最后给答案
"""

import ast
import json
import operator

from . import config, rag, retriever

MAX_STEPS = 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "在政策库里检索与查询相关的政策原文片段。不确定信息在哪份文件里时先调这个。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词，比如'就业见习补贴标准'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_policy",
            "description": "读取某份政策文件的完整内容。已知文号或文件名时用，比如想确认某条政策的前后文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_no": {"type": "string", "description": "发文字号，如'人社部发〔2026〕19号'；也可传文件名片段"},
                },
                "required": ["doc_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算四则运算。涉及金额换算、比例、补贴总额计算时用。只能算数，不能查资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "算式，如 '300000 * 0.5'"},
                },
                "required": ["expression"],
            },
        },
    },
]

SYSTEM = """你是大学生就业创业政策助手。你可以调用工具：

- search_policy：检索政策原文
- read_policy：读某份政策的全文
- calculate：做数学计算

工作方式：
1. 遇到需要政策依据的问题，先调 search_policy；不确定就换个关键词再查。
2. 需要上下文时用 read_policy 读全文。
3. 需要算钱时用 calculate，不要心算。
4. 只能依据工具返回的材料回答。查不到就说"提供的政策文件里没有找到相关内容"，不要猜。
5. 回答时用 [文件名/文号] 标注出处，涉及金额、期限、条件要写清楚。"""


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    """只允许数字和四则运算，不执行任何函数调用。"""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只支持数字")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("不支持的表达式")


def calculate(expression):
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"计算出错: {e}"


def read_policy(index, doc_no):
    """按文号或文件名片段找出整份文件。"""
    key = doc_no.strip()
    matched = [c for c in index["chunks"]
               if key in (c["doc_no"] or "") or key in c["doc"] or key in c["file"]]
    if not matched:
        return f"没有找到文号或文件名包含「{key}」的政策文件。"

    doc_name = matched[0]["doc"]
    same = [c for c in index["chunks"] if c["doc"] == doc_name]
    same.sort(key=lambda c: c["seq"])
    body = "\n".join(c["text"] for c in same)
    return f"《{doc_name}》{matched[0]['doc_no']}\n\n{body[:4000]}"


def run_tool(name, args, index):
    if name == "search_policy":
        query = args.get("query", "")
        hits = retriever.search(index, query, top=config.TOP_K * 3)
        regions = {c.get("region", "全国") for _, _, c in hits}
        hits = rag.region_rerank(hits, query, regions)[:config.TOP_K]
        if not hits:
            return "没有检索到相关内容。"
        out = []
        for i, (_, vsim, c) in enumerate(hits, 1):
            tag = "" if c.get("region", "全国") == "全国" else f"（{c['region']}）"
            out.append(f"[{i}] 《{c['doc']}》{tag}{c['doc_no']}（相关度 {vsim:.2f}）\n{c['text']}")
        return "\n\n".join(out)

    if name == "read_policy":
        return read_policy(index, args.get("doc_no", ""))

    if name == "calculate":
        return calculate(args.get("expression", ""))

    return f"未知工具: {name}"


def ask(question, index, verbose=False, history=None):
    """跑一个完整的 Agent 循环，返回 {answer, steps, used}"""
    client = rag.get_client()
    messages = [
        {"role": "system", "content": SYSTEM},
    ]
    if history:
        messages.extend(history[-config.MAX_TURNS * 2:])
    messages.append({"role": "user", "content": question})
    steps = []
    total_tokens = 0

    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=config.MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.2,
        )
        total_tokens += resp.usage.total_tokens
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return {
                "answer": (msg.content or "").strip(),
                "steps": steps,
                "used": total_tokens,
            }

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name,
                             "arguments": tc.function.arguments},
            } for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            result = run_tool(name, args, index)
            steps.append({"tool": name, "args": args, "result": result[:200]})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:6000],
            })

    return {
        "answer": "这个问题需要太多步检索，没能得出结论。",
        "steps": steps,
        "used": total_tokens,
    }