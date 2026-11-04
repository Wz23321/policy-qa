"""一键演示：跑几个典型问答，把整条链路的效果打出来。

用法:
    python scripts/demo.py            # 用默认问题
    python scripts/demo.py --fast     # 只跑单次问答，不走 Agent

需要 .env 里有 DEEPSEEK_API_KEY，并且已经建过索引。
"""

import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core import agent, config, rag, retriever  # noqa: E402

DEMOS = [
    ("国家层面", "一次性求职补贴哪些人可以领？"),
    ("地方政策", "河南高校毕业生灵活就业社保补贴标准是多少？"),
    ("跨文件对比", "个人和小微企业的创业担保贷款额度分别是多少？"),
    ("多步推理", "想跟同学合伙开公司，启动资金不够怎么办？"),
    ("超纲拒答", "北京落户政策怎么样？"),
]


def hr(title=""):
    print("\n" + "=" * 66)
    if title:
        print(title)
        print("=" * 66)


def run_one(label, question, index, use_agent):
    print(f"\n【{label}】")
    print(f"问：{question}")

    start = time.perf_counter()
    if use_agent:
        r = agent.ask(question, index)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n答：{r['answer']}")
        if r["steps"]:
            print("\n调用记录：")
            for i, s in enumerate(r["steps"], 1):
                args = s["args"]
                q = args.get("query") or args.get("doc_no") or args.get("expression") or ""
                print(f"  {i}. {s['tool']}  {q}")
        print(f"\n耗时 {elapsed:.0f}ms   token {r['used']}")
    else:
        r = rag.ask(question, index)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n答：{r['answer']}")
        if r["sources"]:
            print("\n来源：")
            for i, s in enumerate(r["sources"], 1):
                region = "" if s["region"] == "全国" else f"[{s['region']}] "
                print(f"  [{i}] {region}{s['doc_no'] or '无文号'}  {s['doc'][:36]}")
        print(f"\n耗时 {elapsed:.0f}ms   token {r['used']}")


def main(argv):
    use_agent = "--fast" not in argv

    if not config.DEEPSEEK_KEY:
        print("没读到 DEEPSEEK_API_KEY，先在项目根目录建 .env 写上 key")
        return 1

    if not config.INDEX.exists():
        print("还没建索引，先跑：python scripts/build_index.py")
        return 1

    index = retriever.load()

    hr("政策问答服务 · 演示")
    print(f"语料：{len({c['doc'] for c in index['chunks']})} 份政策，"
          f"{len(index['chunks'])} 个切块")
    print(f"模式：{'Agent（模型自己调工具）' if use_agent else '直接检索'}")

    total = 0
    for label, q in DEMOS:
        run_one(label, q, index, use_agent)
        total += 1

    hr(f"演示结束，共 {total} 个问题")
    print("启动网页版：python -m uvicorn server.app:app --port 8100")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))