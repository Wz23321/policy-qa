"""命令行问一句，快速验证。"""

import sys

from core import config, rag, retriever


def main():
    index = retriever.load()
    print("政策问答（输入 q 退出）\n")

    while True:
        try:
            q = input("问: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() == "q":
            break

        r = rag.ask(q, index)
        print(f"\n答: {r['answer']}\n")
        if r["sources"]:
            print("来源:")
            for i, s in enumerate(r["sources"], 1):
                print(f"  [{i}] {s['doc_no'] or '无文号'}  {s['doc']}")
        print(f"(用了 {r['used']} token)\n")


if __name__ == "__main__":
    main()