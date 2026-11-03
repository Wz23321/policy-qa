"""性能基准：测单次问答的延迟、token 消耗、缓存效果。

跑法:
    python scripts/bench.py          # 默认 15 条
    python scripts/bench.py 30       # 指定条数

输出中位数和 p90/p99 延迟，以及检索与生成各占多少。
"""

import statistics
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core import cache, rag, retriever  # noqa: E402

QUESTIONS = [
    "一次性求职补贴哪些人可以领？",
    "就业见习补贴是发给谁的？",
    "创业担保贷款个人最高能贷多少钱？",
    "小微企业创业担保贷款最高多少？",
    "创业担保贷款的期限是多久？",
    "创业担保贷款财政贴息多少？",
    "灵活就业的高校毕业生社保补贴标准是什么？",
    "灵活就业社保补贴能领多久？",
    "一次性创业补贴需要什么条件？",
    "就业见习补贴的政策期限到什么时候？",
    "高校毕业生到基层就业有什么支持政策？",
    "高校毕业生应征入伍有什么优惠？",
    "科研助理岗位是做什么的？",
    "城乡社区有哪些岗位给高校毕业生？",
    "困难毕业生就业有什么帮扶？",
    "职业培训补贴怎么申领？",
    "职业技能评价补贴是什么？",
    "国有企业招高校毕业生有什么政策？",
    "创业孵化基地有什么支持措施？",
    "毕业生档案应该怎么处理？",
]


def pct(values, p):
    """取第 p 百分位（p 用 0-100）。"""
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def main(n):
    index = retriever.load()
    questions = QUESTIONS[:n]

    total_ms, retrieve_ms, tokens = [], [], []
    failures = 0

    print(f"跑 {len(questions)} 条问题…\n")

    for i, q in enumerate(questions, 1):
        # 单独量一次检索耗时
        t0 = time.perf_counter()
        retriever.search(index, q, top=5)
        t1 = time.perf_counter()

        start = time.perf_counter()
        try:
            r = rag.ask(q, index)
        except Exception as e:
            failures += 1
            print(f"  {i}. 失败: {e}")
            continue
        end = time.perf_counter()

        retrieve_ms.append((t1 - t0) * 1000)
        total_ms.append((end - start) * 1000)
        tokens.append(r["used"])

        print(f"  {i:2d}. {q[:24]:<26} 总 {total_ms[-1]:6.0f}ms  "
              f"检索 {retrieve_ms[-1]:5.1f}ms  token {tokens[-1]}")

    if not total_ms:
        print("没有成功的请求")
        return

    gen_ms = [t - r for t, r in zip(total_ms, retrieve_ms)]

    print("\n" + "=" * 52)
    print(f"样本数        {len(total_ms)}（失败 {failures}）")
    print(f"总延迟  中位  {statistics.median(total_ms):7.0f} ms"
          f"   p90 {pct(total_ms, 90):7.0f} ms"
          f"   p99 {pct(total_ms, 99):7.0f} ms")
    print(f"  ├ 检索      {statistics.median(retrieve_ms):7.1f} ms（中位）")
    print(f"  └ 生成      {statistics.median(gen_ms):7.0f} ms（中位，调模型）")
    print(f"单次 token   中位 {statistics.median(tokens):.0f}"
          f"   合计 {sum(tokens)}")
    print("=" * 52)

    # 缓存效果：同一问题连问 5 次
    print("\n缓存测试：同一问题连问 5 次")
    cache.QUERY_CACHE.clear()
    q = questions[0]
    times = []
    for _ in range(5):
        s = time.perf_counter()
        r = rag.ask(q, index)
        times.append((time.perf_counter() - s) * 1000)
    print(f"  第 1 次（未命中）{times[0]:.0f} ms")
    print(f"  第 2-5 次（命中）平均 {statistics.mean(times[1:]):.1f} ms")
    print(f"  加速比           {times[0] / max(statistics.mean(times[1:]), 0.01):.0f}x")
    print(f"  缓存统计         {cache.QUERY_CACHE.stats()}")

    # 成本估算
    print("\n成本估算（按 deepseek-chat 约 ¥2/百万 token 输入价粗算）")
    avg = statistics.median(tokens)
    print(f"  单次约 {avg:.0f} token ≈ ¥{avg / 1_000_000 * 2:.6f}")
    print(f"  1000 次约 ¥{avg * 1000 / 1_000_000 * 2:.2f}")


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    main(count)