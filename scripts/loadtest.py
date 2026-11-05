"""并发测试：测检索链路在不同并发下的吞吐和延迟。

不调用大模型——只测本地检索（查询向量化 + 余弦相似度），
这才是服务端的 CPU 瓶颈所在。

用法:
    python scripts/loadtest.py            # 默认 1/4/8/16 并发
    python scripts/loadtest.py 32         # 指定最大并发
"""

import statistics
import sys
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core import retriever  # noqa: E402

QUESTIONS = [
    "一次性求职补贴谁能领", "就业见习补贴是给谁的", "创业担保贷款额度",
    "灵活就业社保补贴标准", "河南就业见习补贴", "小微企业贷款多少",
    "基层就业有什么支持", "应征入伍有什么优惠", "职业培训补贴",
    "困难毕业生帮扶", "城乡社区岗位", "科研助理岗位",
]


def timed_search(index, q):
    start = time.perf_counter()
    retriever.search(index, q, top=6)
    return (time.perf_counter() - start) * 1000


def warmup(index):
    """先把模型跑热，第一次调用包含初始化开销。"""
    retriever.search(index, "热身", top=3)


def run_level(index, workers, total):
    """固定并发 worker 数，跑 total 次检索。"""
    jobs = [QUESTIONS[i % len(QUESTIONS)] for i in range(total)]

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        latencies = list(pool.map(lambda q: timed_search(index, q), jobs))
    elapsed = time.perf_counter() - start

    return {
        "workers": workers,
        "total": total,
        "elapsed": elapsed,
        "qps": total / elapsed,
        "p50": statistics.median(latencies),
        "p90": sorted(latencies)[int(len(latencies) * 0.9) - 1],
        "p99": sorted(latencies)[int(len(latencies) * 0.99) - 1],
        "max": max(latencies),
    }


def main(argv):
    top_workers = int(argv[0]) if argv else 16

    print("加载索引…")
    index = retriever.load()
    chunks = len(index["chunks"])
    print(f"语料 {chunks} 块\n")

    print("热身中…")
    warmup(index)

    levels = [1, 4, 8, 16]
    levels = [w for w in levels if w <= top_workers]

    print(f"\n{'并发':>4} {'QPS':>8} {'p50':>8} {'p90':>8} {'p99':>8} {'最大':>8}")
    print("-" * 50)

    results = []
    for w in levels:
        total = max(w * 8, 40)          # 每个 worker 至少跑 8 次
        r = run_level(index, w, total)
        results.append(r)
        print(f"{r['workers']:>4} {r['qps']:>8.1f} {r['p50']:>7.0f}ms "
              f"{r['p90']:>7.0f}ms {r['p99']:>7.0f}ms {r['max']:>7.0f}ms")

    print("-" * 50)
    best = max(results, key=lambda r: r["qps"])
    print(f"\n最高吞吐 {best['qps']:.1f} QPS（{best['workers']} 并发）")
    print(f"单次检索中位 {results[0]['p50']:.1f}ms")

    # 并发加不上去才说明真的卡在 CPU 上
    if len(results) > 1:
        first, last = results[0], results[-1]
        growth = last["qps"] / max(first["qps"], 0.01)
        if growth < 2:
            print(f"\n注意：{last['workers']} 并发时吞吐只有单并发时的 {growth:.1f} 倍，")
            print("      说明检索是纯 CPU 密集型，加并发只是让请求互相抢核。")
            print("      要么加机器/多进程，要么换带索引的向量库（FAISS/HNSW）。")
        else:
            print(f"\n并发扩展比：{last['workers']} 并发相对单并发 {growth:.1f}x")


if __name__ == "__main__":
    main(sys.argv[1:])