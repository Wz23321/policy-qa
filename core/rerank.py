"""Rerank：用 cross-encoder 对粗召回结果重新排序。

原理：向量检索是把「问题」和「文档」分别编码再算相似度，两者没见过面；
cross-encoder 把问题和文档拼在一起送进模型，逐对打分，精度更高但慢得多。

实测（本机 CPU，bge-reranker-base）：
    6 个候选  ≈ 1.3s
    10 个候选 ≈ 2.1s
    20 个候选 ≈ 4.3s

对比：不加 rerank 时，整条问答链路也才 1.3s。所以默认关闭。

它在硬改写问题上确实更强——「我自己开了个小店，政府会给钱吗」
重排前正确答案在榜外，重排后升到第 1。但本项目的向量检索
在加了查询前缀之后已经能答对，重排的收益体现不出来。

开启方式：config.ENABLE_RERANK = True
"""

from . import config

_model = None


def get_model():
    global _model
    if _model is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _model = TextCrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(query, hits, top=config.TOP_K):
    """hits 是 [(fused, vsim, chunk)]，返回同样结构但按重排分数排序。"""
    if not hits:
        return hits

    docs = [c["text"] for _, _, c in hits]

    try:
        scores = list(get_model().rerank(query, docs))
    except Exception:
        return hits          # 重排失败就用原顺序，不影响主流程

    order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
    return [(float(scores[i]), hits[i][1], hits[i][2]) for i in order[:top]]