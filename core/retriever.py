"""检索：关键词（TF-IDF）+ 向量（BGE）双路召回。

两路各有长处：
- 关键词：专有名词、文号、"30万"这种精确词，一打一个准
- 向量：同义改写、"创业贷款" vs "创业担保贷款"，关键词匹配不上，向量能
所以合并两路结果。
"""

import math
import os
import pickle
import re
from collections import Counter

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from . import config  # noqa: E402

try:
    import jieba
    jieba.setLogLevel(20)
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(config.EMBED_MODEL)
    return _embedder


STOP = set("的 了 和 与 及 在 是 为 对 等 中 有 不 也 就 都 而 及其 根据 关于 "
           "按照 以及 并且 或者 应当 可以 相关 有关".split())


def tokenize(text):
    text = text.lower()
    if _HAS_JIEBA:
        return [w for w in jieba.lcut(text)
                if len(w) > 1 and w not in STOP and not w.isdigit()]
    text = re.sub(r"\s+", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1) if text[i:i + 2] not in STOP]


# ---------- 关键词一路 ----------

def _build_keyword(chunks):
    docs, df = [], Counter()
    for c in chunks:
        # 标题和地区加权，让"河南""就业补贴"这类词更容易命中对应文件
        tf = Counter(tokenize(c["text"]) + tokenize(c["doc"]) * 2
                     + tokenize(c.get("region", "")) * 3)
        docs.append(tf)
        for w in tf:
            df[w] += 1

    n = len(docs)
    idf = {w: math.log((n + 1) / (cnt + 1)) + 1 for w, cnt in df.items()}

    vectors = []
    for tf in docs:
        vec = {w: (1 + math.log(cnt)) * idf.get(w, 0) for w, cnt in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({w: v / norm for w, v in vec.items()})

    return {"idf": idf, "vectors": vectors}


def _keyword_search(index, query):
    idf = index["idf"]
    qtf = Counter(tokenize(query))
    qvec = {w: (1 + math.log(c)) * idf[w] for w, c in qtf.items() if w in idf}
    qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
    qvec = {w: v / qnorm for w, v in qvec.items()}

    out = []
    for i, vec in enumerate(index["kvecs"]):
        s = sum(qvec.get(w, 0) * v for w, v in vec.items())
        if s > 0:
            out.append((i, s))
    out.sort(key=lambda x: -x[1])
    return out


# ---------- 向量一路 ----------

def _build_vectors(chunks):
    texts = [f"{c.get('region','')} {c['doc']}\n{c['text']}" for c in chunks]
    vecs = list(get_embedder().embed(texts))
    return [v.tolist() for v in vecs]


def _vector_search(index, query):
    # 短问题缺上下文，加领域前缀能明显提升召回
    text = (config.QUERY_HINT + query) if config.QUERY_HINT else query
    q = list(get_embedder().embed([text]))[0]
    qn = float((sum(x * x for x in q)) ** 0.5) or 1.0

    out = []
    for i, vec in enumerate(index["dvecs"]):
        dot = sum(a * b for a, b in zip(q, vec))
        vn = index["dnorms"][i] or 1.0
        out.append((i, dot / (qn * vn)))
    out.sort(key=lambda x: -x[1])
    return out


# ---------- 对外 ----------

def build_index(chunks):
    kw = _build_keyword(chunks)
    dvecs = _build_vectors(chunks)
    dnorms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in dvecs]
    return {
        "chunks": chunks,
        "idf": kw["idf"],
        "kvecs": kw["vectors"],
        "dvecs": dvecs,
        "dnorms": dnorms,
    }


def save(index, path=config.INDEX):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load(path=config.INDEX):
    with open(path, "rb") as f:
        return pickle.load(f)


def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion：把多路排名的名次折算成分数再相加。
    比直接混合不同量纲的分数稳。"""
    scores = Counter()
    for ranks in rank_lists:
        for pos, (idx, _) in enumerate(ranks, 1):
            scores[idx] += 1.0 / (k + pos)
    return scores.most_common()


def search(index, query, top=config.TOP_K, mode="vector"):
    """返回 [(融合分, 向量相似度, 块), ...]

    向量相似度单独返回，用来判断"这次检索到底靠不靠谱"——
    RRF 的分数量纲很小，不能直接拿来卡阈值。
    """
    if mode == "keyword":
        ranked = _keyword_search(index, query)
        return [(s, s, index["chunks"][i]) for i, s in ranked[:top]]

    if mode == "vector":
        ranked = _vector_search(index, query)
        return [(s, s, index["chunks"][i]) for i, s in ranked[:top]]

    kw = _keyword_search(index, query)[:20]
    vec = _vector_search(index, query)[:20]
    vec_sim = {i: s for i, s in vec}

    merged = rrf([kw, vec])
    return [(f, vec_sim.get(i, 0.0), index["chunks"][i])
            for i, f in merged[:top]]