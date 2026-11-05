# 代码导读

按数据流顺序读，每个文件讲清楚「为什么这么写」。面试前过一遍，能指着代码说细节。

---

## 整体数据流

```
data/raw/*.txt
    │
    ├─ chunker.py      按条款切块
    │
    ├─ retriever.py    建索引：TF-IDF 向量 + bge 向量
    │
    └─ index.pkl
           │
  提问 ────┼──► retriever.search()  双路召回 → RRF 融合
           │
           ├──► rag.region_rerank()  按地区重排
           │
           ├──► rag.format_materials()  拼成提示词
           │
           └──► DeepSeek  ──► answer + sources
```

Agent 模式走另一条路：`agent.py` 让模型自己决定调几次 `search_policy`。

---

## core/config.py

读 `.env`，放常量。没有逻辑，但几个参数是调出来的，不是拍脑袋定的：

| 参数 | 值 | 怎么定的 |
|---|---|---|
| `CHUNK_SIZE` | 400 | 一条政策通常 200–600 字，400 能让整条落在一块里 |
| `TOP_K` | 6 | 语料扩到 725 块后，5 不够、6 刚好召回 100% |
| `MIN_VEC_SIM` | 0.40 | 低于这个值基本是无关内容 |
| `MAX_TURNS` | 6 | 再多上下文会撑爆且变慢 |

---

## core/chunker.py

**关键函数 `split()`**：不按字数切，按 `一、`、`1.` 这些编号切。

```python
ITEM_RE = re.compile(r"^(?:[一二三四五六七八九十]+、|\d+\.)")
```

逐行扫，遇到编号行就是新块的开始。超过 `size * 1.6` 的块再按长度兜底切。

**为什么**：政策文件是「1.就业见习补贴 / 政策内容：……」的结构。按 400 字硬切，很容易切在「政策内容：」后面，那一条政策就断成两半，检索命中半句，模型也答不全。

实测：按字数切，「就业见习补贴」的检索相似度 0.26；按条款切，0.46。

**`parse_doc()`** 读文件头部的元信息（标题 / 地区 / 发文机关 / 发文字号 / 发布日期 / 失效时间 / 来源），正文从第一个不带冒号的行开始。

---

## core/retriever.py

### 关键词一路

`jieba` 分词 → TF-IDF → 余弦相似度。

标题权重 ×2、地区权重 ×3，因为标题含关键词的文件相关性更高。

### 向量一路

`fastembed` 加载 `BAAI/bge-small-zh-v1.5`（512 维，约 90MB）。

索引时把「地区 + 标题 + 正文」一起编码，这样问「河南的补贴」能同时命中地区和内容。

### 融合

```python
def rrf(rank_lists, k=60):
    scores = Counter()
    for ranks in rank_lists:
        for pos, (idx, _) in enumerate(ranks, 1):
            scores[idx] += 1.0 / (k + pos)
```

RRF 只看名次不看分数——因为 TF-IDF 的分数和余弦相似度量纲不同，直接加权没意义。

**默认用纯向量**：实测语料 725 块时，纯向量 top3 命中 100%，合并反而降到 95.2%。合并的价值在 top1 更稳（81% vs 71%），但本流程会传 6 块给模型，不依赖单一 top1。

---

## core/rag.py

### 多轮查询改写

```python
def rewrite_query(question, history):
    if not history:
        return question        # 没有上文就不多花一次调用
```

「那期限呢」这四个字里没有可检索的词。先让模型结合上文改写成「创业担保贷款的期限是多久」，再拿去检索。

没有历史时直接返回原问题——省一次调用。

### 地区重排序

```python
def region_rerank(hits, question, regions):
    target = next((r for r in regions if r != "全国" and r in question), None)
```

问题里出现某省 → 该省 ×1.3、国家级 ×1.1；没出现 → 国家级 ×1.15。

**只调分数，不删候选。** 如果直接过滤掉其他省，遇到「只有地方政策能回答」的问题就彻底查不到了。

这层是加地方政策后补的：没它的时候，问「一次性求职补贴谁能领」会被河南政策抢占，Agent 甚至引用了河南文号。

### 拒答的三层

1. 提示词里写死：材料里没有就说「没有找到相关内容」
2. 检索兜底：向量相似度低于 `MIN_VEC_SIM` 的片段丢掉
3. 答案后处理：模型说没找到时，不返回 sources（否则界面自相矛盾）

### 缓存

```python
cache_key = (mode, question.strip())
if not history:
    hit = cache.QUERY_CACHE.get(cache_key)
```

**只对单轮生效。** 多轮答案依赖上文，同一个问句在不同上下文里答案不同，缓存会串。

---

## core/agent.py

Agent 循环：

```
用户问题 → 模型决定调工具 → 执行 → 结果回给模型 → 还要调就继续 → 给答案
```

三道防护：
- `MAX_STEPS = 5`，防止死循环
- 工具参数用 JSON Schema 约束，非法参数返回错误给模型自己改
- `calculate` 用 `ast` 白名单解析，不调 `eval`

```python
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ...}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只支持数字")
```

只允许数字和四则运算。看到 `__import__('os')` 这种直接抛异常。

---

## core/rerank.py

Cross-encoder 精排，**默认关闭**（`config.ENABLE_RERANK = False`）。

打开后对粗召回的候选用 bge-reranker-base 逐对打分重排，精度更高。但 CPU 上 10 个候选要 2.1 秒，而整条问答链路才 1.3 秒——加了就是延迟翻倍。实测在硬改写问题上确实更准（正确答案从榜外升到第 1），但本项目的向量检索加了查询前缀之后已经能答对，收益体现不出来。

留着的意义是：需要高精度场景（比如离线批量处理）时改个配置就能用。

---

## core/session.py

会话存最近 6 轮，带 TTL 和 LRU 淘汰。

`OrderedDict` + `move_to_end`：每次访问把会话挪到末尾，超容量时 `popitem(last=False)` 淘汰最久没用的。

---

## core/cache.py

两个计数器 `hits` / `misses`，`/health` 会返回命中率。面试时可以说「缓存命中后延迟从 1.7s 降到 0」。

---

## cpp-client/

三个文件，零第三方依赖。

**http.cpp** —— 手写 HTTP/1.1：
- 拼请求行和请求头
- 解析状态行 `HTTP/1.1 200 OK`
- `Content-Length` 和 `Transfer-Encoding: chunked` 两种响应体
- `send_all` 循环处理 TCP 短写
- `SO_RCVTIMEO` / `SO_SNDTIMEO` 防止永久阻塞

**json.cpp** —— 递归下降解析器：
- 对象、数组、字符串、数字、布尔、null
- 转义 `\n \" \\`
- `\uXXXX` 解码，含代理对（`😀` 这种四字节字符由两个 `\u` 组成）
- 非法输入抛异常

**main.cpp** —— 命令行，支持 `-a` Agent 模式、`-h` `-p` 指定地址、无参数进交互模式。

---

## scripts/

| 脚本 | 干什么 |
|---|---|
| `build_index.py` | 重建索引，56 份文件约 41 秒 |
| `eval.py` | 26 条评测集，`all` 参数对比四种模式 |
| `bench.py` | 延迟 / token / 缓存 / 成本 |

---

## tests/

- `test_core.py` 16 条：切块边界、地区重排、LRU 淘汰顺序、TTL、会话截断
- `cpp-client/test_json.cpp` 25 条：JSON 解析边界

**测试抓到过一个真 bug**：JSON 解析器遇到 `{"a": }` 不报错，静默返回数字 0。原因是 `parse_number()` 在没有任何数字时也返回 `strtod` 的默认值。修法是要求数字必须以 `-` 或数字开头。

---

## 面试时能指着说的三个地方

1. `core/chunker.py` 的 `ITEM_RE` —— 切块策略带来的效果（0.26 → 0.46）
2. `core/rag.py` 的 `region_rerank` —— 加一类数据带出新问题，怎么修的
3. `cpp-client/http.cpp` 的 `send_all` —— 为什么必须循环 send