"""读配置：优先环境变量，其次 .env 文件。

环境变量优先是为了容器部署——docker-compose 通过 environment 传 key，
容器里没有 .env 文件。
"""

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INDEX = DATA / "index.pkl"


def load_env():
    """合并环境变量和 .env 文件。环境变量优先级更高。"""
    env = {}

    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE", "MODEL", "HF_ENDPOINT"):
        if os.environ.get(key):
            env[key] = os.environ[key]

    return env


ENV = load_env()

DEEPSEEK_KEY = ENV.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = ENV.get("DEEPSEEK_BASE", "https://api.deepseek.com/v1")
MODEL = ENV.get("MODEL", "deepseek-chat")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

CHUNK_SIZE = 400      # 每块字数
CHUNK_OVERLAP = 60    # 相邻块重叠
TOP_K = 6             # 实测 top_k=4 不会更快（差异在噪声内），保持 6 保召回
MIN_VEC_SIM = 0.40    # 向量相似度低于这个值，判定"没有相关内容"
MAX_TURNS = 6         # 多轮对话保留最近几轮
COMPRESS_AFTER = 8    # 超过这么多轮就压缩旧内容

LLM_TIMEOUT = 60      # 单次模型调用超时（秒）
RETRY_ATTEMPTS = 3    # 失败重试次数
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 8.0

SESSION_FILE = DATA / "sessions.json"

# 查询时给 embedding 加一句领域前缀。
# 短问题（"灵活就业社保补贴标准"）缺少学科语境，加前缀能显著提升召回；
# 不加的话正确答案可能排到 20 名开外。
QUERY_HINT = "高校毕业生就业政策 "

# Rerank：精度更高但慢得多，默认关闭。开启理由和实测数据见 core/rerank.py
ENABLE_RERANK = False
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_CANDIDATES = 10      # 从多少个粗召回结果里重排