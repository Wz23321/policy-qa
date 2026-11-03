"""读 .env 里的配置。"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INDEX = DATA / "index.pkl"


def load_env():
    """把 .env 里的 KEY=VALUE 读进字典"""
    env = {}
    path = ROOT / ".env"
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


ENV = load_env()

DEEPSEEK_KEY = ENV.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

CHUNK_SIZE = 400      # 每块字数
CHUNK_OVERLAP = 60    # 相邻块重叠
TOP_K = 6             # 检索返回几块（语料扩大后从 5 调到 6，召回回到 100%）
MIN_VEC_SIM = 0.40    # 向量相似度低于这个值，判定"没有相关内容"
MAX_TURNS = 6         # 多轮对话保留最近几轮