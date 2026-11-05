"""所有大模型调用都从这里走。

统一处理三件事：
1. 客户端复用
2. 失败重试（指数退避 + 抖动）
3. 超时

为什么要抖动：多个请求同时失败时，如果都按固定间隔重试，会一起再打过去，
把上游打得更惨。加随机量能把重试摊开。
"""

import random
import time

from openai import OpenAI

from . import config

_client = None

# 这些错误重试有意义；参数错、鉴权错这类重试也没用
try:
    from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                        RateLimitError)
    RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError)
    STATUS_RETRYABLE = (429, 500, 502, 503, 504)
except ImportError:  # 老版本 openai
    RETRYABLE = (Exception,)
    STATUS_RETRYABLE = (429, 500, 502, 503, 504)


def get_client():
    global _client
    if _client is None:
        if not config.DEEPSEEK_KEY:
            raise RuntimeError("没读到 DEEPSEEK_API_KEY，检查 .env 或环境变量")
        _client = OpenAI(
            api_key=config.DEEPSEEK_KEY,
            base_url=config.DEEPSEEK_BASE,
            timeout=config.LLM_TIMEOUT,
            max_retries=0,          # 重试我们自己控制，不让 SDK 再来一遍
        )
    return _client


def _is_retryable(exc):
    if isinstance(exc, RETRYABLE):
        return True
    status = getattr(exc, "status_code", None)
    return status in STATUS_RETRYABLE


def _backoff(attempt, base=config.RETRY_BASE_DELAY, cap=config.RETRY_MAX_DELAY):
    """指数退避 + 抖动：1s, 2s, 4s... 各带 ±50% 随机。"""
    delay = min(cap, base * (2 ** attempt))
    return delay * (0.5 + random.random())


def chat(messages, tools=None, temperature=0.2, max_tokens=None,
         stream=False, attempts=config.RETRY_ATTEMPTS, on_retry=None):
    """发一次对话请求，失败自动重试。

    on_retry(attempt, exc, delay) 可选，用来打日志或推给前端。
    """
    last_exc = None

    for attempt in range(attempts):
        try:
            kwargs = {
                "model": config.MODEL,
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
            }
            if tools:
                kwargs["tools"] = tools
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            return get_client().chat.completions.create(**kwargs)

        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == attempts - 1:
                raise

            delay = _backoff(attempt)
            if on_retry:
                on_retry(attempt + 1, exc, delay)
            time.sleep(delay)

    raise last_exc