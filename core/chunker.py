"""把政策文件切成小块，每块记住自己出自哪份文件。"""

import pathlib
import re

from . import config


def parse_doc(path):
    """读一个 txt，拆出元信息和正文。"""
    text = path.read_text(encoding="utf-8")
    meta = {"file": path.name, "title": path.stem,
            "org": "", "doc_no": "", "date": "", "region": "全国",
            "invalid_date": ""}
    body_lines = []
    in_head = True

    for line in text.splitlines():
        if in_head and "：" in line:
            key, _, val = line.partition("：")
            if key == "标题":
                meta["title"] = val.strip()
                continue
            if key == "地区":
                meta["region"] = val.strip()
                continue
            if key == "失效时间":
                meta["invalid_date"] = val.strip()
                continue
            if key == "发文机关":
                meta["org"] = val.strip()
                continue
            if key == "发文字号":
                meta["doc_no"] = val.strip()
                continue
            if key == "发布日期":
                meta["date"] = val.strip()
                continue
            if key == "来源":
                continue
        in_head = False
        body_lines.append(line)

    meta["body"] = "\n".join(body_lines).strip()
    return meta


ITEM_RE = re.compile(r"^(?:[一二三四五六七八九十]+、|\d+\.)")


def split(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """优先按"一、""1."这类条款边界切开；单条太长再按长度切。"""
    lines = text.split("\n")
    blocks, cur = [], []
    for line in lines:
        if ITEM_RE.match(line.strip()) and cur:
            blocks.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))

    chunks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= size * 1.6:
            if len(block) >= 40:
                chunks.append(block)
        else:
            chunks.extend(_split_by_len(block, size, overlap))
    return chunks


def _split_by_len(text, size, overlap):
    """兜底：实在太长的块按长度切，尽量在句末断开。"""
    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("。"), window.rfind("\n"), window.rfind("；"))
            if cut > size * 0.6:
                end = start + cut + 1
        piece = text[start:end].strip()
        if len(piece) >= 40:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def build():
    """扫 data/raw 下所有 txt，返回所有块。"""
    chunks = []
    for path in sorted(config.RAW.glob("*.txt")):
        doc = parse_doc(path)
        for i, piece in enumerate(split(doc["body"])):
            chunks.append({
                "id": len(chunks),
                "doc": doc["title"],
                "doc_no": doc["doc_no"],
                "org": doc["org"],
                "date": doc["date"],
                "region": doc["region"],
                "invalid_date": doc["invalid_date"],
                "file": doc["file"],
                "seq": i,
                "text": piece,
            })
    return chunks


if __name__ == "__main__":
    cs = build()
    print(f"共 {len(cs)} 块")
    if cs:
        c = cs[0]
        print(f"\n示例 —— 来自《{c['doc']}》{c['doc_no']}")
        print(c["text"][:200])