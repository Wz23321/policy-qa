"""重建索引。政策文件有增减时跑一次。"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core import chunker, retriever  # noqa: E402


def main():
    chunks = chunker.build()
    if not chunks:
        print("data/raw 里没有 txt 文件")
        return

    index = retriever.build_index(chunks)
    retriever.save(index)

    docs = {c["doc"] for c in chunks}
    print(f"文件 {len(docs)} 份，切块 {len(chunks)} 块，词表 {len(index['idf'])} 个")
    print(f"索引已保存到 {retriever.config.INDEX}")


if __name__ == "__main__":
    main()