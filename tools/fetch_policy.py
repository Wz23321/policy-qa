"""从政府网站抓一篇政策文件，存成纯文本。

用法: python tools/fetch_policy.py <网址>
输出: data/raw/<标题>.txt
"""

import re
import sys
import pathlib
import urllib.request

OUT = pathlib.Path("data/raw")


def fetch(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def pick(html, pattern):
    m = re.search(pattern, html, re.S)
    return m.group(1).strip() if m else ""


def html_to_text(block):
    block = re.sub(r"<script.*?</script>", "", block, flags=re.S)
    block = re.sub(r"<style.*?</style>", "", block, flags=re.S)
    block = re.sub(r"<br\s*/?>", "\n", block)
    block = re.sub(r"</p>", "\n", block)
    block = re.sub(r"<[^>]+>", "", block)
    block = block.replace("&nbsp;", " ").replace("&amp;", "&")
    block = block.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    lines = [ln.strip() for ln in block.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def main(url):
    html = fetch(url)

    title = pick(html, r"<title>(.*?)</title>").split("_")[0]
    doc_no = pick(html, r"发文字号：</?\w+>(?:.*?)<td[^>]*>(.*?)</td>")
    pub_date = pick(html, r"发布日期：</?\w+>(?:.*?)<td[^>]*>(.*?)</td>")
    org = pick(html, r"发文机关：</?\w+>(?:.*?)<td[^>]*>(.*?)</td>")

    body_m = re.search(r'<div class="[^"]*pages_content[^"]*"[^>]*>(.*?)</div>\s*</div>', html, re.S)
    if not body_m:
        body_m = re.search(r'<div class="[^"]*pages_content[^"]*"[^>]*>(.*)', html, re.S)
    body = html_to_text(body_m.group(1)) if body_m else ""

    if not body:
        print("!! 没找到正文:", url)
        return None

    OUT.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>|]', "_", title)[:60]
    path = OUT / f"{safe}.txt"

    head = f"标题：{title}\n发文机关：{org}\n发文字号：{doc_no}\n发布日期：{pub_date}\n来源：{url}\n\n"
    path.write_text(head + body, encoding="utf-8")

    print(f"OK  {title[:40]}  {len(body)} 字  -> {path.name}")
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python tools/fetch_policy.py <网址>")
        sys.exit(1)
    main(sys.argv[1])