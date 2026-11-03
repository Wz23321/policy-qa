"""抓取省级人社厅的政策文件（目前支持河南省人社厅）。

河南用的是集约化 CMS，列表走 JSON 接口，文章页是静态 HTML。

用法:
    python tools/fetch_henan.py            # 抓就业相关并入库
    python tools/fetch_henan.py --list     # 只列出来看看
"""

import json
import re
import sys
import pathlib
import urllib.request

OUT = pathlib.Path("data/raw")
SITE = "4500000013"
CID = "14357"
REGION = "河南"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Referer": "https://hrss.henan.gov.cn/zcwj/",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

KEEP = ("就业", "毕业生", "创业", "见习")
DROP = ("职称", "竞赛", "评审", "师资", "目录清单", "表彰", "大赛",
        "示范点", "示范县", "示范园", "评估", "机构管理", "一卡通", "评选")


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers=HEADERS)
    return urllib.request.urlopen(req, timeout=timeout).read()


def _jsonp_to_json(text):
    m = re.match(r"^\s*\w+\((.*)\)\s*$", text, re.S)
    return json.loads(m.group(1) if m else text)


def list_docs(max_pages=8, page_size=50):
    """拉政策文件列表。"""
    rows = []
    for page in range(1, max_pages + 1):
        url = (f"https://jyxc.henan.gov.cn/open/documents?siteId={SITE}&cid={CID}"
               f"&pageNumber={page}&pageSize={page_size}&docTypeId=0&themeClassId=0")
        try:
            data = _jsonp_to_json(_get(url).decode("utf-8", "ignore"))
        except Exception as e:
            print(f"  第 {page} 页失败: {e}")
            break
        batch = (data.get("obj") or {}).get("datas") or []
        if not batch:
            break
        rows.extend(batch)
    return rows


def relevant(rows):
    keep = []
    for r in rows:
        t = r.get("title", "")
        if any(k in t for k in KEEP) and not any(k in t for k in DROP):
            keep.append(r)
    return keep


def meta(html, name):
    m = re.search(rf'<meta name="{name}" content="([^"]*)"', html)
    return m.group(1).strip() if m else ""


def html_to_text(block):
    block = re.sub(r"<script.*?</script>", "", block, flags=re.S)
    block = re.sub(r"<style.*?</style>", "", block, flags=re.S)
    block = re.sub(r"<!--.*?-->", "", block, flags=re.S)
    block = re.sub(r"</p>|<br\s*/?>", "\n", block)
    block = re.sub(r"<[^>]+>", "", block)
    block = (block.replace("&nbsp;", " ").replace("&amp;", "&")
                  .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))
    lines = [ln.strip() for ln in block.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def extract_body(html):
    """正文在 title7 标题之后的表格里，页脚之前。"""
    i = html.find('class="title7"')
    if i < 0:
        i = html.find('class="context6"')
    if i < 0:
        return ""

    # 跳过标题 div 本身
    close = html.find("</div>", i)
    seg = html[close + 6:] if close > 0 else html[i:]

    for stop in ("最后更新", "主办单位", "版权所有", "网站标识码",
                 "扫一扫", "class=\"footer", "<!--endprint"):
        j = seg.find(stop)
        if j > 0:
            seg = seg[:j]
            break

    return html_to_text(seg)


def find_attachment(html):
    """河南的政策文件正文常常是 doc/docx/pdf 附件。"""
    cands = re.findall(r'href="([^"]+\.(?:docx?|pdf))"', html, re.I)
    if not cands:
        cands = re.findall(r'"(https?://[^"]+\.(?:docx?|pdf))"', html, re.I)
    return cands[0] if cands else ""


def docx_text(data):
    """从 docx 二进制里抽出纯文字（docx 就是 zip + xml）。"""
    import io
    import zipfile

    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    except Exception:
        return ""

    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    lines = [ln.strip() for ln in xml.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def pdf_text(data):
    try:
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception:
        return ""


def attachment_text(url):
    try:
        data = _get(url, timeout=60)
    except Exception as e:
        return "", f"附件下载失败: {e}"
    low = url.lower()
    if low.endswith(".pdf"):
        return pdf_text(data), ""
    return docx_text(data), ""


def fetch_one(row):
    url = row["url"]
    try:
        html = _get(url, timeout=30).decode("utf-8", "ignore")
    except Exception as e:
        return f"!! 下载失败 {url} : {e}"

    title = meta(html, "ArticleTitle") or row.get("title", "")
    pub = meta(html, "PubDate") or row.get("putDate", "")
    org = meta(html, "ContentSource") or "河南省人力资源和社会保障厅"
    doc_no = row.get("docTypeName", "")

    # 河南的页面里带"失效时间"，正好用来标记政策是否作废
    m = re.search(r"失效时间[:：]</strong>\s*([^<]*)", html)
    invalid = m.group(1).strip() if m else ""

    body = extract_body(html)
    if len(body) < 100:
        att = find_attachment(html)
        if att:
            body, err = attachment_text(att)
            if err:
                return f"!! {err} {title[:30]}"

    if len(body) < 100:
        return f"!! 正文太短 {title[:30]}"

    OUT.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>|]', "_", title)[:60]
    path = OUT / f"{safe}.txt"

    head = (f"标题：{title}\n地区：{REGION}\n发文机关：{org}\n"
            f"发文字号：{doc_no}\n发布日期：{pub[:10]}\n"
            f"失效时间：{invalid or '未标注'}\n来源：{url}\n\n")
    path.write_text(head + body, encoding="utf-8")
    return f"OK  {title[:36]}  {len(body)} 字"


def main(argv):
    print(f"拉取{REGION}人社厅政策列表…")
    rows = list_docs()
    print(f"  共 {len(rows)} 篇，筛出就业相关 {len(relevant(rows))} 篇\n")

    sel = relevant(rows)
    if "--list" in argv:
        for r in sel:
            print(" ", r.get("title", "")[:56], "|", r.get("putDate", ""))
        return

    ok = 0
    for r in sel:
        msg = fetch_one(r)
        print(msg)
        if msg.startswith("OK"):
            ok += 1
    print(f"\n成功入库 {ok} 篇")


if __name__ == "__main__":
    main(sys.argv[1:])