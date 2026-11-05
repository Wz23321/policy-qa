"""评测：跑一批标注好的问题，看答对率。

用法:
    python scripts/eval.py          # 评测 /ask（单次检索）
    python scripts/eval.py agent    # 评测 Agent 模式

判分规则：
- 能答的问题：答案里必须出现至少一个关键词，且引用的文件里包含指定文号
- 该拒答的问题：答案必须明确说找不到
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core import agent, rag, retriever  # noqa: E402

# expect_any: 答案里至少出现其中一个词
# doc: 来源里必须出现这个文号（None 表示不检查）
# refuse: 应该拒答
CASES = [
    # ---- 能答的 ----
    {"q": "一次性求职补贴哪些人可以领？",
     "expect_any": ["低保", "零就业", "残疾", "助学贷款"], "doc": "人社部发〔2026〕19号"},
    {"q": "就业见习补贴是发给谁的？",
     "expect_any": ["单位", "见习单位"], "doc": "人社部发〔2026〕19号"},
    {"q": "创业担保贷款个人最高能贷多少钱？",
     "expect_any": ["30万", "300000", "三十万"], "doc": "人社部发〔2026〕19号"},
    {"q": "小微企业创业担保贷款最高多少？",
     "expect_any": ["400万", "4000000"], "doc": "人社部发〔2026〕19号"},
    {"q": "创业担保贷款的期限是多久？",
     "expect_any": ["3年", "三年"], "doc": "人社部发〔2026〕19号"},
    {"q": "创业担保贷款财政贴息多少？",
     "expect_any": ["50%", "一半"], "doc": "人社部发〔2026〕19号"},
    {"q": "灵活就业的高校毕业生社保补贴标准是什么？",
     "expect_any": ["2/3", "三分之二", "实际缴费"], "doc": "人社部发〔2026〕19号"},
    {"q": "灵活就业社保补贴能领多久？",
     "expect_any": ["2年", "两年"], "doc": "人社部发〔2026〕19号"},
    {"q": "一次性创业补贴需要什么条件？",
     "expect_any": ["首次", "1年", "正常运营"], "doc": "人社部发〔2026〕19号"},
    {"q": "就业见习补贴的政策期限到什么时候？",
     "expect_any": ["2026年12月31日", "2026年底"], "doc": "人社部发〔2026〕19号"},
    {"q": "高校毕业生到基层就业有什么支持政策？",
     "expect_any": ["学费补偿", "助学贷款代偿", "工资"], "doc": None},
    {"q": "高校毕业生应征入伍有什么优惠？",
     "expect_any": ["入伍", "征兵", "服役"], "doc": None},
    {"q": "科研助理岗位是做什么的？",
     "expect_any": ["科研", "助理"], "doc": None},
    {"q": "城乡社区有哪些岗位给高校毕业生？",
     "expect_any": ["社区", "见习"], "doc": None},
    {"q": "困难毕业生就业有什么帮扶？",
     "expect_any": ["低保", "残疾", "帮扶", "特困"], "doc": None},

    # ---- 同义改写（考验向量检索，关键词往往匹配不上）----
    {"q": "我自己开了个小店，刚开始做，政府会给钱吗？",
     "expect_any": ["一次性创业补贴", "创业补贴", "5000", "1年"], "doc": None},
    {"q": "毕业了暂时没工作，自己交社保有补助吗？",
     "expect_any": ["社保", "社会保险", "2/3", "三分之二"], "doc": None},
    {"q": "想跟同学合伙开公司，启动资金不够怎么办？",
     "expect_any": ["创业担保贷款", "30万", "贷款"], "doc": None},
    {"q": "公司招应届生，政府有奖励或者减免吗？",
     "expect_any": ["社会保险补贴", "税收", "吸纳", "补贴"], "doc": None},
    {"q": "实习期间单位给不给生活费？",
     "expect_any": ["见习", "基本生活费"], "doc": None},
    {"q": "没钱参加职业资格考试怎么办？",
     "expect_any": ["职业技能评价补贴", "职业培训补贴", "补贴"], "doc": None},

    # ---- 该拒答的 ----
    {"q": "考研报名时间是什么时候？", "refuse": True},
    {"q": "北京落户政策怎么样？", "refuse": True},
    {"q": "今天天气怎么样？", "refuse": True},
    {"q": "国考报名流程是什么？", "refuse": True},
    {"q": "房贷利率是多少？", "refuse": True},
]


REFUSE_WORDS = ("没有找到", "未找到", "找不到", "无法回答", "不能回答",
                "无法提供", "不在我的", "不在政策库", "没有相关", "无法查询")


def _norm(text):
    """去掉所有空白，避免「30 万元」和「30万」匹配不上。"""
    return "".join(text.split())


def judge(case, answer, sources, answer_text_has_doc=False):
    flat = _norm(answer)

    if case.get("refuse"):
        ok = any(k in flat for k in REFUSE_WORDS)
        return ok, "该拒答" if ok else "编造了答案"

    hit = any(_norm(k) in flat for k in case["expect_any"])
    if not hit:
        return False, "缺少关键信息"

    if case.get("doc"):
        # Agent 不返回结构化 sources，改为在答案正文里找文号
        pool = _norm(" ".join(s.get("doc_no", "") for s in sources))
        if not pool and answer_text_has_doc:
            pool = flat
        if _norm(case["doc"]) not in pool:
            return False, f"来源不对（未引到 {case['doc']}）"

    return True, "正确"


def run(use_agent=False, mode="hybrid"):
    index = retriever.load()
    passed, details = 0, []

    for i, case in enumerate(CASES, 1):
        q = case["q"]
        try:
            if use_agent:
                r = agent.ask(q, index)
                answer, sources = r["answer"], []
            else:
                r = rag.ask(q, index, mode=mode)
                answer, sources = r["answer"], r["sources"]
        except Exception as e:
            details.append((i, q, False, f"异常: {e}"))
            continue

        ok, why = judge(case, answer, sources, answer_text_has_doc=use_agent)
        passed += ok
        details.append((i, q, ok, why))
        if verbose:
            print(f"{'✓' if ok else '✗'} {i:2d}. {q[:30]:<32} {why}")

    total = len(CASES)
    label = "Agent" if use_agent else mode
    print(f"{label:<8} {passed}/{total} 通过，准确率 {passed/total*100:.1f}%")
    return passed, total


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    verbose = arg not in ("all",)

    if arg == "all":
        print("=" * 46)
        for m in ("keyword", "vector", "hybrid"):
            run(mode=m)
        print("=" * 46)
        run(use_agent=True)
    elif arg == "agent":
        run(use_agent=True)
    else:
        run(mode=arg)