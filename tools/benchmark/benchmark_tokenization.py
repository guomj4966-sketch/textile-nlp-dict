"""M1 分词质量基线基准测试（修复版 v2）。

关键修正：
1. 只测试词典中真实存在 + 在政策语料中确实出现的术语
2. 用高频术语集（3-6字），按长度均匀采样
3. 正确检测"默认分词切碎 → 词典分词修复"的案例

运行:
    PYTHONIOENCODING=utf-8 python scripts/nlp_dict/benchmark_tokenization.py
"""

import sys, io, re, csv, argparse, sqlite3
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import jieba

# ============================================================
# 配置
# ============================================================
DB_PATH = Path(__file__).parent.parent.parent / "collector" / "policy.db"
DATA_DIR = Path(__file__).parent / "data"
JIEBA_DICT = DATA_DIR / "jieba_dict.txt"


# ============================================================
# 数据准备
# ============================================================

def load_corpus_present_terms(max_terms=150):
    """从词典中提取 3-6 字纯中文术语，筛选出在政策语料中确实出现的。"""
    # Step 1: 加载词典中的所有术语
    dict_terms = {}   # {term: set() for freq tiers}
    with open(JIEBA_DICT, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line: continue
            parts = line.split()
            word = parts[0] if parts else ''
            if 3 <= len(word) <= 6 and re.match(r'^[一-鿿]{3,}$', word):
                dict_terms[word] = True

    print(f"   词典中 3-6 字纯中文术语: {len(dict_terms)} 个")

    # Step 2: 扫描语料，找出实际出现过的术语
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute("""
        SELECT full_text FROM policy
        WHERE review_status = '通过'
        AND (textile_relevance = '直接' OR textile_relevance = '间接')
        AND doc_type = '政策原文'
        AND full_text IS NOT NULL AND full_text != ''
    """).fetchall()
    conn.close()

    all_text = ''
    for (ft,) in rows:
        all_text += re.sub(r'<[^>]+>', '', ft) + '\n'
    all_text = re.sub(r'https?://\S+', '', all_text)

    print(f"   语料总字符数: {len(all_text):,}")

    # Step 3: 统计在语料中的出现频次
    appearances = {}
    for term in dict_terms:
        c = all_text.count(term)
        if c > 0:
            appearances[term] = c

    print(f"   在语料中出现过: {len(appearances)} 个")
    print(f"   出现 1 次（罕见，不易测）: {sum(1 for v in appearances.values() if v == 1)}")
    print(f"   出现 2+ 次（适合测试）: {sum(1 for v in appearances.values() if v >= 2)}")

    # Step 4: 取出现 2+ 次的术语，按长度均匀采样
    candidates = [(t, f) for t, f in appearances.items() if f >= 2]
    candidates.sort(key=lambda x: -x[1])

    # 按长度分组采样
    by_len = {}
    for t, f in candidates:
        by_len.setdefault(len(t), []).append((t, f))

    sampled = []
    per_len = max(25, max_terms // len(by_len) + 1)
    for l in sorted(by_len):
        sampled.extend(by_len[l][:per_len])

    sampled = sampled[:max_terms]
    print(f"   最终测试术语: {len(sampled)} 个 (3-6字，语料中出现 ≥2 次)")
    return sampled, all_text, appearances


def tokenize(text, load_dict=False):
    """统一分词接口，load_dict=True 时加载自定义词典"""
    jieba.initialize()
    if load_dict:
        jieba.load_userdict(str(JIEBA_DICT))
    return list(jieba.cut(text))


# ============================================================
# 度量计算
# ============================================================

def compute_metrics(terms_to_check, all_text, sample_contexts):
    """
    核心度量逻辑：
    对于每个出现过的术语，检查它在默认分词 vs 词典分词中的切分状态。
    """
    # 对整个语料做两轮分词（太慢的话只对样本上下文做）
    print(f"\n🔍 运行分词对比（这可能要 1-2 分钟）...")
    default_words = set(tokenize(all_text, load_dict=False))
    dict_words = set(tokenize(all_text, load_dict=True))

    total_checked = 0
    total_fixed = 0
    fixed_terms = Counter()
    broken_terms = Counter()

    for term, corpus_freq in terms_to_check:
        total_checked += 1

        in_default = term in default_words
        in_dict = term in dict_words

        if not in_default and in_dict:
            total_fixed += 1
            fixed_terms[term] = corpus_freq
        if not in_default and not in_dict:
            broken_terms[term] = corpus_freq

    # 对采样上下文做更细致的展示
    sample_results = []
    for ctx in sample_contexts:
        text = ctx["segment"]
        def_words = [w.strip() for w in tokenize(text, load_dict=False)
                    if w.strip() and len(w.strip()) >= 2]
        dct_words = [w.strip() for w in tokenize(text, load_dict=True)
                    if w.strip() and len(w.strip()) >= 2]

        ctx_checks = []
        ctx_fixed = 0
        ctx_checked = 0
        for term, _ in terms_to_check:
            if term in text:
                ctx_checked += 1
                in_def = term in def_words
                in_dct = term in dct_words
                fixed = (not in_def and in_dct)
                if fixed: ctx_fixed += 1
                ctx_checks.append({"term": term, "fixed": fixed})

        sample_results.append({
            "title": ctx["title"][:50],
            "terms_found": ctx_checked,
            "terms_fixed": ctx_fixed,
            "checks": ctx_checks,
        })

    return {
        "total_checked": total_checked,
        "total_fixed": total_fixed,
        "fix_rate": round(total_fixed / max(total_checked, 1) * 100, 1),
        "fixed_terms": fixed_terms,
        "broken_terms": broken_terms,
        "samples": sample_results,
    }


def show_examples(metrics, all_text, top_n=15):
    """展示修复案例和仍被切碎的案例，附上下文"""
    # 修复案例
    print(f"\n📊 词典修复的术语 (默认❌ → 词典✅, Top {top_n}):")
    if not metrics["fixed_terms"]:
        print("  (无 — 所有测试术语在默认分词中原本就能正确切分)")

    shown = 0
    for term, _ in metrics["fixed_terms"].most_common(top_n):
        # 在语料中找第一个上下文
        idx = all_text.find(term)
        if idx >= 0:
            ctx = all_text[max(0,idx-12):idx+len(term)+12]
            ctx = re.sub(r'\s+', '', ctx)
            def_words = '/'.join([w.strip() for w in tokenize(ctx, load_dict=False)
                                  if w.strip()])
            dct_words = '/'.join([w.strip() for w in tokenize(ctx, load_dict=True)
                                  if w.strip()])
            print(f"  ✅ {term}")
            print(f"     默认: {def_words}")
            print(f"     词典: {dct_words}")
            print()
            shown += 1
        if shown >= top_n: break

    # 仍被切碎的案例
    print(f"\n⚠️  仍被切碎的术语 (默认❌ → 词典❌, Top 10):")
    if not metrics["broken_terms"]:
        print("  (无 — 所有术语都能被词典修复)")
    shown = 0
    for term, _ in metrics["broken_terms"].most_common(10):
        idx = all_text.find(term)
        if idx >= 0:
            ctx = all_text[max(0,idx-12):idx+len(term)+12]
            ctx = re.sub(r'\s+', '', ctx)
            def_words = '/'.join([w.strip() for w in tokenize(ctx, load_dict=False)
                                  if w.strip()])
            dct_words = '/'.join([w.strip() for w in tokenize(ctx, load_dict=True)
                                  if w.strip()])
            print(f"  ❌ {term}")
            print(f"     默认: {def_words}")
            print(f"     词典: {dct_words}")
            print()
            shown += 1
        if shown >= 10: break


# ============================================================
# 主流程
# ============================================================

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    jieba.setLogLevel(20)

    parser = argparse.ArgumentParser(description="M1 分词质量基线基准测试")
    parser.add_argument("--sample-size", type=int, default=30,
                        help="用于上下文展示的样本数")
    args = parser.parse_args()

    print("=" * 60)
    print("📏 M1: 分词质量基线基准测试 (v2 修复版)")
    print("=" * 60)
    print(f"   jieba_dict: {JIEBA_DICT.name}")

    # 加载测试术语
    print("\n📚 加载测试术语（词典中存在 + 语料中出现过的）...")
    terms_with_freq, all_text, all_appearances = load_corpus_present_terms(max_terms=150)

    # 抽样上下文片段（用于展示）
    print(f"\n📖 准备 {args.sample_size} 个样本上下文...")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT policy_id, title, full_text, publish_date
        FROM policy WHERE review_status = '通过'
        AND (textile_relevance = '直接' OR textile_relevance = '间接')
        AND doc_type = '政策原文'
        AND full_text IS NOT NULL AND length(full_text) > 200
        ORDER BY RANDOM() LIMIT ?
    """, (args.sample_size,)).fetchall()
    conn.close()

    samples = []
    for r in rows:
        text = re.sub(r'<[^>]+>', '', r["full_text"])
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        segment = r["title"] + "。" + text[:800]
        samples.append({"title": r["title"], "segment": segment})

    # 计算度量
    metrics = compute_metrics(terms_with_freq, all_text, samples)

    # 输出结果
    print()
    print("=" * 60)
    print("📊 M1 度量结果")
    print("=" * 60)

    checked = metrics["total_checked"]
    fixed = metrics["total_fixed"]
    fix_rate = metrics["fix_rate"]
    broken = len(metrics["broken_terms"])
    already_ok = checked - fixed - broken

    print(f"""
┌────────────────────────────────────────────────┐
│  测试术语数:                {len(terms_with_freq):>4} 个                    │
│  (均在词典中存在 + 在语料中出现 ≥2次)             │
│                                                │
│  默认分词已正确:              {already_ok:>4} 个 ({round(already_ok/max(checked,1)*100):>4}%)              │
│  词典修复:                   {fixed:>4} 个 ({fix_rate:>4}%)              │
│  仍被切碎:                   {broken:>4} 个 ({round(broken/max(checked,1)*100):>4}%)              │
│                                                │
│  📐 术语修复率:            {fix_rate:>5.1f}%                      │
│     (词典能修复已切碎术语的比例)                    │
└────────────────────────────────────────────────┘
""")

    # 修复案例展示
    show_examples(metrics, all_text)

    # 保存
    output_path = DATA_DIR / "benchmark_results.csv"
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["term", "corpus_freq", "fixed", "broken"])
        for term, freq in terms_with_freq:
            fixed = term in metrics["fixed_terms"]
            brkn = term in metrics["broken_terms"]
            writer.writerow([term, freq, "TRUE" if fixed else "", "TRUE" if brkn else ""])

    print(f"💾 详细报告: {output_path}")

    # ============================================================
    # 基线摘要
    # ============================================================
    print(f"""
{'='*60}
📋 M1 基线度量摘要 (v2.3, {datetime.now().strftime('%Y-%m-%d')})
{'='*60}

  指标                        当前基线       目标
  ────────────────────────────────────────────────
  术语修复率                  {fix_rate:>5.1f}%        ≥ 90%
    (词典能修复被切碎的术语数/总术语数)
  默认可正确切分               {round(already_ok/max(checked,1)*100):>5.1f}%         —
    (不加载词典也能正确切分的比例)
  仍无法修复的术语              {broken:>3} 个        ≤ 10
    (需优化词典/补缺的高频术语)

{'-'*60}
  📌 解读：
  • {fix_rate}% 的术语加载词典后从"被切碎→正确切分"
  • {broken} 个术语仍无法修复 — 需要检查是否未加入 jieba_dict
{'='*60}
""")

    return metrics


if __name__ == "__main__":
    main()
