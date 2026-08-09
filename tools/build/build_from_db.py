"""从 policy.db 全文运行 PMI 新词发现 + 高频术语提取。

双重策略：
  1. 字符级 N-Gram PMI 发现：发现 jieba 不知道的领域新词
  2. jieba 分词频率分析：发现领域内高频但不在种子词库的已知词

注意：此脚本在 Windows 终端可能遇到 GBK 编码问题，建议使用 PYTHONIOENCODING=utf-8 前缀运行:
    PYTHONIOENCODING=utf-8 python scripts/nlp_dict/build_from_db.py

运行方式:
    python scripts/nlp_dict/build_from_db.py
    python scripts/nlp_dict/build_from_db.py --min-freq 3 --min-pmi 10 --max-len 6 --entropy-threshold 1.2

依赖:
    jieba, pandas, pyyaml  (pip install jieba pandas pyyaml)

输出:
    - scripts/nlp_dict/data/pmi_candidates.csv   （PMI 候选词，含评分和频次）
    - scripts/nlp_dict/data/jieba_keywords.csv   （jieba 高频词，含 TF 和文档频率）

数据源:
    - collector/policy.db (policy 表 full_text 字段)
    - 筛选: review_status = '通过', textile_relevance IN ('直接', '间接'), full_text 非空
"""

import sys
import re
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jieba
import sqlite3


# ============================================================
# 配置
# ============================================================

DB_PATH = Path(__file__).parent.parent.parent / "collector" / "policy.db"
DATA_DIR = Path(__file__).parent / "data"
SEED_PATH = DATA_DIR / "seed_lexicon.yaml"

# jieba 初始化：使用精确模式
jieba.setLogLevel(20)  # 静默 jieba 日志


# ============================================================
# 数据获取
# ============================================================


def get_conn():
    """获取只读数据库连接。"""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_texts(conn):
    """获取所有通过审核的纺织相关政策的全文。"""
    rows = conn.execute("""
        SELECT policy_id, title, full_text, publish_date
        FROM policy
        WHERE review_status = '通过'
        AND (textile_relevance = '直接' OR textile_relevance = '间接')
        AND full_text IS NOT NULL AND full_text != ''
        ORDER BY publish_date DESC
    """).fetchall()
    return [(r["policy_id"], r["title"], r["full_text"], r["publish_date"]) for r in rows]


def clean_text(text):
    """清洗政策全文：去 HTML、URL、空白规范化、去 JS/CSS 残留。"""
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除 URL
    text = re.sub(r'https?://\S+', '', text)
    # 移除长串英文字母（JS/CSS 残留）
    text = re.sub(r'[a-zA-Z_][a-zA-Z0-9_\.]{20,}', '', text)
    # 移除连续数字串（非年份/日期的长数字）
    text = re.sub(r'\b\d{6,}\b', '', text)
    # 空白规范化
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================================
# 策略一：字符级 N-Gram PMI 新词发现
# ============================================================


def extract_chinese_chars(text):
    """提取纯中文字符序列。"""
    return re.sub(r'[^一-鿿]', '', text)


def generate_char_ngrams(chinese_text, max_len=6):
    """从纯中文文本生成字符级 N-gram（2 到 max_len 字）。"""
    ngrams = []
    for n in range(2, max_len + 1):
        for i in range(len(chinese_text) - n + 1):
            ngrams.append(chinese_text[i:i + n])
    return ngrams


def compute_pmi(ngram, ngram_freq, char_freq, total_chars):
    """计算 N-gram 的标准化 PMI 值（NPMI）。

    PMI = log( ngram_freq * total_chars / Π char_freq[wi] )
    NPMI = PMI / -log( ngram_freq / total_chars )

    NPMI 值域 [-1, 1]，可在不同长度 N-gram 间比较。

    返回 (pmi, npmi) 元组。
    """
    n = len(ngram)
    if n <= 1 or ngram_freq == 0:
        return (0.0, 0.0)

    chars_product = 1.0
    for char in ngram:
        cf = char_freq.get(char, 0)
        if cf == 0:
            return (0.0, 0.0)
        chars_product *= cf

    if chars_product == 0:
        return (0.0, 0.0)

    pmi = math.log(ngram_freq * total_chars / chars_product)
    # NPMI: 归一化到 [-1, 1]
    denom = -math.log(ngram_freq / total_chars) if ngram_freq < total_chars else 1.0
    npmi = pmi / denom if denom != 0 else 0.0

    return (round(pmi, 2), round(npmi, 4))


def has_domain_relevance(ngram, title_vocab):
    """过滤与纺织行业领域无关的 N-gram。

    返回 True 表示可能具有领域相关性。

    设计原则：
    - 3+ 字词：基本放行（长度本身就是很强的过滤）
    - 2 字词：必须至少包含一个域相关字，或整体命中域相关词表
    """
    n = len(ngram)

    # ========== 全局排除（所有长度） ==========

    # 政治/新闻套话
    for gp in ['习近平', '新时代', '中国特色', '社会主义', '中国梦',
                '伟大复兴', '以人民为中心', '党中央', '总书记',
                '脱贫攻坚', '共同富裕', '四个全面', '五位一体', '不忘初心']:
        if gp in ngram:
            return False

    # 通讯社/媒体残留
    if any(kw in ngram for kw in ['华社', '日电', '记者', '新华', '扫一扫', '语音播']):
        return False

    # 省/自治区/直辖市模板
    if any(kw in ngram for kw in ['省自治区直辖', '省自治区', '自治区直辖']):
        return False

    # ========== 3+ 字词 ==========
    if n >= 3:
        return True

    # ========== 2 字词：严格的域字匹配 ==========

    # 纺织产业链核心字
    textile_chars = set('纺织服装棉麻丝毛纤维纱布染印经编缝纫缫浆氨涤锦腈纶丙维碳芳')
    # 政策/经济/产业
    policy_chars = set('关税退税贸易跨境海关出口进口产销供需补贴减税降费扶持')
    # 绿色/环保
    green_chars = set('碳排能耗污废清洁绿节')
    # 制造/科技
    mfg_chars = set('制造产业品牌链群数字智创研发专利')
    # 标准化/质检
    std_chars = set('标准检测认证质量合规')
    # 纺织地名
    geo_chars = set('柯桥绍兴阿克苏石河子滨州')

    domain_chars = textile_chars | policy_chars | green_chars | mfg_chars | std_chars | geo_chars
    if any(c in ngram for c in domain_chars):
        return True

    # 完整的 2 字域词（更精确的匹配）
    domain_words_2char = {
        '纺织', '服装', '家纺', '面料', '化纤', '纤维', '棉花', '纱线',
        '棉纱', '织物', '针织', '梭织', '经编', '纬编',
        '印染', '染整', '缫丝', '缝纫', '制衣', '成衣', '轻工',
        '氨纶', '涤纶', '锦纶', '腈纶', '丙纶', '维纶', '丝绸',
        '毛纺', '麻纺', '絮用', '纺机',
        '出口', '进口', '关税', '退税', '贸易', '跨境', '海关', '外贸',
        '碳排', '碳达', '碳中', '零碳', '低碳', '双碳', '碳税', '碳足',
        '排污', '废水', '废气', '固废',
        '品牌', '专利', '技能', '培训',
        '会展', '非遗', '文旅',
        '小额', '融资', '贷款', '担保', '外贸', '关税', '汇率',
    }
    if ngram in domain_words_2char:
        return True

    # 在标题中频繁出现的：额外保留
    if ngram in title_vocab:
        return True

    return False


def compute_boundary_entropy(ngram, texts_sample, forward=True, max_sample=80):
    """计算左邻字熵（forward=False）或右邻字熵（forward=True）。

    邻字熵越高说明该 N-gram 的边界越自由 → 越可能是一个独立的词。
    熵低说明前后搭配固定 → 可能只是更大短语的一部分。
    """
    neighbors = Counter()
    for text in texts_sample[:max_sample]:
        chinese = extract_chinese_chars(text)
        for match in re.finditer(re.escape(ngram), chinese):
            if forward:
                # 右邻字
                pos = match.end()
                if pos < len(chinese):
                    neighbors[chinese[pos]] += 1
            else:
                # 左邻字
                pos = match.start() - 1
                if pos >= 0:
                    neighbors[chinese[pos]] += 1

    total = sum(neighbors.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in neighbors.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy


# ============================================================
# 策略二：jieba 分词频率分析
# ============================================================


def jieba_keyword_analysis(texts_data):
    """用 jieba 对全部政策文本分词，统计高频词。

    返回 (term_freq, doc_freq)：
      - term_freq: {词: 总出现次数}
      - doc_freq: {词: 出现在多少篇文档中}

    按 CLAUDE.md 中 jieba 自定义词典的构思，先加载种子词库中
    已知术语作为用户词典，提高分词准确性。
    """
    # 加载种子词库作为 jieba 用户词典（如果有的话）
    seed_terms = _load_seed_terms()
    for term in seed_terms:
        jieba.add_word(term, freq=50)  # 较高权重确保不被切碎

    term_freq = Counter()
    doc_freq = Counter()

    for _, _, text, _ in texts_data:
        cleaned = clean_text(text)
        words = jieba.lcut(cleaned)
        # 只保留中文词，长度 >= 2
        chinese_words = [
            w.strip() for w in words
            if len(w.strip()) >= 2
            and re.match(r'^[一-鿿]+$', w.strip())
        ]
        term_freq.update(chinese_words)
        # 文档频率：每个唯一词在每个文档中计 1 次
        doc_freq.update(set(chinese_words))

    return term_freq, doc_freq


def _load_seed_terms():
    """从 seed_lexicon.yaml 加载所有已知术语。"""
    if not SEED_PATH.exists():
        return set()

    import yaml
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        seed = yaml.safe_load(f)

    terms = set()

    def extract(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, list):
                    terms.update(t for t in v if isinstance(t, str) and len(t) >= 2)
                elif isinstance(v, dict):
                    extract(v)
                elif isinstance(v, str) and k != "description" and k != "source" and k != "note":
                    if len(k) >= 2:
                        terms.add(k)
        elif isinstance(obj, list):
            for item in obj:
                extract(item)

    layers = seed.get("layers", {})
    for layer_name, layer_data in layers.items():
        terms_data = layer_data.get("terms", {})
        if isinstance(terms_data, dict):
            extract(terms_data)
        elif isinstance(terms_data, list):
            terms.update(t for t in terms_data if isinstance(t, str) and len(t) >= 2)

    return terms


# ============================================================
# 候选词后处理
# ============================================================


def is_meaningful_ngram(ngram):
    """过滤无意义的 N-gram。

    排除：
      - 重复单字组成的（如"的的的"）
      - 以虚词/数词开头的（如"的第一""三个"）
    """
    # 全是同一字符
    if len(set(ngram)) == 1:
        return False

    # 以常见虚词/数词开头
    function_words_start = {
        '的', '了', '在', '是', '和', '与', '及', '或',
        '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
        '这', '那', '其', '之', '等', '第', '从', '对', '被', '把',
        '也', '就', '都', '而', '但', '却', '所', '为', '以',
    }
    if ngram[0] in function_words_start:
        return False

    return True


# ============================================================
# 曝光度统计（新增：按季度统计术语出现趋势）
# ============================================================


def compute_trend(ngram, texts_data, max_sample=100):
    """计算 N-gram 在最近 3 个月 vs 前 3 个月的出现频率变化。

    返回趋势标签: '📈上升' / '📉下降' / '➡平稳' / '🆕新词'
    """
    # 按日期分组统计
    recent_count = 0
    older_count = 0
    recent_chars = 0
    older_chars = 0

    # 找到中位日期
    dates = sorted(set(d for _, _, _, d in texts_data if d))
    if len(dates) < 4:
        return '➡平稳（数据不足）'

    median_idx = len(dates) // 2

    for _, _, text, date in texts_data[:max_sample]:
        if not date:
            continue
        chinese = extract_chinese_chars(text)
        count = chinese.count(ngram)
        chars = len(chinese)
        if date >= dates[median_idx]:
            recent_count += count
            recent_chars += chars
        else:
            older_count += count
            older_chars += chars

    if older_count == 0 and recent_count == 0:
        return '➡平稳'
    if older_count == 0 and recent_count > 0:
        return '🆕近期新出现'
    if older_count > 0 and recent_count == 0:
        return '📉近期消失'

    recent_rate = recent_count / max(recent_chars, 1)
    older_rate = older_count / max(older_chars, 1)

    if recent_rate > older_rate * 1.5:
        return '📈上升'
    elif recent_rate < older_rate * 0.5:
        return '📉下降'
    else:
        return '➡平稳'


# ============================================================
# 主流程
# ============================================================


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="PMI 新词发现与高频术语提取")
    parser.add_argument("--min-freq", type=int, default=3,
                        help="PMI 候选词最小频次（默认 3）")
    parser.add_argument("--min-pmi", type=float, default=10,
                        help="PMI 最小阈值（默认 10，越低候选越多）")
    parser.add_argument("--max-len", type=int, default=6,
                        help="N-Gram 最长字数（默认 6）")
    parser.add_argument("--entropy-threshold", type=float, default=1.2,
                        help="邻字熵最小阈值（默认 1.2，越高要求越严）")
    parser.add_argument("--top-n", type=int, default=500,
                        help="输出的候选词数量上限（默认 500）")
    args = parser.parse_args()

    conn = get_conn()
    texts_data = get_texts(conn)
    conn.close()

    print(f"📖 读取 {len(texts_data)} 篇政策全文")

    # 清洗文本
    cleaned_texts = []
    for pid, title, ft, pdate in texts_data:
        ct = clean_text(ft)
        if ct and len(ct) > 100:  # 至少 100 字符
            cleaned_texts.append((pid, title, ct, pdate))

    total_chars = sum(len(c[2]) for c in cleaned_texts)
    total_chinese = sum(len(extract_chinese_chars(c[2])) for c in cleaned_texts)
    print(f"   清洗后有效文本: {len(cleaned_texts)} 篇")
    print(f"   总字符数: {total_chars:,}（中文: {total_chinese:,}）")
    print(f"   时间范围: {cleaned_texts[-1][3]} ~ {cleaned_texts[0][3]}")
    print()

    # ============================================================
    # 策略一：字符级 N-Gram PMI
    # ============================================================

    print("=" * 60)
    print("🔍 策略一：字符级 N-Gram PMI 新词发现")
    print("=" * 60)

    # 统计所有 N-gram 和字符频率
    all_ngrams = Counter()
    char_freq = Counter()

    for _, _, text, _ in cleaned_texts:
        chinese = extract_chinese_chars(text)
        ngrams = generate_char_ngrams(chinese, args.max_len)
        all_ngrams.update(ngrams)
        for char in chinese:
            char_freq[char] += 1

    total_ngram_count = sum(all_ngrams.values())
    total_chars = sum(char_freq.values())
    print(f"\n📊 N-gram 统计:")
    print(f"   独立 N-gram 数: {len(all_ngrams):,}")
    print(f"   总出现次数: {total_ngram_count:,}")

    # 频次过滤
    candidates = {
        ng: freq for ng, freq in all_ngrams.items()
        if freq >= args.min_freq and is_meaningful_ngram(ng)
    }
    print(f"   频次 >= {args.min_freq} 且有意义的候选: {len(candidates):,}")

    # 计算 NPMI & 域相关性过滤
    print(f"\n📐 计算 NPMI + 域相关性过滤...")
    # 构建标题词表
    title_vocab = set()
    for _, title, _, _ in cleaned_texts:
        for term in jieba.lcut(title):
            if len(term) >= 2:
                title_vocab.add(term)

    pmi_results = []
    for ng, freq in candidates.items():
        pmi_val, npmi_val = compute_pmi(ng, freq, char_freq, total_chars)
        # 更严格的 NPMI 阈值
        if len(ng) == 2 and npmi_val < 0.65:
            continue
        if len(ng) >= 3 and npmi_val < 0.55:
            continue
        # 域相关性必须通过（且不再仅靠标题词表豁免）
        if not has_domain_relevance(ng, title_vocab):
            continue
        pmi_results.append({
            'ngram': ng,
            'length': len(ng),
            'frequency': freq,
            'pmi': pmi_val,
            'npmi': npmi_val,
        })

    # 排序：3+字词优先，然后按 NPMI
    pmi_results.sort(key=lambda x: (0 if x['length'] >= 3 else 1, -x['npmi'], -x['frequency']))
    top_n_limit = args.top_n * 2
    pmi_results = pmi_results[:top_n_limit]
    print(f"   域相关 + 筛选后: {len(pmi_results)} 个候选")

    # 计算邻字熵（对 Top 候选抽样计算，采样 50 篇文本提速）
    if pmi_results:
        print(f"\n📏 计算邻字熵（Top {min(len(pmi_results), 100)} 候选抽样，采样 50 篇文本）...")
        top_for_entropy = sorted(pmi_results, key=lambda x: -x['pmi'])[:100]
        sample_texts = [c[2] for c in cleaned_texts[:50]]  # 只采样 50 篇
        for r in top_for_entropy:
            left_ent = compute_boundary_entropy(r['ngram'], sample_texts, forward=False)
            right_ent = compute_boundary_entropy(r['ngram'], sample_texts, forward=True)
            r['left_entropy'] = round(left_ent, 2)
            r['right_entropy'] = round(right_ent, 2)
            r['min_entropy'] = round(min(left_ent, right_ent), 2)
            r['pass_entropy'] = min(left_ent, right_ent) >= args.entropy_threshold

    # 加载种子词库判断重叠
    seed_terms = _load_seed_terms()
    print(f"\n📚 种子词库已有 {len(seed_terms)} 个术语")

    for r in pmi_results:
        r['in_seed'] = r['ngram'] in seed_terms
        # 也检查是否被种子词库中的某个词包含
        r['contains_seed'] = any(
            s in r['ngram'] or r['ngram'] in s
            for s in seed_terms if len(s) >= 2
        ) if not r['in_seed'] else False

    # 分类：新词 vs 已知
    new_pmi = [r for r in pmi_results if not r['in_seed']]
    print(f"   扣除种子词库已有的: 净增 {len(new_pmi)} 个（共 {len(pmi_results)}）")

    # 按综合评分排序：PMI + 频率加权
    for r in new_pmi:
        r['score'] = round(r['npmi'] * math.log(r['frequency'] + 1), 2)
    new_pmi.sort(key=lambda x: (-x['npmi'], -x['frequency']))

    # ============================================================
    # 策略二：jieba 分词频率分析
    # ============================================================

    print()
    print("=" * 60)
    print("🔍 策略二：jieba 分词频率分析")
    print("=" * 60)

    term_freq, doc_freq = jieba_keyword_analysis(cleaned_texts)
    total_terms = sum(term_freq.values())
    print(f"\n📊 jieba 分词统计:")
    print(f"   独立词数: {len(term_freq):,}")
    print(f"   总词数: {total_terms:,}")

    # 按文档频率过滤（至少出现在 min_freq 篇文档中）
    jieba_results = []
    for term, tf in term_freq.items():
        df = doc_freq.get(term, 0)
        if df >= args.min_freq and len(term) >= 2:
            jieba_results.append({
                'term': term,
                'length': len(term),
                'term_freq': tf,
                'doc_freq': df,
                'tf_per_1000': round(tf / total_terms * 1000, 2),
                'in_seed': term in seed_terms,
            })

    jieba_results.sort(key=lambda x: (-x['doc_freq'], -x['term_freq']))
    new_jieba = [r for r in jieba_results if not r['in_seed']]
    print(f"   文档频率 >= {args.min_freq}: {len(jieba_results)} 个")
    print(f"   扣除种子词库: 净增 {len(new_jieba)} 个")

    # ============================================================
    # 合并结果保存
    # ============================================================

    print()
    print("=" * 60)
    print("💾 保存结果")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 保存 PMI 候选词 CSV
    pmi_output = DATA_DIR / "pmi_candidates.csv"
    import csv

    with open(pmi_output, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            'ngram', 'length', 'frequency', 'pmi', 'score',
            'left_entropy', 'right_entropy', 'min_entropy', 'pass_entropy',
            'in_seed', 'contains_seed', 'manual_verdict'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        # 写入：pass_entropy 的优先
        pass_ent = [r for r in new_pmi if r.get('pass_entropy')]
        fail_ent = [r for r in new_pmi if not r.get('pass_entropy')]
        for r in pass_ent[:args.top_n] + fail_ent[:args.top_n // 2]:
            r['manual_verdict'] = ''  # 人工校验留空
            writer.writerow(r)

    print(f"✅ PMI 候选词: {pmi_output}")
    pmi_saved = min(len(new_pmi), args.top_n + args.top_n // 2)

    # 保存 jieba 高频词 CSV
    jieba_output = DATA_DIR / "jieba_keywords.csv"
    with open(jieba_output, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            'term', 'length', 'term_freq', 'doc_freq', 'tf_per_1000',
            'in_seed', 'manual_verdict'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in new_jieba[:args.top_n]:
            r['manual_verdict'] = ''
            writer.writerow(r)

    print(f"✅ jieba 高频词: {jieba_output}")

    # ============================================================
    # 终端输出摘要
    # ============================================================

    print()
    print("=" * 60)
    print("📊 PMI 新词 Top 30")
    print("=" * 60)
    print(f"{'候选词':<12} {'字':<4} {'频次':<8} {'PMI':<8} {'评分':<8} {'熵':<8} {'种子'}")
    print("-" * 65)
    for r in new_pmi[:30]:
        in_seed_mark = '✅' if r['in_seed'] else '  '
        ent_mark = f"{r.get('min_entropy', 'N/A')}"
        print(f"{r['ngram']:<12} {r['length']:<4} {r['frequency']:<8} "
              f"{r['pmi']:<8.2f} {r.get('score', 0):<8.2f} {ent_mark:<8} {in_seed_mark}")

    print()
    print("=" * 60)
    print("📊 jieba 高频词 Top 20（不在种子词库中）")
    print("=" * 60)
    print(f"{'术语':<16} {'字':<4} {'词频':<8} {'文档数':<8} {'permille'}")
    print("-" * 50)
    for r in new_jieba[:20]:
        print(f"{r['term']:<16} {r['length']:<4} {r['term_freq']:<8} "
              f"{r['doc_freq']:<8} {r['tf_per_1000']}")

    # 长度分布
    print()
    print("📏 PMI 新词长度分布:")
    by_len = Counter(r['length'] for r in new_pmi)
    for l in sorted(by_len):
        bar = '█' * (by_len[l] // max(1, max(by_len.values()) // 20))
        print(f"   {l}字词: {by_len[l]:>4} {bar}")

    print()
    print("—" * 60)
    print(f"📦 总计:")
    print(f"   PMI 新词候选: {len(new_pmi)} 个（含熵过滤通过 {len(pass_ent)} 个）")
    print(f"   jieba 高频词候选: {len(new_jieba)} 个")
    print(f"   种子词库已有: {len(seed_terms)} 个")
    print()
    print("下一步:")
    print("  1. (人工) 审阅 pmi_candidates.csv 和 jieba_keywords.csv")
    print("  2. 在 manual_verdict 列标注 ✅ 通过 / ❌ 拒绝 / 🔀 合并")
    print("  3. 保存审阅结果（推荐覆盖原文件或保存为 pmi_verified.csv）")
    print("  4. 运行 merge_lexicon.py 合并生成完整词典 v1.0 + jieba 自定义词典")

    return new_pmi, new_jieba


if __name__ == "__main__":
    main()
