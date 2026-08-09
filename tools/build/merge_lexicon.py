"""合并种子词库 + 校验后的 PMI 候选词 → 七层完整词典 v1.0 + jieba 自定义词典。

运行方式:
    python scripts/nlp_dict/merge_lexicon.py
    python scripts/nlp_dict/merge_lexicon.py --category textile_chain  # 仅更新第三层
    python scripts/nlp_dict/merge_lexicon.py --export-jieba-only        # 仅重新导出 jieba 词典

依赖:
    pyyaml, sqlite3

输出:
    - scripts/nlp_dict/data/lexicon_v1.yaml      ← 七层完整词典 v1.0
    - scripts/nlp_dict/data/jieba_dict.txt       ← jieba 自定义词典格式
    - scripts/nlp_dict/data/lexicon_report.md    ← 词典构建报告（可读版）
"""

import sys
import csv
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml
import sqlite3


# ============================================================
# 配置
# ============================================================

DATA_DIR = Path(__file__).parent / "data"
SEED_PATH = DATA_DIR / "seed_lexicon.yaml"
PMI_PATH = DATA_DIR / "pmi_candidates.csv"
JIEBA_KW_PATH = DATA_DIR / "jieba_keywords.csv"
LEXICON_PATH = DATA_DIR / "lexicon_v2.yaml"
JIEBA_DICT_PATH = DATA_DIR / "jieba_dict.txt"
REPORT_PATH = DATA_DIR / "lexicon_report.md"

DB_PATH = Path(__file__).parent.parent.parent / "collector" / "policy.db"


# ============================================================
# 工具函数
# ============================================================


def load_csv_verified(path):
    """加载 CSV 文件，返回已通过的词条列表（manual_verdict = '✅'）。

    如果文件不存在，返回空列表（假定尚未人工校验）。
    """
    if not path.exists():
        print(f"⚠️  文件不存在: {path}（跳过，假定尚未人工校验）")
        return []

    verified = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verdict = (row.get('manual_verdict') or '').strip()
            if verdict in ('✅', '通过', 'accept'):
                term_key = 'ngram' if 'ngram' in row else 'term'
                verified.append(row[term_key])

    print(f"✅ 从 {path.name} 加载 {len(verified)} 个已通过词条")
    return verified


def load_yaml_safe(path):
    """加载 YAML 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data, path):
    """保存 YAML 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )
    print(f"💾 {path}")


def count_all_terms(lexicon):
    """递归统计词典中的唯一词条数。"""
    seen = set()

    def traverse(value):
        if isinstance(value, str) and len(value) >= 2:
            seen.add(value)
        elif isinstance(value, list):
            for item in value:
                traverse(item)
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, list):
                    for item in v:
                        traverse(item)
                elif isinstance(v, dict):
                    traverse(v)
                elif isinstance(v, str):
                    if k not in ('description', 'source', 'note', '产业定位', '政策类型'):
                        traverse(k)
                        traverse(v)

    traverse(lexicon)
    return len(seen)


def categorize_new_terms(terms, existing_terms, label):
    """将新词条按语义类别给人工分类提示。

    terms: set of new terms
    existing_terms: set of known terms (用于去重)
    label: 默认分类标签

    返回 {label: [sorted_terms]}
    """
    new = sorted(terms - existing_terms)

    # 简单启发式分类（待人工确认）
    result = {}

    # 纺织产业链相关
    textile_keywords = {
        '纺织', '服装', '棉', '麻', '丝', '毛', '化纤', '纤维',
        '纺', '织', '染', '印', '纱', '布', '面料', '家纺',
        '针织', '梭织', '经编', '纬编', '非织造', '无纺',
        '缝纫', '成衣', '制衣', '染整', '印花', '绣花',
        '氨纶', '涤纶', '锦纶', '腈纶', '丙纶', '维纶',
        '碳纤维', '芳纶', '超高分子量聚乙烯',
    }
    chain = [t for t in new if any(kw in t for kw in textile_keywords)]

    # 绿色合规
    green_keywords = {
        '碳', '排放', '能', '耗', '绿色', '环保', '清洁', '循环',
        '节能', '降碳', '低碳', '零碳', '碳中和', '碳达峰',
        '污染物', '废水', '废气', '固废', 'VOCs',
    }
    green = [t for t in new if any(kw in t for kw in green_keywords) and t not in chain]

    # 数字/智能
    digital_keywords = {
        '智能', '数字', 'AI', '数据', '互联网', '物联网', '区块链',
        '自动化', '信息化', '机器人', '5G', '大模型',
    }
    digital = [t for t in new if any(kw in t for kw in digital_keywords) and t not in chain]

    # 贸易/经济
    trade_keywords = {
        '出口', '进口', '关税', '退税', '贸易', '跨境', '海关',
        '外汇', '供应链', '产业链', '内外贸',
    }
    trade = [t for t in new if any(kw in t for kw in trade_keywords) and t not in chain]

    # 政策/行政
    policy_keywords = {
        '政策', '规划', '方案', '意见', '办法', '通知', '条例',
        '措施', '试点', '示范', '认定', '申报', '审批',
        '财政', '补贴', '资金', '税', '奖励', '扶持',
    }
    policy_terms_local = [t for t in new if any(kw in t for kw in policy_keywords)
                          and t not in chain and t not in trade]

    # 其他归为未分类
    classified = set(chain + green + digital + trade + policy_terms_local)
    other = [t for t in new if t not in classified]

    result['产业链实体（纺织/服装/面料）'] = chain
    result['绿色合规（碳/排放/节能）'] = green
    result['数字智能（AI/数字化/物联网）'] = digital
    result['贸易经济（出口/关税/退税）'] = trade
    result['政策行政（规划/方案/试点）'] = policy_terms_local
    result['待人工分类'] = other

    # 去除空类
    return {k: v for k, v in result.items() if v}


# ============================================================
# 从数据库提取补充术语（扩展第三至五层）
# ============================================================


def extract_db_supplement(conn):
    """从 policy.db 中提取可与种子词库互补的术语。

    策略：
    - 标题分词：高频术语（如"专项行动""高质量发展"）已在种子词库外
    - issuing_authority：发文机构中频繁出现的"等N部门"模式
    """
    supplement = {
        "layer_3_textile_chain": [],
        "layer_4_policy_semantics": [],
        "layer_5_cross_domain": [],
    }

    # 标题分词频率
    titles = conn.execute(
        "SELECT title FROM policy "
        "WHERE review_status = '通过' "
        "AND (textile_relevance = '直接' OR textile_relevance = '间接')"
    ).fetchall()

    # 标题常见政策焦点词
    focus_patterns = [
        (r'专项行动', 'layer_4_policy_semantics'),
        (r'高质量发展', 'layer_5_cross_domain'),
        (r'绿色制造', 'layer_5_cross_domain'),
        (r'智能制造示范', 'layer_5_cross_domain'),
        (r'数字化转型', 'layer_5_cross_domain'),
        (r'先进制造业', 'layer_5_cross_domain'),
        (r'专精特新', 'layer_5_cross_domain'),
        (r'产业集群', 'layer_5_cross_domain'),
        (r'内外贸一体化', 'layer_5_cross_domain'),
        (r'消费品', 'layer_3_textile_chain'),
        (r'轻工', 'layer_3_textile_chain'),
        (r'印染集聚', 'layer_3_textile_chain'),
        (r'产业用纺织品', 'layer_3_textile_chain'),
        (r'功能性纤维', 'layer_3_textile_chain'),
        (r'绿色纤维', 'layer_3_textile_chain'),
        (r'再生纤维', 'layer_3_textile_chain'),
        (r'生物基', 'layer_3_textile_chain'),
        (r'以旧换新', 'layer_5_cross_domain'),
        (r'零碳工厂', 'layer_5_cross_domain'),
        (r'碳足迹', 'layer_5_cross_domain'),
        (r'碳关税', 'layer_5_cross_domain'),
        (r'CBAM', 'layer_5_cross_domain'),
        (r'出口退税', 'layer_5_cross_domain'),
        (r'跨境电商', 'layer_5_cross_domain'),
        (r'循环经济', 'layer_5_cross_domain'),
        (r'新质生产力', 'layer_5_cross_domain'),
        (r'两新', 'layer_5_cross_domain'),
        (r'十五五', 'layer_4_policy_semantics'),
        (r'消费品以旧换新', 'layer_5_cross_domain'),
    ]

    seen = set()
    for title_row in titles:
        title = title_row[0]
        for pattern, layer in focus_patterns:
            match = re.search(pattern, title)
            if match and match.group() not in seen:
                seen.add(match.group())
                supplement[layer].append(match.group())

    return supplement


# ============================================================
# 生成 jieba 自定义词典
# ============================================================


def generate_jieba_dict(lexicon, output_path):
    """从七层词典生成 jieba 自定义词典文件。

    格式（每行）：词 词频 词性
    - 词频：按在政策和词典中出现频率（覆盖频率用 50）
    - 词性：使用 jieba 兼容标记
      - nz: 专有名词（机构/地名/产业链实体）
      - vn: 动名词（政策动作）
      - n: 普通名词
      - eng: 英文/缩写
    """
    pos_map = {
        # 机构实体 → nz（专有名词）
        'layer_1_agencies': 'nz',
        # 文书类型 → n（名词）
        'layer_2_document_types': 'n',
        # 产业链实体 → nz（行业专有名词）
        'layer_3_textile_chain': 'nz',
        # 政策语义 → vn（动名词）
        'layer_4_policy_semantics': 'vn',
        # 交叉领域 → n（名词）
        'layer_5_cross_domain': 'n',
        # 地理 → ns（地名）
        'layer_6_geography': 'ns',
        # 时间 → t（时间词）
        'layer_7_time_expressions': 't',
    }

    entries = {}  # {word: (freq, pos)}

    META_KEYS = {'description','source','note','terms','top10_provinces','date_range',
                 '产业定位','政策类型','policy_count','cluster_info','type','aliases','definition'}

    def extract_terms(value, default_pos='n', base_freq=50):
        if isinstance(value, str):
            w = value.strip()
            if w and len(w) >= 2:
                if w not in entries or base_freq > entries[w][0]:
                    entries[w] = (base_freq, default_pos)
        elif isinstance(value, list):
            for item in value:
                extract_terms(item, default_pos, base_freq)
        elif isinstance(value, dict):
            for k, v in value.items():
                if k in META_KEYS:
                    # 元数据键不视为术语，但递归提取其值
                    if isinstance(v, (dict, list)):
                        extract_terms(v, default_pos, base_freq)
                    elif isinstance(v, str):
                        extract_terms(v, default_pos, base_freq)
                else:
                    # 键本身可能是术语（如"棉花""化纤"）
                    if isinstance(k, str) and k.strip() and len(k.strip()) >= 2:
                        wk = k.strip()
                        if wk not in entries:
                            entries[wk] = (base_freq, default_pos)
                    # 递归提取值
                    if isinstance(v, dict):
                        # 机构实体、地理等嵌套结构
                        if 'aliases' in v:
                            aliases = v.get('aliases', [])
                            for alias in aliases:
                                clean = alias.strip()
                                if clean and len(clean) >= 2:
                                    wk = 80  # 机构名高频权重
                                    entries[clean] = (wk, default_pos)
                        # 含 definition 的术语条目
                        extract_terms(v, default_pos, base_freq)
                    elif isinstance(v, list):
                        extract_terms(v, default_pos, base_freq)
                    elif isinstance(v, str):
                        w = v.strip()
                        if w and len(w) >= 2:
                            if w not in entries or base_freq > entries[w][0]:
                                entries[w] = (base_freq, default_pos)

    layers = lexicon.get("layers", {})
    for layer_name, layer_data in layers.items():
        pos = pos_map.get(layer_name, 'n')
        terms_data = layer_data.get("terms", {})
        if isinstance(terms_data, dict):
            for category, items in terms_data.items():
                if isinstance(items, dict):
                    extract_terms(items, pos, 80)
                elif isinstance(items, list):
                    extract_terms(items, pos, 60)

    # 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 纺织行业政策 NLP 词典 — jieba 自定义词典\n")
        f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("# 格式: 词 词频 词性\n")
        f.write("# 词性: nz=专有名词, vn=动名词, n=普通名词, ns=地名, t=时间, eng=英文\n\n")
        for word, (freq, pos) in sorted(entries.items()):
            f.write(f"{word} {freq} {pos}\n")

    print(f"💾 jieba 自定义词典: {output_path}（{len(entries)} 个词条）")
    return len(entries)


# ============================================================
# 生成可读词典报告
# ============================================================


def generate_report(lexicon, pmi_count, jieba_kw_count, output_path):
    """生成 Markdown 格式的词典构建报告。"""

    layers = lexicon.get("layers", {})
    total_terms = lexicon.get("meta", {}).get("total_terms", 0)

    lines = [
        "# 纺织行业政策 NLP 词典 — 构建报告",
        "",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 词典版本：{lexicon.get('meta', {}).get('version', 'v1.0')}",
        f"> 收录术语：**{total_terms}** 个词条",
        "",
        "---",
        "",
        "## 词典概览",
        "",
        "| 层级 | 名称 | 词条数 | 来源 |",
        "|:--|:--|:--|:--|",
    ]

    for layer_key, layer_data in layers.items():
        name = layer_data.get("description", layer_key)
        source = layer_data.get("source", "—")
        # 数词条
        count = 0

        def count_layer(val):
            nonlocal count
            if isinstance(val, str) and len(val) >= 2:
                count += 1
            elif isinstance(val, list):
                for item in val:
                    count_layer(item)
            elif isinstance(val, dict):
                for k, v in val.items():
                    count_layer(v)

        terms = layer_data.get("terms", {})
        count_layer(terms)
        lines.append(f"| {layer_key} | {name} | {count} | {source} |")

    lines.extend([
        "",
        "## PMI 新词发现", "",
        f"- PMI 候选词总数：{pmi_count}",
        f"- jieba 高频词候选：{jieba_kw_count}",
        f"- 预计经人工校验后可纳入词典的新词：约 {pmi_count // 3}～{pmi_count // 2} 个",
        "",
        "## 使用方式",
        "",
        "### 加载 jieba 自定义词典",
        "```python",
        "import jieba",
        "jieba.load_userdict('data/jieba_dict.txt')",
        "# 此后 jieba 分词将识别词典中的行业术语",
        "# 例如 '印染' → 一个词，不会被切成 '印' + '染'",
        "```",
        "",
        "### 查询词典",
        "```python",
        "import yaml",
        "with open('data/lexicon_v1.yaml', 'r', encoding='utf-8') as f:",
        "    lexicon = yaml.safe_load(f)",
        "print(lexicon['layers']['layer_3_textile_chain']['terms'])",
        "```",
        "",
        "## 下一步",
        "",
        "1. 人工校验 PMI 候选词（`pmi_candidates.csv` → `manual_verdict` 列填 ✅/❌）",
        "2. 重新运行 `merge_lexicon.py` 纳入校验结果",
        "3. 在月报/论文中试用 jieba 自定义词典，收集分词错误反馈",
        "4. 积累反馈后进入 NER 训练数据标注（`ner/annotate.py`）",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📄 词典报告: {output_path}")


# ============================================================
# 主流程
# ============================================================


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="合并种子词库 + PMI 候选 → 完整词典 v1.0")
    parser.add_argument("--export-jieba-only", action="store_true",
                        help="仅重新导出 jieba 自定义词典（不重新合并）")
    parser.add_argument("--lexicon-path", type=str,
                        default=str(DATA_DIR / "lexicon_v2.yaml"),
                        help="词典文件路径（默认 lexicon_v2.yaml）")
    parser.add_argument("--category", choices=['textile_chain', 'cross_domain', 'all'],
                        default='all', help="仅更新指定分类（默认全部）")
    args = parser.parse_args()

    # 仅导出 jieba 模式
    if args.export_jieba_only:
        lexicon_path_to_use = Path(args.lexicon_path)
        if not lexicon_path_to_use.exists():
            print(f"❌ {lexicon_path_to_use} 不存在，请先运行完整合并流程")
            sys.exit(1)
        lexicon = load_yaml_safe(lexicon_path_to_use)
        n = generate_jieba_dict(lexicon, JIEBA_DICT_PATH)
        print(f"\n✅ 已重新导出 jieba 词典（{n} 词条）")
        return

    # 加载种子词库
    print("📂 加载种子词库...")
    if not SEED_PATH.exists():
        print("❌ seed_lexicon.yaml 不存在，请先运行 build_seed_lexicon.py")
        sys.exit(1)
    seed = load_yaml_safe(SEED_PATH)

    # 连接数据库
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    # ============================================================
    # Step 1: 加载并通过的 PMI 候选词
    # ============================================================

    print("\n📋 加载 PMI 候选词...")
    pmi_verified = load_csv_verified(PMI_PATH)
    jieba_verified = load_csv_verified(JIEBA_KW_PATH)

    # 也加载全部候选（用于报告计数）
    all_pmi = 0
    if PMI_PATH.exists():
        with open(PMI_PATH, "r", encoding="utf-8-sig") as f:
            all_pmi = sum(1 for _ in csv.DictReader(f))

    all_jieba_kw = 0
    if JIEBA_KW_PATH.exists():
        with open(JIEBA_KW_PATH, "r", encoding="utf-8-sig") as f:
            all_jieba_kw = sum(1 for _ in csv.DictReader(f))

    all_new_terms = set(pmi_verified + jieba_verified)
    print(f"   校验通过的新词总计: {len(all_new_terms)}")

    # ============================================================
    # Step 2: 数据库补充提取
    # ============================================================

    print("\n📂 从数据库补充提取...")
    db_supplement = extract_db_supplement(conn)

    for layer, terms in db_supplement.items():
        new_terms = [t for t in terms if t not in all_new_terms]
        print(f"   {layer}: {len(new_terms)} 个补充词（标题高频）")

    conn.close()

    # ============================================================
    # Step 3: 合并——扩充第三/四/五层
    # ============================================================

    # 收集所有已存在的词条用于去重
    existing_all = set()

    def collect_existing(value):
        if isinstance(value, str) and len(value) >= 2:
            existing_all.add(value)
        elif isinstance(value, list):
            for item in value:
                collect_existing(item)
        elif isinstance(value, dict):
            for k, v in value.items():
                collect_existing(v)

    for layer_name, layer_data in seed.get("layers", {}).items():
        collect_existing(layer_data.get("terms", {}))

    # 分类新词
    print(f"\n🔀 分类新词（共 {len(all_new_terms)} 个）...")
    categorized = categorize_new_terms(all_new_terms, existing_all, "unclassified")

    for cat, terms in categorized.items():
        if terms:
            print(f"   {cat}: {len(terms)} 词")
            for t in terms[:5]:  # 打印前 5 个示例
                print(f"     - {t}")
            if len(terms) > 5:
                print(f"     ... 等 {len(terms)} 个")

    # ============================================================
    # Step 4: 扩充种子词库第三层
    # ============================================================

    # 将新词扩充到对应层

    # 第三层：纺织产业链 → 扩展
    layer3 = seed["layers"]["layer_3_textile_chain"]["terms"]
    chain_terms = set(t for t in categorized.get('产业链实体（纺织/服装/面料）', []))
    if chain_terms:
        layer3["PMI新词（产业链）"] = sorted(chain_terms)
        print(f"\n🧵 第三层扩充: {len(chain_terms)} 个产业链新词")

    # 第四层：政策语义 → 扩展新词
    layer4 = seed["layers"]["layer_4_policy_semantics"]["terms"]
    policy_terms = set(t for t in categorized.get('政策行政（规划/方案/试点）', []))
    # 也加数据库补充
    policy_terms.update(db_supplement.get("layer_4_policy_semantics", []))
    if policy_terms:
        layer4["PMI新词（政策行政）"] = sorted(policy_terms)
        print(f"📋 第四层扩充: {len(policy_terms)} 个政策行政新词")

    # 第五层：交叉领域 → 扩展到已有类别
    layer5 = seed["layers"]["layer_5_cross_domain"]["terms"]
    green_terms = set(t for t in categorized.get('绿色合规（碳/排放/节能）', []))
    digital_terms = set(t for t in categorized.get('数字智能（AI/数字化/物联网）', []))
    trade_terms = set(t for t in categorized.get('贸易经济（出口/关税/退税）', []))
    cross_from_db = db_supplement.get("layer_5_cross_domain", [])

    # 全部合并到 layer_5 的新类别
    if green_terms:
        layer5["PMI新词_绿色合规"] = sorted(green_terms)
    if digital_terms:
        layer5["PMI新词_数字智能"] = sorted(digital_terms)
    if trade_terms:
        layer5["PMI新词_贸易经济"] = sorted(trade_terms)
    if cross_from_db:
        layer5["数据库标题补充"] = sorted(set(cross_from_db))

    cross_total = len(green_terms) + len(digital_terms) + len(trade_terms) + len(cross_from_db)
    if cross_total:
        print(f"🔄 第五层扩充: {cross_total} 个交叉领域新词")

    # 未分类的：暂放第五层等待人工
    other_terms = categorized.get('待人工分类', [])
    if other_terms:
        layer5["待人工分类"] = sorted(other_terms)
        print(f"❓ 待人工分类: {len(other_terms)} 词")

    # ============================================================
    # Step 5: 更新元数据并保存
    # ============================================================

    total = count_all_terms(seed)
    seed["meta"]["version"] = "v1.0"
    seed["meta"]["total_terms"] = total
    seed["meta"]["generated"] = datetime.now().strftime("%Y-%m-%d")
    seed["meta"]["pmi_verified_terms"] = len(all_new_terms)
    seed["meta"]["note"] = (
        f"种子词库 {seed['meta'].get('total_terms', '?')} 词 + "
        f"PMI 校验通过 {len(all_new_terms)} 词 + "
        f"数据库补充 {sum(len(v) for v in db_supplement.values())} 词"
    )

    dump_yaml(seed, LEXICON_PATH)

    # ============================================================
    # Step 6: 生成 jieba 自定义词典
    # ============================================================

    print(f"\n📖 生成 jieba 自定义词典...")
    n_jieba = generate_jieba_dict(seed, JIEBA_DICT_PATH)

    # ============================================================
    # Step 7: 生成可读报告
    # ============================================================

    print(f"\n📄 生成词典报告...")
    generate_report(seed, all_pmi, all_jieba_kw, REPORT_PATH)

    # ============================================================
    # 摘要
    # ============================================================

    print()
    print("=" * 60)
    print("📊 七层词典 v1.0 构建摘要")
    print("=" * 60)
    for layer_key, layer_data in seed["layers"].items():
        name = layer_data.get("description", layer_key)
        count = 0

        def c(val):
            nonlocal count
            if isinstance(val, str) and len(val) >= 2:
                count += 1
            elif isinstance(val, list):
                for item in val:
                    c(item)
            elif isinstance(val, dict):
                for k, v in val.items():
                    c(v)

        c(layer_data.get("terms", {}))
        print(f"   {layer_key:<30} {count:>5} 词  — {name}")

    print(f"\n   {'总计':<30} {total:>5} 词")
    print(f"   {'jieba 自定义词典':<30} {n_jieba:>5} 词条")
    print()
    print("下一步:")
    print("  1. 测试 jieba 词典: python -c \"import jieba; jieba.load_userdict('scripts/nlp_dict/data/jieba_dict.txt'); print('/'.join(jieba.cut('工业和信息化部发布纺织工业数字化转型三年行动方案')))\"")
    print("  2. 人工审阅 pmi_candidates.csv 中未分类词条")
    print("  3. 在 pmi_candidates.csv 的 manual_verdict 列补充 ✅ 后重新运行 merge_lexicon.py")
    print("  4. 积累使用反馈后进入 NER 训练数据标注阶段")


if __name__ == "__main__":
    main()
