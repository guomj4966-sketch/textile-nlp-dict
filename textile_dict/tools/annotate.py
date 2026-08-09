"""半自动 NER 标注工具：加载词典 → 在政策文本中预标注已知术语 → 导出 BIO 格式训练数据。

工作流程：
  1. 加载 lexicon_v2.yaml → 构建实体类型映射（agency/doc_type/textile_term/etc.）
  2. 从 policy.db 取政策全文 → 逐句切分
  3. 词典匹配预标注（自动打 BIO 标签）
  4. 对未标注的句段，展示上下文供人工确认
  5. 导出 ner_train.jsonl（可被 HuggingFace datasets 直接加载）

运行方式:
    python scripts/nlp_dict/ner/annotate.py
    python scripts/nlp_dict/ner/annotate.py --sample-size 50 --output data/ner_train.jsonl
    python scripts/nlp_dict/ner/annotate.py --interactive  # 逐句交互式修正

依赖:
    jieba, pyyaml, sqlite3

输出:
    - scripts/nlp_dict/data/ner_train.jsonl   ← BIO 标注训练数据
    - scripts/nlp_dict/data/ner_stats.json    ← 标注统计（实体类型分布/数量）

BIO 格式说明:
    B-AGENCY    = 发文机关实体开头
    I-AGENCY    = 发文机关实体内部
    B-DOC_TYPE  = 政策文书类型开头
    I-DOC_TYPE  = 政策文书类型内部
    B-TEXTILE   = 纺织产业链实体开头
    I-TEXTILE   = 纺织产业链实体内部
    B-POLICY    = 政策语义标签开头
    I-POLICY    = 政策语义标签内部
    B-GEO       = 地理/产业集群开头
    I-GEO       = 地理/产业集群内部
    B-TIME      = 时间表达开头
    I-TIME      = 时间表达内部
    O           = 非实体
"""

import sys
import re
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import jieba
import yaml
import sqlite3

jieba.setLogLevel(20)

# ============================================================
# 配置
# ============================================================

DB_PATH = Path(__file__).parent.parent.parent.parent / "collector" / "policy.db"
DATA_DIR = Path(__file__).parent.parent / "data"
LEXICON_PATH = DATA_DIR / "lexicon_v2.yaml"
JIEBA_DICT_PATH = DATA_DIR / "jieba_dict.txt"

# 加载 jieba 词典
if JIEBA_DICT_PATH.exists():
    jieba.load_userdict(str(JIEBA_DICT_PATH))

# 实体类型 ← 词典层级映射
LAYER_TO_ENTITY = {
    "layer_1_agencies": "AGENCY",
    "layer_2_document_types": "DOC_TYPE",
    "layer_3_textile_chain": "TEXTILE",
    "layer_4_policy_semantics": "POLICY",
    "layer_5_cross_domain": "POLICY",  # 交叉领域归入政策语义
    "layer_6_geography": "GEO",
    "layer_7_time_expressions": "TIME",
}


# ============================================================
# 词典 → 实体词表（带优先级）
# ============================================================


def build_entity_vocab():
    """从七层词典构建实体词表。

    返回 {term: (entity_type, priority)}
    priority: 越长的词优先匹配（避免"发展"覆盖"高质量发展"）
    """
    if not LEXICON_PATH.exists():
        print(f"⚠️  词典不存在: {LEXICON_PATH}")
        return {}

    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        lexicon = yaml.safe_load(f)

    vocab = {}  # {term: (entity_type, layer_name)}

    for layer_key, layer_data in lexicon.get("layers", {}).items():
        entity_type = LAYER_TO_ENTITY.get(layer_key, "O")
        if entity_type == "O":
            continue

        terms_data = layer_data.get("terms", {})

        def extract_terms(data, etype, layer):
            if isinstance(data, str):
                w = data.strip()
                if len(w) >= 2 and not _is_overly_generic(w):
                    # 保留已有或使用更具体的 entity type
                    if w not in vocab or layer in ("layer_1_agencies", "layer_3_textile_chain"):
                        vocab[w] = (etype, layer)
            elif isinstance(data, list):
                for item in data:
                    extract_terms(item, etype, layer)
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict) and "aliases" in v:
                        # 机构实体：别名也映射到 AGENCY
                        aliases = v.get("aliases", [])
                        for alias in aliases:
                            clean = alias.strip()
                            if len(clean) >= 2:
                                vocab[clean] = (etype, layer)
                    elif isinstance(v, list):
                        extract_terms(v, etype, layer)
                    elif isinstance(v, dict):
                        extract_terms(v, etype, layer)

        extract_terms(terms_data, entity_type, layer_key)

    return vocab


# ============================================================
# 过于通用的词过滤（防止"发展""产业""提升"等词作为实体标注）
# ============================================================

GENERIC_TERMS = {
    '发展', '建设', '提升', '实施', '重点', '推进', '领域',
    '政策', '工作', '支持', '服务', '企业', '开展', '国家',
    '优化', '相关', '关于', '管理', '市场', '创新', '实现',
    '形成', '推动', '促进', '加强', '完善', '持续', '进一步',
    '不断', '加大', '提高', '增强', '加快', '深化', '强化',
    '转型', '升级', '培育', '保障', '改善',
    '增长', '打造', '示范', '试点', '推广',
    '具有', '进行', '提供', '包括', '使用', '采用',
    '组织', '开展', '负责', '协调', '建立', '制定',
    '科技', '环境', '数据', '网络', '技术',
    '材料', '质量', '品牌', '协同', '体系', '功能',
    '装备', '智能', '智慧', '数字', '生态', '保护',
    '零售', '便利', '特殊', '美丽', '改革', '辐射',
    '污染', '洗涤', '废弃', '海洋', '温室', '废物',
    '洗毛', '蚕丝', '纸浆', '石墨', '棉垛',
    # 第五层 cross-cutting 单字/通用词
    '碳', '标准', '规范', '绿色', '环保', '节能', '清洁',
    '循环', '排放', '低碳', '能耗', '减排',
    '数字化', '数字', '智能制造', '降碳', '跨境',
    '高质量', '高水平', '现代化', '新型', '综合',
    '专项行动', '专项', '行动', '名单', '目录',
    '高质量发展', '绿色发展', '科技创新',
}


def _is_overly_generic(term):
    """判断一个术语是否过于通用，不应被标注为实体。"""
    if term in GENERIC_TERMS:
        return True
    # 单字
    if len(term) < 2:
        return True
    return False


# ============================================================
# 句子切分
# ============================================================


def split_sentences(text):
    """将全文切分为标注友好的句子。

    按段落先分，再按句号/分号/换行切分。保留长度 10-200 字的句段。
    """
    paragraphs = re.split(r'\n+', text)
    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 5:
            continue
        # 切句
        sub_sents = re.split(r'[。；;]', para)
        for s in sub_sents:
            s = s.strip()
            if 10 <= len(s) <= 250:
                sentences.append(s)

    return sentences


# ============================================================
# 词典匹配 → BIO 标注
# ============================================================


def bio_annotate(sentence, vocab):
    """对句子进行基于词典的 BIO 标注。

    策略：按词长降序匹配（长词优先），已标注位置跳过。
    增强：不仅匹配术语本身，也尝试匹配术语的子串（避免
    像"绿色工厂"被先标了"绿色"再被"绿色工厂"覆盖的问题实际已由长词优先保证）。

    返回: [(char, label), ...] 字符级 BIO 标签
    """
    chars = list(sentence)
    labels = ["O"] * len(chars)

    # 按词长降序排序术语（长词优先匹配）
    sorted_terms = sorted(vocab.keys(), key=lambda t: -len(t))

    for term in sorted_terms:
        entity_type, layer = vocab[term]

        # 在句子中找所有出现位置（包括部分重叠）
        pos = 0
        while pos < len(chars):
            pos = sentence.find(term, pos)
            if pos == -1:
                break

            # 检查该区间是否全部为 O（未被标注），或者被更短的词部分标注
            # 只标注未被占用的区间
            if all(l == "O" for l in labels[pos:pos + len(term)]):
                labels[pos] = f"B-{entity_type}"
                for i in range(pos + 1, pos + len(term)):
                    labels[i] = f"I-{entity_type}"

            pos += 1  # 继续找下一个出现位置

    return list(zip(chars, labels))


# ============================================================
# 标注数据导出
# ============================================================


def bio_to_jsonl(sentences, vocab, output_path, max_sentences=None):
    """对所有句子执行 BIO 标注，导出为 JSONL 格式。

    每行: {"tokens": [...], "labels": [...]}
    """
    annotated = []
    stats = Counter()

    for i, sent in enumerate(sentences):
        if max_sentences and i >= max_sentences:
            break

        char_labels = bio_annotate(sent, vocab)
        tokens = []
        labels = []

        for char, label in char_labels:
            tokens.append(char)
            labels.append(label)
            if label != "O":
                stats[label] += 1

        # 只保留至少有 1 个实体的句子（对训练更有价值）
        if any(l != "O" for l in labels):
            annotated.append({"tokens": tokens, "labels": labels})

    # 导出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ann in annotated:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    return annotated, stats


# ============================================================
# 标注质量报告
# ============================================================


def generate_stats(annotated, stats, output_path):
    """生成标注统计报告。"""
    # 实体类型分布
    entity_counts = defaultdict(int)
    for label, count in stats.items():
        if label.startswith("B-"):
            entity_type = label[2:]
            entity_counts[entity_type] += count

    report = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_sentences": len(annotated),
        "total_tokens": sum(len(a["tokens"]) for a in annotated),
        "entity_counts_by_type": dict(entity_counts),
        "total_entity_tokens": sum(stats.values()),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


# ============================================================
# 交互式人工修正（预留接口）
# ============================================================


def interactive_review(annotated):
    """交互式审阅标注结果，允许人工修正。

    使用方法：终端显示标注后的句子，用键盘修正错误。
    这是标注工具的"半自动"核心——大部分由词典自动标注，
    少量由人工修正。
    """
    print(f"\n📝 交互式审阅模式（{len(annotated)} 句）")
    print("  命令: y=接受  n=跳过  s=修改标签后保存  q=退出")
    print()

    reviewed = []
    for i, ann in enumerate(annotated):
        tokens = ann["tokens"]
        labels = ann["labels"]

        # 用颜色/标记显示实体
        display = ""
        j = 0
        while j < len(tokens):
            if labels[j].startswith("B-"):
                entity_type = labels[j][2:]
                entity_chars = []
                k = j
                while k < len(tokens) and labels[k].startswith("I-"):
                    entity_chars.append(tokens[k])
                    k += 1
                entity_text = "".join([tokens[j]] + entity_chars)
                display += f"【{entity_type}:{entity_text}】"
                j = k + 1 if k > j else j + 1
            else:
                display += tokens[j]
                j += 1

        print(f"\n[{i+1}/{len(annotated)}]")
        print(f"  {display}")

        cmd = input("  (y/n/q) > ").strip().lower()

        if cmd == "q":
            break
        elif cmd == "y":
            reviewed.append(ann)
        elif cmd == "n":
            continue
        # 's' 修改模式：调用编辑器暂不实现

    return reviewed


# ============================================================
# 主流程
# ============================================================


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="半自动 NER 标注工具")
    parser.add_argument("--sample-size", type=int, default=80,
                        help="标注句子数上限（默认 80，建议初期 50-100）")
    parser.add_argument("--output", default="data/ner_train.jsonl",
                        help="输出 JSONL 路径")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式审阅模式（逐句确认/修正）")
    args = parser.parse_args()

    # 1. 构建实体词表
    print("📚 构建实体词表...")
    vocab = build_entity_vocab()
    print(f"   词表: {len(vocab)} 条")

    # 统计各实体类型的词条数
    type_counts = Counter(v[0] for v in vocab.values())
    for etype, count in sorted(type_counts.items()):
        print(f"     {etype}: {count} 条")
    print()

    # 2. 从数据库取政策文本
    print("📖 读取政策全文...")
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # 优先取直接相关的，再补充间接（全量）
    direct = conn.execute("""
        SELECT policy_id, title, full_text
        FROM policy
        WHERE review_status = '通过'
        AND textile_relevance = '直接'
        AND full_text IS NOT NULL AND full_text != ''
        ORDER BY publish_date DESC
    """).fetchall()

    # 全量补充：直接 + 间接
    indirect = conn.execute("""
        SELECT policy_id, title, full_text
        FROM policy
        WHERE review_status = '通过'
        AND textile_relevance = '间接'
        AND full_text IS NOT NULL AND full_text != ''
        ORDER BY publish_date DESC
    """).fetchall()

    rows = list(direct) + list(indirect)
    conn.close()
    print(f"   选取 {len(rows)} 条政策（直接 {len(direct)} + 间接 {len(indirect)}）")

    # 3. 切句
    print("🔪 句子切分...")
    all_sentences = []
    for row in rows:
        title = row["title"] or ""
        full_text = row["full_text"] or ""
        combined = title + "\n" + full_text
        sentences = split_sentences(combined)
        all_sentences.extend(sentences)

    print(f"   共 {len(all_sentences)} 个句段")

    # 4. BIO 标注
    print("🏷️  BIO 标注...")
    output_filename = args.output.replace("data/", "").replace("data\\", "")
    output_path = DATA_DIR / output_filename
    stats_path = DATA_DIR / "ner_stats.json"

    annotated, stats = bio_to_jsonl(
        all_sentences, vocab, output_path,
        max_sentences=args.sample_size
    )

    print(f"   标注句子: {len(annotated)}（至少含 1 个实体）")
    print(f"   实体 token 数: {sum(stats.values())}")

    # 5. 统计
    report = generate_stats(annotated, stats, stats_path)

    print()
    print("=" * 60)
    print("📊 标注统计")
    print("=" * 60)
    print(f"总句数:    {report['total_sentences']}")
    print(f"总 token:  {report['total_tokens']}")
    print(f"实体 token: {report['total_entity_tokens']}")
    print()
    print("实体类型分布:")
    for etype, count in sorted(report["entity_counts_by_type"].items(), key=lambda x: -x[1]):
        bar = "█" * (count // 20) if count >= 20 else "▏" * count
        print(f"  {etype:<10} {count:>5} {bar}")

    # 6. 展示部分标注结果
    print()
    print("=" * 60)
    print("📋 标注样本预览（前 10 句）")
    print("=" * 60)
    for i, ann in enumerate(annotated[:10]):
        tokens = ann["tokens"]
        labels = ann["labels"]

        # 构建高亮显示
        display = ""
        j = 0
        while j < len(tokens):
            if labels[j].startswith("B-"):
                entity_type = labels[j][2:]
                entity_chars = []
                k = j
                while k < len(tokens) and labels[k].startswith("I-"):
                    entity_chars.append(tokens[k])
                    k += 1
                entity_chars_flat = "".join([tokens[j]] + entity_chars)
                display += f"【{entity_type}:{entity_chars_flat}】"
                j = k + 1 if k > j else j + 1
            else:
                display += tokens[j]
                j += 1

        # 截断过长句子
        if len(display) > 150:
            display = display[:150] + "..."

        print(f"\n[{i+1}] {display}")

    # 7. 交互式审阅（如果启用）
    if args.interactive:
        reviewed = interactive_review(annotated)
        # 覆盖保存
        with open(output_path, "w", encoding="utf-8") as f:
            for ann in reviewed:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")
        print(f"\n✅ 已保存 {len(reviewed)} 句（经人工审阅）")

    print(f"\n📄 训练数据: {output_path}")
    print(f"📄 统计报告: {stats_path}")
    print()
    print("下一步:")
    print("  1. 审阅 ner_train.jsonl 中标注质量（随机抽样检查）")
    print("  2. 对错误标注手动修正")
    print("  3. 积累 500+ 标注句后运行 ner/train.py 训练模型")
    print("  4. 可选: 运行 --interactive 逐句交互式修正")


if __name__ == "__main__":
    main()
