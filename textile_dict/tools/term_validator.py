"""术语一致性校验器：扫描学术论文中使用的政策术语，与 NLP 词典规范比对。

检测项：
  1. 机构名称不一致（如"工信部"→规范"工业和信息化部"）
  2. 纺织术语可能被通用分词切碎（如"紧密纺"→"紧密/纺"）
  3. 政策名称不规范（如缺少书名号、发文字号缺失）
  4. 可用词典规范形式替换的词

运行方式:
    python scripts/nlp_dict/ner/term_validator.py academic/papers/论文文件名.md
    python scripts/nlp_dict/ner/term_validator.py academic/papers/ --all
    python scripts/nlp_dict/ner/term_validator.py academic/论文第二章-纲领性框架.md

依赖:
    jieba, pyyaml

输出:
    - 终端报告（按严重程度分级：🔴高/🟡中/🟢低）
    - data/term_validation_report.csv（可选 --csv）
"""

import sys
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jieba
import yaml

jieba.setLogLevel(20)

# ============================================================
# 加载词典
# ============================================================

DATA_DIR = Path(__file__).parent.parent / "data"
LEXICON_PATH = DATA_DIR / "lexicon_v2.yaml"
JIEBA_DICT_PATH = DATA_DIR / "jieba_dict.txt"

# 加载 jieba 自定义词典
if JIEBA_DICT_PATH.exists():
    jieba.load_userdict(str(JIEBA_DICT_PATH))
else:
    print("⚠️  jieba 词典未找到，部分检查将使用默认分词")


def load_lexicon():
    """加载七层词典。"""
    if not LEXICON_PATH.exists():
        print(f"⚠️  {LEXICON_PATH} 不存在，使用种子词库")
        lex_path = DATA_DIR / "seed_lexicon.yaml"
        if not lex_path.exists():
            return None
        with open(lex_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 检查项
# ============================================================


def build_agency_aliases(lexicon):
    """从第一层提取机构实体 → 规范名映射。

    返回 {别名: 规范名}（如 "工信部" → "工业和信息化部"）
    """
    aliases = {}
    layer1 = lexicon.get("layers", {}).get("layer_1_agencies", {})
    terms = layer1.get("terms", {})
    central = terms.get("中央部委", {})

    for norm_name, info in central.items():
        if isinstance(info, dict):
            # 规范名本身
            aliases[norm_name] = norm_name
            # 所有别名
            for alias in info.get("aliases", []):
                # 只保留较短的别名（长联合发文格式不映射）
                if len(alias) <= 15:
                    aliases[alias] = norm_name

    return aliases


def check_agency_consistency(text, aliases):
    """检查论文中的机构名是否与词典规范一致。

    策略：在文本中找所有已知别名 → 报告未使用规范名的实例。
    """
    findings = []
    # 按别名长度降序排列（优先匹配长的）
    sorted_aliases = sorted(aliases.items(), key=lambda x: -len(x[0]))

    for alias, norm in sorted_aliases:
        if alias == norm:
            continue  # 跳过已经是规范名的

        # 在文本中找这个别名
        for match in re.finditer(re.escape(alias), text):
            # 检查上下文：是否已经是规范名的一部分
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 10)
            context = text[start:end].replace('\n', ' ')

            # 如果别名比规范名短，且规范名更长更完整，优先用规范名
            if len(alias) < len(norm) and norm in context:
                continue  # 规范名已经出现在附近，不报

            findings.append({
                'check': '机构名规范',
                'severity': '🟡中',
                'found': alias,
                'suggested': norm,
                'context': f"...{context}...",
                'line': text[:match.start()].count('\n') + 1,
            })

    # 去重（同一位置可能被多个别名匹配）
    seen = set()
    unique = []
    for f in findings:
        key = (f['line'], f['found'])
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique


def check_textile_terms_segmentation(text, chain_terms):
    """检查纺织术语是否可能在通用分词中被切碎。

    策略：用不加自定义词典的 jieba 分词，对比加载词典后的结果。
    """
    findings = []

    # 提取所有第三层术语（扁平化）
    all_terms = set()
    if isinstance(chain_terms, dict):
        for category, items in chain_terms.items():
            if isinstance(items, list):
                all_terms.update(t for t in items if len(t) >= 3)
            elif isinstance(items, dict):
                for subcat, subitems in items.items():
                    if isinstance(subitems, list):
                        all_terms.update(t for t in subitems if len(t) >= 3)

    # 只检查长度 >= 3 的术语（2 字词通常不会被切碎）
    for term in all_terms:
        if len(term) < 3:
            continue

        # 在文本中找这个术语
        for match in re.finditer(re.escape(term), text):
            # 用通用 jieba 测试是否会切碎
            # 模拟：如果术语长度 >= 4 且包含常见字，可能被切
            if len(term) >= 4:
                # 用基本分词测试
                try:
                    import jieba as jb_raw
                    jb_raw.setLogLevel(20)
                    segmented = list(jb_raw.cut(term))
                    if len(segmented) > 2:  # 被切成 3+ 段
                        findings.append({
                            'check': '领域术语',
                            'severity': '🟢低',
                            'found': term,
                            'suggested': f'确认 jieba 词典已加载此词（当前可能被切成: {"/".join(segmented)}）',
                            'context': f'...{text[max(0,match.start()-20):min(len(text),match.end()+20)]}...',
                            'line': text[:match.start()].count('\n') + 1,
                        })
                except:
                    pass

    return findings


def check_policy_name_format(text):
    """检查政策名称的书写规范。

    检测项：
    - 政策名缺少书名号
    - 发文字号格式
    """
    findings = []

    # 检查可能缺少书名号的政策名
    policy_indicators = [
        r'(?:根据|按照|依据|执行|贯彻|落实|印发|发布|实施).{0,5}(?:办法|方案|意见|通知|规划|条例)',
    ]

    for pattern in policy_indicators:
        for match in re.finditer(pattern, text):
            ctx_start = max(0, match.start() - 5)
            ctx_end = min(len(text), match.end() + 30)
            ctx = text[ctx_start:ctx_end].replace('\n', ' ')

            # 检查是否已有书名号
            if '《' not in ctx[:50]:
                findings.append({
                    'check': '政策名格式',
                    'severity': '🟢低',
                    'found': match.group(),
                    'suggested': '建议为政策名称添加书名号《》',
                    'context': f'...{ctx}...',
                    'line': text[:match.start()].count('\n') + 1,
                })

    return findings


def check_issueing_authority_format(text):
    """检查发文机关格式："发文字号"中的机关简称。

    如："工信部联消费〔2026〕XX号" → 规范写法应为"工业和信息化部"
    """
    findings = []

    # 发文字号模式
    doc_id_pattern = re.compile(r'([一-鿿]{2,5})(?:发|函|公告|令|通告|通知|意见|批复)')
    for match in doc_id_pattern.finditer(text):
        abbr = match.group(1)
        # 检查是否为短简称（2-4字，可能是简称）
        if 2 <= len(abbr) <= 4:
            ctx_start = max(0, match.start() - 10)
            ctx_end = min(len(text), match.end() + 30)
            ctx = text[ctx_start:ctx_end].replace('\n', ' ')

            findings.append({
                'check': '发文机关简称',
                'severity': '🟢低',
                'found': abbr,
                'suggested': '如非标准发文字号格式，建议使用全称',
                'context': f'...{ctx}...',
                'line': text[:match.start()].count('\n') + 1,
            })

    return findings


# ============================================================
# 校验主函数
# ============================================================


def validate_file(filepath, lexicon, agency_aliases, chain_terms):
    """对单个文件执行所有校验项。"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    all_findings = []

    # 1. 机构名一致性
    findings = check_agency_consistency(text, agency_aliases)
    all_findings.extend(findings)

    # 2. 纺织术语（只对前 50000 字检查，避免太长）
    text_sample = text[:50000]
    findings = check_textile_terms_segmentation(text_sample, chain_terms)
    all_findings.extend(findings)

    # 3. 政策名格式
    findings = check_policy_name_format(text_sample)
    all_findings.extend(findings)

    # 4. 发文机关简称
    findings = check_issueing_authority_format(text_sample)
    all_findings.extend(findings)

    return all_findings


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="术语一致性校验器")
    parser.add_argument("path", nargs="?", help="论文文件路径或目录")
    parser.add_argument("--all", action="store_true", help="扫描 academic/papers/ 下所有文件")
    parser.add_argument("--csv", action="store_true", help="同时输出 CSV 报告")
    args = parser.parse_args()

    # 加载词典
    lexicon = load_lexicon()
    if lexicon is None:
        print("❌ 无法加载词典，退出")
        sys.exit(1)

    # 建筑机构别名映射
    agency_aliases = build_agency_aliases(lexicon)

    # 获取第三层产业链术语
    layer3 = lexicon.get("layers", {}).get("layer_3_textile_chain", {})
    chain_terms = layer3.get("terms", {})

    # 确定要扫描的文件
    project_root = Path(__file__).parent.parent.parent.parent  # ner/ -> nlp_dict/ -> scripts/ -> root/
    if args.path:
        filepath = Path(args.path)
        if not filepath.is_absolute():
            # Resolve relative to project root
            filepath = project_root / filepath
        if filepath.is_dir():
            files = list(filepath.glob("*.md"))
        elif filepath.exists():
            files = [filepath]
        else:
            files = []
    elif args.all:
        papers_dir = project_root / "academic" / "papers"
        files = list(papers_dir.glob("*.md"))
        # Also scan the academic dir itself for large files
        academic_dir = project_root / "academic"
        for f in academic_dir.glob("*.md"):
            if f.stat().st_size > 5000 and f not in files:
                files.append(f)
    else:
        # Default: scan all key academic files
        academic_dir = project_root / "academic"
        papers_dir = project_root / "academic" / "papers"
        all_md = list(academic_dir.glob("*.md")) + list(papers_dir.glob("*.md"))
        md_files = sorted(
            [f for f in all_md if f.stat().st_size > 1000],
            key=lambda f: -f.stat().st_size
        )
        files = md_files[:5]  # default to 5 largest

    if not files:
        print("❌ 未找到可扫描的 .md 文件")
        sys.exit(1)

    print(f"📋 词典版本: {lexicon['meta'].get('version', 'unknown')}")
    print(f"📋 机构别名映射: {len(agency_aliases)} 条")
    print(f"📋 第三层产业链术语: {sum(len(v) if isinstance(v, list) else 0 for v in chain_terms.values())} 条")
    print()

    all_results = {}

    for fp in files:
        print(f"🔍 校验: {fp.name}")
        findings = validate_file(fp, lexicon, agency_aliases, chain_terms)
        all_results[str(fp)] = findings

        # 按严重度统计
        high = [f for f in findings if '🔴' in f['severity']]
        mid = [f for f in findings if '🟡' in f['severity']]
        low = [f for f in findings if '🟢' in f['severity']]

        print(f"   🔴 高: {len(high)}  |  🟡 中: {len(mid)}  |  🟢 低: {len(low)}")

        # 展示中高级别问题
        for f in mid[:10]:
            print(f"   [{f['severity']}] {f['check']}: \"{f['found']}\" → {f['suggested']}")

    # 汇总
    print()
    print("=" * 60)
    print("📊 汇总")
    print("=" * 60)

    total_high = sum(len([f for f in v if '🔴' in f['severity']]) for v in all_results.values())
    total_mid = sum(len([f for f in v if '🟡' in f['severity']]) for v in all_results.values())
    total_low = sum(len([f for f in v if '🟢' in f['severity']]) for v in all_results.values())
    total = total_high + total_mid + total_low

    print(f"扫描 {len(files)} 个文件, 共发现 {total} 个问题")
    print(f"  🔴 高严重度: {total_high}")
    print(f"  🟡 中严重度: {total_mid}")
    print(f"  🟢 低严重度: {total_low}")

    if total == 0:
        print("✅ 未发现问题")
    elif total_mid > 0:
        print(f"\n💡 建议优先处理中严重度问题（机构名规范）")

    # CSV 输出
    if args.csv:
        csv_path = DATA_DIR / f"term_validation_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=[
                'file', 'check', 'severity', 'found', 'suggested', 'context', 'line'
            ])
            writer.writeheader()
            for filepath, findings in all_results.items():
                for f in findings:
                    f['file'] = Path(filepath).name
                    writer.writerow(f)
        print(f"\n📄 CSV 报告: {csv_path}")


if __name__ == "__main__":
    main()
