"""月报政策自动板块分类器 —— 基于 NLP 词典第三层产业链术语 + 第四层政策语义。

读取当月新政策，用词典中的术语命中 + 文本相似度，自动建议板块归属。

运行方式:
    python scripts/nlp_dict/ner/auto_classify.py
    python scripts/nlp_dict/ner/auto_classify.py --period 2026-07
    python scripts/nlp_dict/ner/auto_classify.py --period 2026-07 --update-db   # 写入数据库

依赖:
    jieba, pyyaml, sqlite3

输出:
    - 终端报告（每条政策 + 建议板块 + 置信度）
    - 可选：更新 policy.db 的 sectors 字段

数据源:
    - collector/policy.db (policy 表)
    - scripts/nlp_dict/data/lexicon_v2.yaml (第三层 + 第四层术语)
"""

import sys
import re
import math
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

# 六大产业链板块
SIX_SECTORS = {
    "原料端": "棉花、化纤原料相关政策",
    "制造端": "纺织制造、品牌培育、产业政策",
    "出口/商贸端": "跨境电商、出口退税、消费促进",
    "绿色合规端": "排污标准、碳足迹、节能降碳",
    "科技教育人才和文化端": "职业教育、科技创新、非遗",
    "智能制造端": "AI+制造、数字化转型、首台套装备",
}

# 加载 jieba 自定义词典
if JIEBA_DICT_PATH.exists():
    jieba.load_userdict(str(JIEBA_DICT_PATH))


# ============================================================
# 构建板块 → 术语关键词映射
# ============================================================


def build_sector_term_map():
    """从第三层 + 第四层 + 第五层提取每个板块的关键词。

    返回 {sector_name: {keywords: weight}}
    """
    if not LEXICON_PATH.exists():
        print("⚠️  lexicon_v2.yaml 不存在")
        return {}

    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        lexicon = yaml.safe_load(f)

    sector_map = defaultdict(lambda: defaultdict(float))

    # --- 第三层：产业链实体 → 按段映射到板块 ---
    layer3 = lexicon.get("layers", {}).get("layer_3_textile_chain", {}).get("terms", {})

    # 段 → 板块映射
    section_to_sector = {
        "原料端": "原料端",
        "纺纱端": "制造端",
        "织造端": "制造端",
        "染整端": "制造端",
        "成衣/缝制端": "制造端",
        "终端品类": "制造端",
        "智能制造": "智能制造端",
        "绿色低碳循环": "绿色合规端",
    }

    # 第五层的 8 大类行业排除词 → 不用于板块指认（这些是排除其他行业的，不是纺织政策分类依据）
    industry_exclude_categories = {
        "文旅体育出版", "金融地产", "农渔林牧", "食品医药",
        "电子信息能源", "重化工钢铁", "汽车交通", "公共安全事务"
    }

    for section_name, section_data in layer3.items():
        target_sector = None
        for prefix, sector in section_to_sector.items():
            if prefix in section_name:
                target_sector = sector
                break

        if target_sector is None:
            target_sector = "制造端"  # default

        if isinstance(section_data, list):
            for term in section_data:
                sector_map[target_sector][term] = max(sector_map[target_sector].get(term, 0), 1.0)
        elif isinstance(section_data, dict):
            for subcat, terms in section_data.items():
                if isinstance(terms, list):
                    for term in terms:
                        sector_map[target_sector][term] = max(
                            sector_map[target_sector].get(term, 0), 1.0
                        )

    # --- 第四层：政策语义 → 不直接映射板块 ---

    # --- 第五层：交叉领域 → 补充各板块（排除 8 大行业排除词类别）---
    layer5 = lexicon.get("layers", {}).get("layer_5_cross_domain", {}).get("terms", {})

    # 绿色合规端关键词（来自第五层绿色合规类别）
    green_terms = layer5.get("绿色合规", [])
    for term in green_terms:
        # 跳过过于通用的跨领域词（它们出现在各种政策中）
        if term in ("标准", "规范", "数字化", "转型"):
            continue
        sector_map["绿色合规端"][term] = 2.0  # 提高绿色词权重

    # 出口/商贸端关键词
    trade_keywords = [
        "出口退税", "跨境电商", "关税", "以旧换新", "内外贸一体化",
        "消费品以旧换新", "外贸", "贸易", "零售", "消费", "商贸",
        "出口", "进口", "供应链", "品牌",
    ]
    for term in trade_keywords:
        sector_map["出口/商贸端"][term] = 1.0

    # 制造端关键词补充（通用制造/产业词）
    mfg_keywords = [
        "制造业", "产业升级", "产业链", "轻工", "纺织工业",
        "印染", "纺纱", "织造", "服装", "面料", "家纺",
        "纤维", "化纤", "棉花", "棉纺", "纱线",
    ]
    for term in mfg_keywords:
        sector_map["制造端"][term] = 1.0

    # --- 板块专属高区分度关键词（覆盖术语不够精确的问题） ---

    # 绿色合规端独有词（加强区分）
    green_specific = {
        "绿色合规端": [
            ("零碳", 3.0), ("碳排放", 3.0), ("碳达峰", 3.0), ("碳中和", 3.0),
            ("碳市场", 2.5), ("碳足迹", 2.5), ("CCER", 2.5), ("碳交易", 2.5),
            ("排污", 2.5), ("水污染物", 2.5), ("大气污染", 2.5),
            ("VOCs", 2.5), ("温室气体", 2.5), ("清洁生产", 2.5),
            ("绿色工厂", 2.0), ("绿色制造", 2.0), ("零碳工厂", 3.0),
            ("循环经济", 2.0), ("应对气候变化", 2.5),
            ("节能降碳", 2.5), ("能耗", 2.0), ("减排", 2.0),
        ],
        "出口/商贸端": [
            ("跨境电商", 2.5), ("出口退税", 2.5), ("关税", 2.5),
            ("以旧换新", 3.0), ("扩大消费", 3.0), ("零售", 2.5),
            ("消费", 2.0), ("内外贸", 2.5), ("商贸", 2.0),
            ("外贸", 2.0), ("品牌", 1.5),
        ],
        "智能制造端": [
            ("数字化转型", 3.0), ("智能制造", 3.0), ("工业互联网", 3.0),
            ("数字孪生", 3.0), ("AI", 2.5), ("大模型", 2.5),
            ("智能工厂", 3.0), ("机器视觉", 2.5), ("揭榜挂帅", 2.0),
        ],
        "科技教育人才和文化端": [
            ("非物质文化遗产", 3.0), ("非遗", 3.0), ("职业教育", 3.0),
            ("技能培训", 3.0), ("人才培养", 3.0), ("创新", 2.0),
            ("科技", 2.0), ("知识产权", 2.5), ("专利", 2.5),
        ],
        "原料端": [
            ("棉花", 3.0), ("目标价格", 3.0), ("滑准税", 3.0),
            ("储备棉", 3.0), ("配额", 2.5), ("蚕桑", 2.5),
            ("羊毛", 2.0), ("化纤原料", 2.0),
        ],
    }

    for sector, keyword_weights in green_specific.items():
        for kw, weight in keyword_weights:
            sector_map[sector][kw] = max(sector_map[sector].get(kw, 0), weight)

    # 智能制造
    digital_keywords = ["智能制造", "数字化", "数字", "人工智能", "AI", "大数据",
                        "工业互联网", "5G", "物联网", "转型"]
    for term in digital_keywords:
        if term in layer5.get("绿色合规", []):  # "转型" also in green, lower weight
            sector_map["智能制造端"][term] = 0.5
        else:
            sector_map["智能制造端"][term] = 1.0

    # 科技/人才
    tech_keywords = ["科技", "创新", "人才", "教育", "技能", "非遗", "文化",
                     "知识产权", "专利", "研发"]
    for term in tech_keywords:
        sector_map["科技教育人才和文化端"][term] = 1.0

    return sector_map


# ============================================================
# 分类核心逻辑
# ============================================================


def classify_policy(title, full_text, sector_map):
    """对单条政策进行分类。

    策略：
    1. 标题命中 → 高权重
    2. 全文关键词命中 → 加权累计
    3. 返回 Top-2 板块 + 置信度

    返回 [(sector, score, matched_terms), ...]
    """
    # 合并标题 + 全文
    combined = (title or "") + " " + (full_text or "")[:5000]  # 前5000字

    sector_scores = {}
    sector_matches = {}

    for sector, terms_dict in sector_map.items():
        score = 0
        matched = []

        for term, weight in terms_dict.items():
            if len(term) < 2:
                continue

            # 标题命中 → 2x 权重
            title_hits = len(re.findall(re.escape(term), title or ""))
            text_hits = len(re.findall(re.escape(term), combined))

            if title_hits > 0:
                score += weight * 2 * title_hits
                matched.append(f"【标题】{term}")
            elif text_hits > 0:
                score += weight * min(text_hits, 3)  # 最多计3次
                matched.append(term)

        if score > 0:
            sector_scores[sector] = score
            sector_matches[sector] = matched

    # 排序
    ranked = sorted(sector_scores.items(), key=lambda x: -x[1])

    # 归一化置信度 (0-100)
    total = sum(v for _, v in ranked) if ranked else 0
    results = []
    for sector, score in ranked:
        confidence = round(score / total * 100, 1) if total > 0 else 0
        results.append((sector, confidence, sector_matches[sector][:5]))

    return results


# ============================================================
# 主流程
# ============================================================


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="月报政策自动板块分类器")
    parser.add_argument("--period", help="统计期次，如 2026-07")
    parser.add_argument("--update-db", action="store_true", help="写入 policy.db sectors 字段")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 条（0=全部）")
    parser.add_argument("--show-all", action="store_true", help="显示所有政策详情")
    args = parser.parse_args()

    # 构建板块术语映射
    sector_map = build_sector_term_map()

    total_terms = sum(len(v) for v in sector_map.values())
    print(f"📋 板块术语映射: {total_terms} 个关键词 → 6 个板块")
    for sector, terms in sector_map.items():
        print(f"   {sector}: {len(terms)} 个关键词")
    print()

    # 连接数据库
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # 查询待分类政策
    query = """
        SELECT policy_id, title, full_text, publish_date, sectors,
               issuing_authority, textile_relevance
        FROM policy
        WHERE review_status = '通过'
        AND (textile_relevance = '直接' OR textile_relevance = '间接')
        AND doc_type = '政策原文'
    """
    params = []
    if args.period:
        period_start = f"{args.period}-01"
        # Calculate end of period
        parts = args.period.split("-")
        y, m = int(parts[0]), int(parts[1])
        if m == 12:
            period_end = f"{y+1}-01-01"
        else:
            period_end = f"{y}-{m+1:02d}-01"
        query += " AND publish_date >= ? AND publish_date < ?"
        params = [period_start, period_end]

    query += " ORDER BY publish_date DESC"
    if args.limit > 0:
        query += f" LIMIT {args.limit}"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    print(f"📖 处理 {len(rows)} 条政策")
    print()

    # 分类每条政策
    results = []
    sector_counts = Counter()
    changes = 0

    for row in rows:
        pid = row["policy_id"]
        title = row["title"] or ""
        full_text = row["full_text"] or ""
        existing_sectors = row["sectors"] or ""
        authority = row["issuing_authority"] or ""
        relevance = row["textile_relevance"] or ""

        classified = classify_policy(title, full_text, sector_map)

        # Top-1 建议
        top_sector = classified[0][0] if classified else "制造端"
        top_confidence = classified[0][1] if classified else 0

        # 检查是否与现有分类不同
        if existing_sectors and top_sector not in existing_sectors:
            changes += 1

        sector_counts[top_sector] += 1

        # 存结果（用于可能的 db update）
        results.append((pid, top_sector, top_confidence, classified, existing_sectors))

        # 详情输出
        if args.show_all or top_confidence >= 80:
            print(f"[{pid}] {title[:80]}")
            print(f"   发布: {row['publish_date']} | {authority[:30]}")
            print(f"   现有板块: {existing_sectors or '(未分类)'}  →  建议: {top_sector} (置信度 {top_confidence}%)")
            if len(classified) >= 2:
                print(f"   次选: {classified[1][0]} ({classified[1][1]}%)")
            if top_confidence >= 60:
                print(f"   命中词: {', '.join(classified[0][2][:5])}")
            print()

    # 汇总
    print("=" * 60)
    print("📊 板块分布")
    print("=" * 60)
    for sector, desc in SIX_SECTORS.items():
        count = sector_counts.get(sector, 0)
        bar = "█" * (count // 2) if count > 0 else ""
        print(f"  {sector:<16} {count:>3} 条 {bar}")
    print(f"\n  现有分类与建议不一致: {changes} 条")

    # 如果指定了 --update-db 且有变化
    if args.update_db and changes > 0:
        print(f"\n🔧 将更新 {changes} 条政策的板块分类...")
        # 需要读写连接
        conn_rw = sqlite3.connect(str(DB_PATH))
        for pid, top_sector, confidence, _, existing in results:
            if confidence >= 60 and top_sector not in (existing or ""):
                # 追加到现有 sectors（用逗号分隔）
                new_sectors = f"{existing},{top_sector}" if existing else top_sector
                # 去重
                parts = [s.strip() for s in new_sectors.split(",") if s.strip()]
                new_sectors = ",".join(sorted(set(parts)))
                conn_rw.execute(
                    "UPDATE policy SET sectors = ? WHERE policy_id = ?",
                    (new_sectors, pid)
                )
        conn_rw.commit()
        conn_rw.close()
        print("✅ 数据库已更新")
    elif args.update_db:
        print("\n✅ 无需更新（所有分类一致）")


if __name__ == "__main__":
    main()
