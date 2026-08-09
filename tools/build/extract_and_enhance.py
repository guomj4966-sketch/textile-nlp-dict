"""从 policy.db + 省级注册表 提取机构实体/地理集群/时间表达，补全七层种子词库。

运行方式:
    python scripts/nlp_dict/extract_and_enhance.py

输出:
    - scripts/nlp_dict/data/layer_1_agencies.yaml   ← 第一层：机构实体
    - scripts/nlp_dict/data/layer_6_geography.yaml  ← 第六层：地理与集群
    - scripts/nlp_dict/data/layer_7_time.yaml       ← 第七层：时间表达
    - scripts/nlp_dict/data/seed_lexicon.yaml       ← 更新版（补充三层到七层）
"""

import sqlite3
import re
import sys
from pathlib import Path
from collections import Counter

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================================
# 数据库连接
# ============================================================

DB_PATH = Path(__file__).parent.parent.parent / "collector" / "policy.db"
PROVINCIAL_REGISTRY = (
    Path(__file__).parent.parent.parent / "数据源注册表-省级.md"
)


def get_conn():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# 第一层：机构实体提取
# ============================================================


def extract_agencies(conn):
    """从 policy.db 的 issuing_authority 字段提取发文机关实体。

    处理中文编码显示问题——sqlite3 在 Windows 终端可能乱码，
    但 .fetchall() 返回的是正确的 Unicode 字符串，内部处理正常。
    """
    rows = conn.execute(
        "SELECT issuing_authority, level, region, COUNT(*) as cnt "
        "FROM policy "
        "WHERE issuing_authority IS NOT NULL AND issuing_authority != '' "
        "AND review_status = '通过' "
        "GROUP BY issuing_authority, level, region "
        "ORDER BY cnt DESC"
    ).fetchall()

    # 手动分类聚合
    central = {}  # {规范名: {aliases: [], count: N}}
    provincial = {}  # {省份: {厅局名: count}}

    for r in rows:
        name = r["issuing_authority"].strip()
        level = r["level"] or ""
        region = r["region"] or ""
        cnt = r["cnt"]

        if level == "中央" or region == "中央" or region == "国务院":
            # 归并到中央机构
            normalized = _normalize_agency(name)
            if normalized not in central:
                central[normalized] = {"aliases": set(), "count": 0}
            central[normalized]["aliases"].add(name)
            central[normalized]["count"] += cnt
        else:
            # 省级机构
            prov = region if region and region != "中央" else "未知省份"
            if prov not in provincial:
                provincial[prov] = {}
            dept = _normalize_provincial_dept(name)
            if dept not in provincial[prov]:
                provincial[prov][dept] = 0
            provincial[prov][dept] += cnt

    # 按频次排序
    central_sorted = sorted(
        central.items(), key=lambda x: -x[1]["count"]
    )
    provincial_sorted = sorted(
        provincial.items(), key=lambda x: -sum(x[1].values())
    )

    return {
        "description": "发文机关实体——中央部委全称/简称 + 省级厅局",
        "source": "policy.db: issuing_authority + level + region 字段",
        "central": {
            name: {
                "aliases": sorted(info["aliases"]),
                "policy_count": info["count"],
            }
            for name, info in central_sorted
        },
        "provincial": {
            prov: {
                dept: count
                for dept, count in sorted(depts.items(), key=lambda x: -x[1])
            }
            for prov, depts in provincial_sorted
        },
    }


def _normalize_agency(name):
    """将发文机关全称/简称归一到统一名称。仅做基础合并，详细映射需人工审阅。"""
    # 国务院系统
    if any(kw in name for kw in ["国务院", "国办"]):
        if "中共中央" in name or "中办" in name:
            return "中共中央、国务院（含中办国办联合发文）"
        return "国务院（含办公厅）"
    if "中共中央" in name or name in ("中国共产党中央委员会", "中共中央"):
        if "国务院" not in name:
            return "中共中央"
        return "中共中央、国务院（含中办国办联合发文）"

    # 发改委系统
    if "发展" in name and ("改革" in name or "发改" in name):
        return "国家发展和改革委员会"
    # 工信部
    if "工业" in name and "信息" in name:
        return "工业和信息化部"
    # 商务部
    if "商务" in name and len(name) <= 6:
        return "商务部"
    # 生态环境部
    if "生态" in name and "环境" in name:
        return "生态环境部"
    # 财政部
    if "财政" in name and len(name) <= 6:
        return "财政部"
    # 教育部
    if "教育" in name and len(name) <= 6:
        return "教育部"
    # 人社部
    if "人力" in name or "人社" in name:
        return "人力资源和社会保障部"
    # 科技部
    if "科学" in name and "技术" in name and "部" in name and len(name) <= 6:
        return "科学技术部"
    # 市场监管总局 / 国标委
    if "市场监管" in name or "标准委" in name or "标准化" in name:
        return "国家市场监督管理总局（含国家标准委）"
    # 文旅部
    if "文化" in name and "旅游" in name:
        return "文化和旅游部"
    # 税务总局
    if "税务" in name or "税收" in name:
        return "国家税务总局"
    # 海关总署
    if "海关" in name:
        return "海关总署"
    # 司法部
    if "司法" in name and len(name) <= 6:
        return "司法部"
    # 人社部/民政部 等其他
    if "人力" in name and "社保" in name or "社会保障" in name:
        return "人力资源和社会保障部"

    # 无法归类的，保留原名
    return name


def _normalize_provincial_dept(name):
    """将省级厅局名称归一到规范简称。"""
    # 去掉"江苏省""浙江省"等前缀，保留厅局名
    prov_prefixes = [
        "江苏省", "浙江省", "山东省", "福建省", "广东省",
        "湖北省", "安徽省", "江西省", "上海市", "新疆维吾尔自治区",
        "新疆", "北京市", "天津市", "河北省", "山西省",
        "辽宁省", "吉林省", "黑龙江省",
        "河南省", "湖南省", "海南省",
        "四川省", "贵州省", "云南省",
        "陕西省", "甘肃省", "青海省",
        "广西壮族自治区", "内蒙古自治区", "宁夏回族自治区",
        "西藏自治区", "重庆市",
    ]
    short = name
    for p in sorted(prov_prefixes, key=len, reverse=True):
        if short.startswith(p):
            short = short[len(p):]
            break

    # 统一厅局命名
    replacements = {
        "工业和信息化厅": "工信厅",
        "经济和信息化厅": "经信厅",
        "经济和信息化委员会": "经信委",
        "发展和改革委员会": "发改委",
        "发展改革委": "发改委",
        "发展改革委员会": "发改委",
        "生态环境厅": "生态环境厅",
        "生态环境局": "生态环境局",
        "商务厅": "商务厅",
        "商务委员会": "商务委",
        "财政厅": "财政厅",
        "人民政府办公厅": "省政府办公厅",
        "人民政府": "省政府",
        "市场监督管理局": "市场监管局",
        "人大": "省人大",
        "人大常委会": "省人大",
    }
    for full, abbr in replacements.items():
        short = short.replace(full, abbr)

    return short.strip()


# ============================================================
# 第六层：地理与产业集群提取
# ============================================================


def extract_geography(conn):
    """从 policy.db + 省级注册表提取地理与产业集群信息。"""
    # 从 policy.db 获取省级政策分布
    regions = conn.execute(
        "SELECT region, COUNT(*) as cnt "
        "FROM policy "
        "WHERE region IS NOT NULL AND region != '' AND region != '中央' "
        "AND review_status = '通过' "
        "GROUP BY region "
        "ORDER BY cnt DESC"
    ).fetchall()

    # 从省级注册表提取省份和纺织产业定位
    provincial_clusters = {
        "江苏": {
            "产业定位": "高端纺织万亿产业集群",
            "重点城市": ["苏州", "无锡", "常州", "南通", "盐城"],
            "政策类型": "产业升级/智能制造/印染集聚",
        },
        "浙江": {
            "产业定位": "现代纺织强省、绍兴柯桥国际纺都",
            "重点城市": ["绍兴（柯桥）", "杭州", "嘉兴", "宁波"],
            "政策类型": "数字化/绿色印染/产业集群",
        },
        "山东": {
            "产业定位": "全国棉纺第一大省、工装产业基地",
            "重点城市": ["滨州", "潍坊", "淄博", "青岛"],
            "政策类型": "棉纺升级/工装方案/先进制造业",
        },
        "福建": {
            "产业定位": "万亿纺织鞋服产业集群",
            "重点城市": ["泉州", "福州", "莆田"],
            "政策类型": "鞋服品牌/运动服装/产业用纺织品",
        },
        "广东": {
            "产业定位": "时尚产业生态、纺织服装外贸",
            "重点城市": ["广州", "深圳", "东莞", "普宁"],
            "政策类型": "时尚消费/跨境电商/品牌培育",
        },
        "新疆": {
            "产业定位": "国家优质棉花棉纱基地",
            "重点城市": ["乌鲁木齐", "阿克苏", "喀什", "石河子"],
            "政策类型": "棉花目标价格/棉纺基地/产业援疆",
        },
        "上海": {
            "产业定位": "时尚消费之都、纺织贸易总部",
            "重点城市": ["上海"],
            "政策类型": "时尚产业/国际消费中心/外贸总部",
        },
        "湖北": {
            "产业定位": "中部纺织产业转移承接区",
            "重点城市": ["仙桃", "孝感", "荆州", "襄阳"],
            "政策类型": "产业转移/非织造/服装加工",
        },
        "安徽": {
            "产业定位": "中部纺织新兴承接区",
            "重点城市": ["合肥", "芜湖", "安庆"],
            "政策类型": "产业转移/纺织优化升级方案",
        },
        "江西": {
            "产业定位": "承接东部纺织产业转移基地",
            "重点城市": ["南昌", "赣州"],
            "政策类型": "产业转移/纺织服装/外贸加工",
        },
    }

    region_stats = [
        {
            "name": r["region"],
            "count": r["cnt"],
            "cluster": provincial_clusters.get(
                r["region"].replace("省", "").replace("市", ""),
                None,
            ),
        }
        for r in regions
    ]

    return {
        "description": "地理与产业集群——省级纺织产业布局",
        "source": "policy.db: region 字段 + 数据源注册表-省级.md",
        "region_distribution": {
            r["region"]: {
                "policy_count": r["cnt"],
                "cluster_info": provincial_clusters.get(
                    r["region"].replace("省", "").replace("市", ""),
                    {},
                ),
            }
            for r in regions
        },
        "top10_provinces": [
            r["region"] for r in regions[:10]
        ],
    }


# ============================================================
# 第七层：时间表达提取
# ============================================================


def extract_time_patterns(conn):
    """从 policy.db 和已知政策用语中提取时间表达模式。

    时间表达主要有三种：
    - 绝对时间：具体日期、发文字号中的年份
    - 相对时间：政策文本中常见的期间表达
    - 周期表达：月度/季度/年度等周期性表述
    """
    # 从 publish_date 字段统计日期分布特征
    dates = conn.execute(
        "SELECT publish_date FROM policy "
        "WHERE publish_date IS NOT NULL AND publish_date != '' "
        "AND publish_date >= '2026-01-01'"
    ).fetchall()

    # 从 full_text 中搜索时间模式关键词（抽样 50 篇纺织直接相关）
    samples = conn.execute(
        "SELECT full_text FROM policy "
        "WHERE review_status = '通过' AND textile_relevance = '直接' "
        "AND full_text IS NOT NULL AND full_text != '' "
        "ORDER BY publish_date DESC LIMIT 30"
    ).fetchall()

    # 时间表达正则模式
    absolute_patterns = [
        "YYYY年M月D日起施行",
        "自发布之日起",
        "自发文之日起施行",
        "自印发之日起",
        "自YYYY年M月D日起",
        "施行日期",
        "有效期至",
    ]

    relative_patterns_found = Counter()
    for row in samples:
        text = row["full_text"] or ""
        for pat in [
            r"(\d+)年内",
            r"十四五.{0,5}期间",
            r"十五五.{0,5}期间",
            r"未来(\d+)年",
            r"本规划期",
            r"每年\S{0,5}前",
            r"每[月季度年]",
            r"逐[月日]",
            r"按[月季度年]",
        ]:
            matches = re.findall(pat, text)
            for m in matches:
                relative_patterns_found[m if isinstance(m, str) else pat] += 1

    return {
        "description": "政策文本中的时间表达——绝对时间/相对时间/周期",
        "source": "policy.db: publish_date + full_text 抽样搜索",
        "terms": {
            "绝对时间": [
                "YYYY年M月D日起施行",
                "自发布之日起施行",
                "自发文之日起施行",
                "自印发之日起施行",
                "自发布之日起30日后施行",
                "有效期至YYYY年M月D日",
                "施行日期",
                "发布日期",
            ],
            "相对时间": [
                "三年内",
                "十四五期间",
                "十五五期间",
                "未来五年",
                "本规划期",
                "近期（1-2年）",
                "中长期（3-5年）",
                "五年规划期内",
            ],
            "周期表达": [
                "年度申报",
                "每年X月前",
                "逐月报送",
                "按季度",
                "按月调度",
                "年底前",
                "上半年/下半年",
                "每月",
            ],
            "政策生命周期节点": [
                "征求意见期",
                "公示期",
                "过渡期",
                "实施日期",
                "有效期",
                "申报窗口期",
                "评估验收期",
            ],
        },
        "date_range": {
            "earliest": dates[-1]["publish_date"] if dates else "N/A",
            "latest": dates[0]["publish_date"] if dates else "N/A",
            "total_count": len(dates),
        },
    }


# ============================================================
# 合并到 seed_lexicon.yaml
# ============================================================


def merge_to_seed_lexicon(agencies, geography, time_patterns):
    """将三层的提取结果合并到已有的 seed_lexicon.yaml 中。"""
    seed_path = Path(__file__).parent / "data" / "seed_lexicon.yaml"
    if not seed_path.exists():
        print("⚠️  seed_lexicon.yaml 不存在，跳过合并")
        return

    with open(seed_path, "r", encoding="utf-8") as f:
        seed = yaml.safe_load(f)

    # 更新第一层
    seed["layers"]["layer_1_agencies"] = {
        "description": agencies["description"],
        "source": agencies["source"],
        "terms": {
            "中央部委": {
                name: info
                for name, info in agencies["central"].items()
            },
            "省级厅局": dict(
                sorted(
                    agencies["provincial"].items(),
                    key=lambda x: -sum(x[1].values()),
                )
            ),
        },
    }

    # 更新第六层
    seed["layers"]["layer_6_geography"] = {
        "description": geography["description"],
        "source": geography["source"],
        "terms": geography["region_distribution"],
        "top10_provinces": geography["top10_provinces"],
    }

    # 更新第七层
    seed["layers"]["layer_7_time_expressions"] = time_patterns

    # 重新统计
    total = _count_all_terms(seed)
    seed["meta"]["total_terms"] = total
    seed["meta"]["generated"] = "2026-07-29_v1_extracted"

    with open(seed_path, "w", encoding="utf-8") as f:
        yaml.dump(
            seed,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )

    print(f"✅ 已更新 seed_lexicon.yaml（共 {total} 条目）")


def _count_all_terms(data):
    """递归统计所有叶节点词条数。"""
    total = 0
    for key, value in data.items():
        if key == "terms":
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, list):
                        total += len(sub_val)
                    elif isinstance(sub_val, dict):
                        # 机构实体层的嵌套结构
                        for inner_key, inner_val in sub_val.items():
                            if isinstance(inner_val, dict) and "aliases" in inner_val:
                                total += len(inner_val["aliases"])
                            elif isinstance(inner_val, (int, float)):
                                total += 1
        elif isinstance(value, dict):
            total += _count_all_terms(value)
    return total


# ============================================================
# 保存独立 YAML 文件
# ============================================================


def save_layer_file(filename, data):
    path = Path(__file__).parent / "data" / filename
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"   {filename}")


def main():
    conn = get_conn()

    print("提取第一层：机构实体...")
    agencies = extract_agencies(conn)
    save_layer_file("layer_1_agencies.yaml", agencies)
    central_count = len(agencies["central"])
    prov_count = sum(len(v) for v in agencies["provincial"].values())
    print(f"   中央机构: {central_count}，省级机构: {prov_count}")

    print("提取第六层：地理与产业集群...")
    geography = extract_geography(conn)
    save_layer_file("layer_6_geography.yaml", geography)
    print(f"   省份: {len(geography['region_distribution'])}")

    print("提取第七层：时间表达...")
    time_patterns = extract_time_patterns(conn)
    save_layer_file("layer_7_time.yaml", time_patterns)
    print(f"   时间模式类别: {len(time_patterns['terms'])}")

    conn.close()

    print("\n合并到 seed_lexicon.yaml...")
    merge_to_seed_lexicon(agencies, geography, time_patterns)

    print("\n🎉 七层种子词库已全部就绪。")
    print("下一步:")
    print("  1. (手动) 审阅 layer_1_agencies.yaml 中的机构名称映射")
    print("  2. (手动) 审阅 layer_6_geography.yaml 中的产业定位")
    print("  3. 运行 build_from_db.py 做 PMI 新词发现扩充第三至五层")


if __name__ == "__main__":
    main()
