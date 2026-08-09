"""从 collector/filter_rules.py 提取种子词库 → YAML。

这是 NLP 词典构建的第一步。filter_rules.py 是项目中最集中、最经过验证
的术语资产，包含 200+ 个领域关键词，100% 准确（均为领域专家标注）。

运行方式:
    python scripts/nlp_dict/build_seed_lexicon.py

输出:
    - scripts/nlp_dict/data/seed_lexicon.yaml  （结构化七层种子词库）
"""

import re
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 动态导入 filter_rules 模块，获取所有常量
import importlib.util

_filter_path = (
    Path(__file__).parent.parent.parent / "collector" / "filter_rules.py"
)
spec = importlib.util.spec_from_file_location("filter_rules", str(_filter_path))
filter_rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_rules)

# ============================================================
# 映射关系：filter_rules.py 常量 → 七层词典结构
# ============================================================

seed = {
    "meta": {
        "version": "0.1.0",
        "generated": filter_rules.get_filter_stats().get("version", "unknown"),
        "source": "collector/filter_rules.py",
        "description": "纺织行业政策 NLP 词典种子词库——从 filter_rules.py 规则引擎提取",
        "total_terms": 0,  # 由 build 函数填充
    },
    "layers": {
        # 第二层：政策文书类型（L1 过滤规则中隐含了文书类型关键词）
        "layer_2_document_types": {
            "description": "政策文书类型与解读形式",
            "source": "TITLE_BLACKLIST 中各规则类名推断",
            "terms": {
                "解读发布": [
                    "新闻发布会", "新闻发布", "答记者问", "国新办吹风会",
                    "国务院政策例行吹风会", "国新办举行", "国新办就",
                ],
                "新闻评论": [
                    "新华时评", "评论员文章", "社论", "人民日报评论",
                    "学习问答", "专家解读文章之",
                ],
                "视频宣传": [
                    "图解", "一图读懂", "动漫", "H5", "视频",
                    "海报", "长图", "速览", "划重点", "快看",
                    "数读", "图说", "一图速览",
                ],
                "行政文书": [
                    "建议的答复", "提案的答复", "人大建议", "政协提案",
                ],
            },
        },
        # 第三层：纺织产业链实体（白名单核心）
        "layer_3_textile_chain": {
            "description": "纺织产业链实体术语",
            "source": "TEXTILE_WHITELIST",
            "terms": {
                "原料": ["棉花", "化纤", "纤维"],
                "制造工艺": [
                    "纺纱", "织造", "非织造", "针织", "印染",
                    "染整", "缫丝", "缝纫",
                ],
                "产品": [
                    "纺织", "服装", "家纺", "面料",
                    "毛纺", "麻纺", "丝绸", "絮用",
                ],
                "设备": ["纺机"],
            },
        },
        # 第四层：政策语义动作（排除/过滤方向的关键词）
        "layer_4_policy_semantics": {
            "description": "政策语义标签——约束、排除、激励方向",
            "source": "TITLE_BLACKLIST + INDUSTRY_EXCLUDE",
            "terms": {
                "约束排除类": [
                    # L1: 各类排除关键词
                    "环保督察整改", "督察", "整改", "销号", "通报整改",
                    "质量抽检", "监督抽查", "不合格",
                    "审批资格", "许可证",
                ],
                "审核评估类": [
                    "课题征集", "课题入选", "研究课题",
                    "教学成果奖", "技能大赛", "作品征集",
                    "招生", "招聘", "引才", "人才招聘",
                    "高新技术企业认定", "工业遗产认定",
                ],
                "纪律司法类": [
                    "党建", "纪检监察", "巡视", "问责",
                    "律师", "公证", "仲裁", "审计",
                    "统计法", "信访", "保密",
                ],
                "涉外军事类": [
                    "领事", "外交", "国防", "军事", "武器",
                    "退役军人", "军人抚恤",
                ],
            },
        },
        # 第五层：交叉领域术语（跨界属性词 + 行业排除词）
        "layer_5_cross_domain": {
            "description": "交叉领域术语——绿色/贸易/科技/其他行业",
            "source": "CROSS_CUTTING_ATTRS + INDUSTRY_EXCLUDE",
            "terms": {
                "绿色合规": filter_rules.CROSS_CUTTING_ATTRS,
                "汽车交通": filter_rules.INDUSTRY_EXCLUDE.get("汽车交通", []),
                "重化工钢铁": filter_rules.INDUSTRY_EXCLUDE.get("重化工钢铁", []),
                "电子信息能源": filter_rules.INDUSTRY_EXCLUDE.get("电子信息能源", []),
                "食品医药": filter_rules.INDUSTRY_EXCLUDE.get("食品医药", []),
                "农渔林牧": filter_rules.INDUSTRY_EXCLUDE.get("农渔林牧", []),
                "金融地产": filter_rules.INDUSTRY_EXCLUDE.get("金融地产", []),
                "文旅体育出版": filter_rules.INDUSTRY_EXCLUDE.get("文旅体育出版", []),
                "公共安全事务": filter_rules.INDUSTRY_EXCLUDE.get("公共安全事务", []),
                "通用政策关键词（非排除）": filter_rules.GENERIC_BUZZWORDS,
            },
        },
        # 第六层：地理与集群（暂缺——filter_rules.py 不含地理信息）
        "layer_6_geography": {
            "description": "地理与产业集群",
            "source": "暂无（filter_rules.py 不含地理关键词）",
            "terms": {},
            "note": "此层需从 policy.db 的 region 字段 + 省级注册表提取，不在种子词库阶段产出",
        },
        # 第七层：时间表达（暂缺——filter_rules.py 不含时间信息）
        "layer_7_time_expressions": {
            "description": "时间表达",
            "source": "暂无（filter_rules.py 不含时间关键词）",
            "terms": {},
            "note": "此层需从 policy.db 的 publish_date 等字段统计 + PMI 新词发现补充，不在种子词库阶段产出",
        },
    },
}

# ============================================================
# 统计与输出
# ============================================================


def count_terms(data):
    """递归统计所有 terms 中的词条数。"""
    total = 0
    for key, value in data.items():
        if key == "terms":
            if isinstance(value, dict):
                for category, words in value.items():
                    if isinstance(words, list):
                        total += len(words)
            elif isinstance(value, list):
                total += len(value)
        elif isinstance(value, dict):
            total += count_terms(value)
    return total


def main():
    total = count_terms(seed)
    seed["meta"]["total_terms"] = total

    output_path = (
        Path(__file__).parent / "data" / "seed_lexicon.yaml"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            seed,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )

    print(f"✅ 种子词库已生成: {output_path}")
    print(f"   共 {total} 个种子术语（分布于七层结构中的 5 层）")

    # 统计每层
    for layer_key, layer_data in seed["layers"].items():
        layer_total = 0
        terms_data = layer_data.get("terms", {})
        if isinstance(terms_data, dict):
            for cat, words in terms_data.items():
                if isinstance(words, list):
                    layer_total += len(words)
        elif isinstance(terms_data, list):
            layer_total = len(terms_data)
        print(f"   {layer_key}: {layer_total} 词  ({layer_data['source']})")

    print()
    print("下一步:")
    print("  1. (手动) 审阅 seed_lexicon.yaml，补充遗漏的机构实体（第一层）")
    print("  2. 运行 build_from_db.py 做 PMI 新词发现，扩充第三至五层")
    print("  3. 运行 merge_lexicon.py 合并种子词库 + PMI候选 + 人工补充")


if __name__ == "__main__":
    main()
