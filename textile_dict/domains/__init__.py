"""领域子集 — 按需加载词典的不同部分。

每个子集函数返回该领域下的所有术语列表。
不同项目可以按需 import 所需的领域子集。

用法:
    from textile_dict.domains.policy import terms
    from textile_dict.domains.industry_chain import categories
"""

from pathlib import Path
from ..core.lexicon import Lexicon

_lex = Lexicon()


def _get_terms(layer_key: str, exclude_categories: list[str] = None) -> list[str]:
    """内部工具：从指定层提取所有术语。"""
    layer = _lex.get_layer(layer_key)
    terms = []
    if isinstance(layer, dict):
        for cat, items in layer.items():
            if exclude_categories and cat in exclude_categories:
                continue
            if isinstance(items, list):
                terms.extend(t for t in items if isinstance(t, str))
            elif isinstance(items, dict):
                for sub_items in items.values():
                    if isinstance(sub_items, list):
                        terms.extend(t for t in sub_items if isinstance(t, str))
    return sorted(set(terms))


# ============================================================
# 政策领域: Layer 1 (机构) + Layer 2 (文书) + Layer 4 (语义)
# ============================================================

def policy_agencies() -> list[str]:
    """发文机关实体 — 中央部委 + 省级厅局。"""
    return _get_terms("layer_1_agencies")


def policy_document_types() -> list[str]:
    """政策文书类型 — 解读/新闻/视频/行政文书。"""
    return _get_terms("layer_2_document_types")


def policy_semantics() -> list[str]:
    """政策语义标签 — 约束/排除/激励方向。"""
    return _get_terms("layer_4_policy_semantics")


def policy_all() -> list[str]:
    """政策领域全部术语。"""
    return (
        policy_agencies()
        + policy_document_types()
        + policy_semantics()
    )


# ============================================================
# 产业链领域: Layer 3 (纺织产业链八段式)
# ============================================================

def industry_chain() -> list[str]:
    """纺织产业链实体术语 — 全部八段。"""
    return _get_terms("layer_3_textile_chain")


def industry_chain_categories() -> dict:
    """按分类组织的产业链术语。"""
    layer = _lex.get_layer("layer_3_textile_chain")
    result = {}
    for cat, items in layer.items():
        if isinstance(items, list):
            result[cat] = [t for t in items if isinstance(t, str)]
        elif isinstance(items, dict):
            for subcat, sub_items in items.items():
                key = f"{cat}/{subcat}"
                if isinstance(sub_items, list):
                    result[key] = [t for t in sub_items if isinstance(t, str)]
    return result


# ============================================================
# 绿色合规领域: Layer 5 中的绿色/环保子集
# ============================================================

def green_compliance() -> list[str]:
    """绿色合规术语 — 碳排放/节能/废水处理/绿色认证等。"""
    layer = _lex.get_layer("layer_5_cross_domain")
    terms = []
    for cat in ["绿色合规", "PMI新词_绿色合规"]:
        if cat in layer:
            items = layer[cat]
            if isinstance(items, list):
                terms.extend(t for t in items if isinstance(t, str))
    return sorted(set(terms))


# ============================================================
# 贸易领域: Layer 5 贸易经济子集
# ============================================================

def trade() -> list[str]:
    """贸易/外贸术语 — 关税/跨境电商/出口退税等。"""
    layer = _lex.get_layer("layer_5_cross_domain")
    terms = []
    for cat in ["贸易经济", "PMI新词_贸易经济"]:
        if cat in layer:
            items = layer[cat]
            if isinstance(items, list):
                terms.extend(t for t in items if isinstance(t, str))
    return sorted(set(terms))


# ============================================================
# 地理/时间
# ============================================================

def geography() -> list[str]:
    """地理与产业集群术语。"""
    return _get_terms("layer_6_geography")


def time_expressions() -> list[str]:
    """时间表达术语。"""
    return _get_terms("layer_7_time_expressions")
