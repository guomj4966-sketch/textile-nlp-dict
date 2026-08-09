"""共享提取工具 — 各 DataSourceAdapter 的公共实现。

提供基于 jieba 的通用术语发现逻辑，以及启发式纺织术语分类函数。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Optional

import jieba
import yaml

# 加载已有词典（如果可用）
_DICT_PATH = Path(__file__).parent.parent / "data" / "jieba_dict.txt"
if _DICT_PATH.exists():
    jieba.load_userdict(str(_DICT_PATH))


# ═══════════════════════════════════════════════════════════════
# 停用词与过滤
# ═══════════════════════════════════════════════════════════════

STOP_WORDS: set[str] = {
    "根据", "按照", "关于", "有关", "相关", "应当", "可以", "予以",
    "负责", "确保", "组织", "协调", "建立", "制定", "实施", "执行",
    "落实", "推进", "开展", "加强", "完善", "进一步", "持续", "不断",
    "着力", "加大", "提高", "降低", "减少", "增加", "扩大",
    "目前", "已经", "正在", "计划", "准备", "预计", "预计到",
    "通过", "采用", "利用", "应用", "使用", "包括", "包含",
    "主要", "其中", "其中以", "同时", "此外", "另外",
    "以上", "以下", "上述", "下列", "如下",
    "为", "的是", "以及", "及其", "并按", "根据各",
    "发展", "建设", "提升", "重点", "领域", "工作", "支持",
    "服务", "产业", "开展", "国家", "企业", "优化",
    "市场", "创新", "实现", "形成", "推动",
    "这些", "它们", "所有", "整个", "全部", "部分",
    "第一", "第二", "第三", "事项", "内容", "情况",
    "产品", "技术", "行业", "项目", "单位", "部门",
    "该产品", "该项目", "采用", "生产", "加工",
    "等方面", "等领域", "等行业", "的研发", "的研制",
    "推进落实", "持续优化", "共同推进",
}

GENERIC_SUFFIXES: tuple[str, ...] = (
    "能力", "水平", "效率", "质量", "效益", "效果",
    "程度", "速度", "力度", "强度", "广度", "深度",
    "问题", "需求", "目标", "任务", "措施", "方法",
    "方案", "规划", "政策", "法规", "标准", "规范",
    "工作", "业务", "流程", "过程", "环节", "步骤",
    "设备", "装置", "系统", "平台", "工具", "软件",
    "材料", "原料", "资源", "能源", "资金", "资产",
)


def _is_generic(term: str) -> bool:
    """判断一个词是否过于通用，不应作为领域术语。"""
    return term.endswith(GENERIC_SUFFIXES)


def _extract_chinese_parts(text: str, min_len: int = 2, max_len: int = 8) -> list[str]:
    """从文本中提取连续中文片段。"""
    return re.findall(rf"[一-鿿]{{{min_len},{max_len}}}", text)


# ═══════════════════════════════════════════════════════════════
# 现有词典加载（用于去重）
# ═══════════════════════════════════════════════════════════════

_EXISTING_TERMS_CACHE: Optional[set[str]] = None


def load_existing_terms() -> set[str]:
    """加载 lexicon_v2.yaml 中所有已有术语（带缓存）。"""
    global _EXISTING_TERMS_CACHE
    if _EXISTING_TERMS_CACHE is not None:
        return _EXISTING_TERMS_CACHE

    lex_path = Path(__file__).parent.parent / "data" / "lexicon_v2.yaml"
    _EXISTING_TERMS_CACHE = set()
    if not lex_path.exists():
        return _EXISTING_TERMS_CACHE

    with open(lex_path, encoding="utf-8") as f:
        lex = yaml.safe_load(f)

    META_KEYS = {
        "description", "source", "note", "产业定位", "政策类型",
        "policy_count", "cluster_info", "type", "aliases", "definition",
    }

    def collect(obj):
        if isinstance(obj, str) and len(obj) >= 2:
            _EXISTING_TERMS_CACHE.add(obj)
        elif isinstance(obj, list):
            for item in obj:
                collect(item)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k not in META_KEYS:
                    if len(k) >= 2:
                        _EXISTING_TERMS_CACHE.add(k)
                    collect(v)

    for layer_data in lex.get("layers", {}).values():
        collect(layer_data.get("terms", {}))
    return _EXISTING_TERMS_CACHE


def clear_existing_terms_cache():
    """清除缓存（用于词典更新后刷新）。"""
    global _EXISTING_TERMS_CACHE
    _EXISTING_TERMS_CACHE = None


# ═══════════════════════════════════════════════════════════════
# 通用术语发现
# ═══════════════════════════════════════════════════════════════


def discover_terms_from_texts(
    texts: list[str],
    *,
    min_len: int = 2,
    min_freq: int = 2,
    max_terms: int = 500,
    skip_existing: bool = True,
) -> Counter:
    """从文本集中发现高频中文术语（不在已有词典中）。

    适用于所有 DataSourceAdapter 的 extract_terms() 实现。

    Args:
        texts: 文本字符串列表。
        min_len: 词条最短长度（字符）。
        min_freq: 最低频次阈值。
        max_terms: 最多返回候选数。
        skip_existing: 是否跳过已在 lexicon_v2.yaml 中的词。

    Returns:
        Counter{term: frequency}，按频次降序。
    """
    existing = load_existing_terms() if skip_existing else set()
    term_counter: Counter = Counter()

    for text in texts:
        words = jieba.lcut(text)
        chinese_words = [
            w.strip()
            for w in words
            if len(w.strip()) >= min_len
            and re.match(r"^[一-鿿a-zA-Z0-9/+]+$", w.strip())
            and w.strip() not in existing
        ]
        term_counter.update(chinese_words)

    # 过滤
    filtered: Counter = Counter()
    for term, freq in term_counter.most_common():
        if freq < min_freq:
            break
        if len(filtered) >= max_terms:
            break

        # 排除纯数字
        if re.match(r"^[\d\.\+\-/]+$", term):
            continue
        # 排除单字
        if len(term) < min_len:
            continue
        # 排除停用词
        if term in STOP_WORDS:
            continue
        # 排除通用后缀词
        if _is_generic(term):
            continue

        filtered[term] = freq

    return filtered


# ═══════════════════════════════════════════════════════════════
# 纺织品相关性判定（用于过滤非纺织品候选）
# ═══════════════════════════════════════════════════════════════

# 纺织核心字（至少包含一个才视为相关）
TEXTILE_CORE_CHARS: set[str] = set(
    "纺织服装棉麻丝毛纤维纱布染印经编缝纫缫浆氨涤锦腈纶维碳芳"
)

# 纺织/工业关键词（放宽匹配）
TEXTILE_BROAD_KEYWORDS: list[str] = [
    "面", "料", "色", "纱", "线", "绳", "带", "絮", "绒",
    "衫", "裤", "裙", "巾", "毯", "帘", "垫", "枕", "被",
    "袖", "领", "扣", "链", "衬",
    # 产业用
    "土工", "过滤", "车用", "医用", "防护", "卫生",
    # 家纺
    "床品", "毛巾", "窗帘", "地毯",
    # 设备
    "纺机", "织机", "染整设备", "针织机", "无纺设备",
]


def is_textile_relevant(term: str) -> bool:
    """判断一个术语是否与纺织领域相关。

    宽松策略：宁可多收（人工校验阶段再筛），不要漏掉。
    """
    # 包含核心汉字
    if any(c in term for c in TEXTILE_CORE_CHARS):
        return True
    # 包含宽泛关键词
    if any(kw in term for kw in TEXTILE_BROAD_KEYWORDS):
        return True
    # 纺织相关英文缩写
    if term.upper() in {
        "GRS", "GOTS", "OCS", "BCI", "OEKO-TEX", "ZDHC",
        "CBAM", "EPD", "PEF", "LCA", "DPP", "SMS", "SMMS",
        "ERP", "MES", "PLM", "SCM", "WMS", "APS",
    }:
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# 启发式分类（candidate → layer + category）
# ═══════════════════════════════════════════════════════════════


def classify_candidate(term: str) -> tuple[str, str]:
    """启发式分类：将候选术语分配到最可能的 layer + category。

    Returns:
        (layer_key, category) — 如 ("layer_3_textile_chain", "1_原料端 / 天然纤维")
    """

    # ─── Layer 1: 发文机关 ───
    agency_patterns = [
        ("国务院", "layer_1_agencies", "中央部委"),
        ("省政府", "layer_1_agencies", "省级厅局"),
        ("发改委", "layer_1_agencies", "中央部委"),
        ("工信部", "layer_1_agencies", "中央部委"),
        ("商务部", "layer_1_agencies", "中央部委"),
        ("市场监管", "layer_1_agencies", "中央部委"),
        ("自治区", "layer_1_agencies", "省级厅局"),
    ]
    for pat, lk, cat in agency_patterns:
        if pat in term:
            return (lk, cat)

    # ─── Layer 3: 纺织产业链（八段）───

    # 1. 原料
    raw_keywords = [
        "纤维", "丝", "棉", "麻", "毛", "绒", "茧", "蚕",
        "涤纶", "锦纶", "氨纶", "腈纶", "丙纶", "维纶", "氯纶",
        "莱赛尔", "莫代尔", "粘胶", "醋酸", "铜氨",
        "碳纤维", "芳纶", "玄武岩",
    ]
    if any(kw in term for kw in raw_keywords):
        if any(kw in term for kw in ["碳纤维", "芳纶", "玄武岩", "超高分子量"]):
            return ("layer_3_textile_chain", "1_原料端 / 高性能纤维")
        if any(kw in term for kw in ["涤纶", "锦纶", "氨纶", "腈纶", "丙纶"]):
            return ("layer_3_textile_chain", "1_原料端 / 合成纤维")
        if any(kw in term for kw in ["莱赛尔", "莫代尔", "粘胶", "醋酸", "铜氨", "天丝"]):
            return ("layer_3_textile_chain", "1_原料端 / 再生纤维素纤维")
        return ("layer_3_textile_chain", "1_原料端 / 天然纤维")

    # 2. 纺纱
    spinning_kw = [
        "纺纱", "环锭", "紧密纺", "转杯纺", "喷气纺", "涡流纺", "气流纺",
        "精梳", "粗梳", "络筒", "捻线", "并条", "粗纱", "细纱",
    ]
    if any(kw in term for kw in spinning_kw):
        return ("layer_3_textile_chain", "2_纺纱工艺")

    # 3. 织造
    weaving_kw = [
        "织造", "针织", "梭织", "经编", "纬编", "横机", "圆机",
        "提花", "色织", "面料", "织物", "坯布",
    ]
    if any(kw in term for kw in weaving_kw):
        return ("layer_3_textile_chain", "3_织造端")

    # 4. 染整
    dyeing_kw = [
        "染", "印花", "整理", "染色", "丝光", "预缩", "阻燃整理",
        "防水整理", "防污整理", "数码印花", "热转印",
    ]
    if any(kw in term for kw in dyeing_kw):
        return ("layer_3_textile_chain", "4_染整端")

    # 5. 成衣
    garment_kw = [
        "缝制", "裁剪", "缝纫", "整烫", "模板缝制", "吊挂系统",
    ]
    if any(kw in term for kw in garment_kw):
        return ("layer_3_textile_chain", "5_成衣/缝制端")

    # 6. 终端品类
    product_kw = [
        "服装", "家纺", "纺织", "面料", "纱", "布", "线",
        "纺织品", "产业用", "非织造", "无纺布", "牛仔",
        "衬衫", "裤", "裙", "内衣", "袜", "毛巾",
        "羽绒服", "冲锋衣", "工装", "防护服",
    ]
    if any(kw in term for kw in product_kw):
        return ("layer_3_textile_chain", "6_终端品类")

    # 7. 智能制造
    smart_kw = [
        "智能制造", "数字化", "数字孪生", "机器视觉", "AI质检",
        "工业互联网", "AGV", "智能验布", "智能吊挂",
    ]
    if any(kw in term for kw in smart_kw):
        return ("layer_3_textile_chain", "7_智能制造")

    # 8. 绿色低碳
    green_kw = [
        "绿色", "低碳", "零碳", "循环", "节能", "碳足迹", "碳标签",
        "废水处理", "废气治理", "中水回用",
    ]
    if any(kw in term for kw in green_kw):
        return ("layer_3_textile_chain", "8_绿色低碳循环")

    # ─── Layer 4: 政策语义 ───
    policy_kw = [
        "行动方案", "规划", "意见", "通知", "方案", "措施",
        "试点", "示范", "专项行动", "三年行动",
    ]
    if any(kw in term for kw in policy_kw):
        return ("layer_4_policy_semantics", "政策工具类")

    # ─── Layer 5: 交叉领域 ───
    trade_kw = ["出口", "进口", "关税", "退税", "贸易", "跨境", "海关", "供应链"]
    if any(kw in term for kw in trade_kw):
        return ("layer_5_cross_domain", "贸易经济")

    digital_kw = ["AI", "大模型", "5G", "物联网", "区块链"]
    if any(kw in term for kw in digital_kw):
        return ("layer_5_cross_domain", "数字经济")

    # ─── Layer 6: 地理 ───
    province_set = {
        "浙江", "江苏", "山东", "广东", "福建", "上海", "新疆",
        "河北", "河南", "湖北", "四川", "江西", "安徽",
    }
    if term in province_set or any(
        term.endswith(suf) for suf in ["省", "市", "区", "县", "州"]
    ):
        return ("layer_6_geography", "地理")

    # ─── 兜底 ───
    return ("layer_5_cross_domain", "待人工分类")
