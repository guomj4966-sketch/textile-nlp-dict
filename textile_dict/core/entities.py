"""核心实体类型定义 (dataclass)。

渐进式数据模型的 Python 表示，对应 lexicon_v2.yaml 的 L1-L4 字段。
所有字段均可选 (Optional)，支持渐进增强。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Term:
    """词典中的一条术语条目。

    对应 YAML 中的 term + layer + category 基线字段，
    以及可选的 definition / variants / context_examples 增强字段。
    """

    term: str
    layer: int
    category: str
    subcategory: Optional[str] = None
    pos: Optional[str] = None  # noun / verb / adj / nz / vn
    frequency: int = 0

    # === L2: 术语库级 ===
    definition: Optional[str] = None
    definition_source: Optional[str] = None
    variants: list["Term"] = field(default_factory=list)
    related_terms: list["Relation"] = field(default_factory=list)
    context_examples: list[dict] = field(default_factory=list)

    # === L3: 标注语料级 ===
    ner_tag: Optional[str] = None  # B-TEXTILE / I-AGENCY etc.
    annotated_occurrences: int = 0

    # === L4: 关系/事件级 ===
    relations: list["Relation"] = field(default_factory=list)

    # === 元数据 ===
    maturity: str = "lexicon"  # lexicon | termbase | annotated | linked
    created: Optional[str] = None
    updated: Optional[str] = None
    source_files: list[str] = field(default_factory=list)


@dataclass
class Entity:
    """NER 标注中的实体引用。"""

    text: str
    entity_type: str  # TEXTILE | AGENCY | POLICY | TIME | DOC_TYPE | GEO
    start: int = 0
    end: int = 0
    term_id: Optional[str] = None  # 链接到 Term


@dataclass
class Relation:
    """术语间的关系。"""

    type: str  # similar | alternative_to | prerequisite | produces | used_in
    target: str  # term_id or term string
    context: Optional[str] = None
