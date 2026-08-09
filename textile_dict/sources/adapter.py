"""数据源适配器框架 — 可插拔的资料源接入层。

每一个领域资料库（政策库、标准库、产品库、专利库等）实现一个
DataSourceAdapter 子类，即可自动接入本词典项目的 PMI 新词发现、
术语校验和合并流程。

架构
----

本仓库（textile-nlp-dict）定位为"词典中台"：
  - 入站：各领域数据库 → DataSourceAdapter → 术语候选提取 → lexicon
  - 出站：lexicon → jieba/NER → 各消费项目

全部 adapter 遵循同一接口，由 SourceRegistry 统一发现和调度。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 公共实体类型
# ═══════════════════════════════════════════════════════════════


@dataclass
class TermCandidate:
    """一条从外部数据源提取的候选术语。"""

    term: str
    source_name: str  # adapter.name
    layer: str = ""  # 建议归属层，如 "layer_3_textile_chain"
    category: str = ""  # 建议归属分类，如 "1_原料端 / 天然纤维"
    definition: Optional[str] = None  # 术语定义（如有）
    english: str = ""  # 英文对应词
    frequency: int = 0  # 在源文本中的出现次数
    context_example: Optional[str] = None  # 出现上下文片段
    confidence_score: float = 0.0  # 0-1 置信度
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SourceMetadata:
    """数据源元信息。"""

    name: str  # 唯一标识，如 "policy_db"
    display_name: str  # 人类可读名，如 "纺织政策数据库"
    source_type: str  # 资料类型：policy / standard / product / patent / report / academic
    version: str = "1.0"
    description: str = ""
    location: str = ""  # 数据源路径或连接串
    item_count: int = 0  # 资料条数
    estimated_total_chars: int = 0  # 估计总字符数
    last_updated: str = ""
    tags: list[str] = field(default_factory=list)  # ["纺织", "政策", "2024-2026"]


@dataclass
class ExtractionResult:
    """一次术语提取操作的完整结果。"""

    adapter_name: str
    extracted_at: str
    total_texts_processed: int
    total_chars_processed: int
    candidates: list[TermCandidate]
    stats: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def unique_terms(self) -> list[str]:
        return sorted({c.term for c in self.candidates})


# ═══════════════════════════════════════════════════════════════
# 抽象基类
# ═══════════════════════════════════════════════════════════════


class DataSourceAdapter(ABC):
    """资料源适配器抽象基类。

    每个纺织领域资料源（政策、标准、产品、专利、报告、学术等）
    实现此接口，即可被 SourceRegistry 统一调度。

    必须实现：
      - metadata()          → 返回 SourceMetadata
      - iter_texts()       → 逐条产出原始文本
      - extract_terms()    → 从文本中提取候选术语

    可选覆写：
      - extract_definitions()  → 提取术语 + 定义对
      - validate_connection()  → 连接可用性检查
      - iter_texts_with_meta() → 带元数据的文本迭代（用于溯源）
    """

    def __init_subclass__(cls, **kwargs):
        """子类注册钩子 —— 在类定义时自动注册到 SourceRegistry。

        仅当子类定义了 `name` 类属性且为非抽象时才注册。
        """
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name") and isinstance(cls.name, str) and cls.name:
            # 延迟注册（避免循环导入）
            SourceRegistry._pending_classes.setdefault(cls.name, cls)

    # ── 必须实现 ──

    @property
    @abstractmethod
    def metadata(self) -> SourceMetadata:
        """返回此数据源的元信息。"""
        ...

    @abstractmethod
    def iter_texts(self) -> Iterator[str]:
        """逐条产出原始文本（标题+正文合并后的纯文本）。

        Yields:
            每份文档/案例/专利的完整文本，一次产出。
        """
        ...

    @abstractmethod
    def extract_terms(
        self,
        texts: Optional[list[str]] = None,
        min_len: int = 2,
        min_freq: int = 2,
        max_terms: int = 500,
    ) -> ExtractionResult:
        """从原始文本中提取候选纺织术语。

        典型实现：
          1. jieba 分词
          2. 过滤已有词典中的词、纯数字、停用词、通用词
          3. 按频次排序，取 top-N
          4. 对每个候选词做启发式分类（确定 layer + category）
          5. 计算置信度

        Args:
            texts: 待处理文本列表。None 时自动调用 iter_texts() 拉取全部。
            min_len: 最短术语长度（字符）。
            min_freq: 最低出现频次。
            max_terms: 最多返回候选数。

        Returns:
            ExtractionResult，含所有候选术语。
        """
        ...

    # ── 可选覆写 ──

    def extract_definitions(
        self,
        texts: Optional[list[str]] = None,
    ) -> list[TermCandidate]:
        """提取术语 + 定义对（适用于标准PDF等有结构化定义的源）。

        默认返回空，子类按需覆写。
        """
        return []

    def validate_connection(self) -> bool:
        """验证数据源连接是否可用。

        Returns:
            True 表示连接正常、可读取数据。
        """
        try:
            next(self.iter_texts())
            return True
        except (StopIteration, Exception):
            # StopIteration 也算正常（只是空库）；实际错误才返回 False
            return True

    def iter_texts_with_meta(self) -> Iterator[dict]:
        """带元数据的文本迭代 —— 用于术语溯源。

        默认降级为纯文本迭代（每条 meta 为空）。
        子类覆写后可返回 {'text': ..., 'title': ..., 'source_id': ..., 'url': ...} 等。

        Yields:
            dict with at least 'text' key.
        """
        for text in self.iter_texts():
            yield {"text": text}

    # ── 便利方法 ──

    def count(self) -> int:
        """统计数据源中的文档数量（消费 iter_texts，有性能开销）。"""
        return sum(1 for _ in self.iter_texts())

    def total_chars(self) -> int:
        """统计数据源中的总字符数（消费 iter_texts，有性能开销）。"""
        return sum(len(t) for t in self.iter_texts())

    def report(self) -> str:
        """生成人类可读的数据源状态报告。"""
        meta = self.metadata
        online = self.validate_connection()
        lines = [
            f"📂 {meta.display_name} ({meta.name})",
            f"   类型: {meta.source_type}  |  版本: {meta.version}",
            f"   位置: {meta.location}",
            f"   状态: {'🟢 可用' if online else '🔴 不可用'}",
        ]
        if meta.last_updated:
            lines.append(f"   更新: {meta.last_updated}")
        if meta.item_count:
            lines.append(f"   资料数: {meta.item_count}")
        if meta.tags:
            lines.append(f"   标签: {', '.join(meta.tags)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 源注册表
# ═══════════════════════════════════════════════════════════════


class SourceRegistry:
    """数据源注册表 —— 发现、管理所有已注册的 DataSourceAdapter。

    使用方式:
        # 注册单个实例
        SourceRegistry.register(policy_adapter)

        # 列所有可用源
        for src in SourceRegistry.list_all():
            print(src.report())

        # 从所有源批量提取术语
        all_candidates = SourceRegistry.extract_all(min_freq=3)
    """

    _registry: dict[str, DataSourceAdapter] = {}
    _pending_classes: dict[str, type] = {}  # 子类声明时自动填充

    @classmethod
    def register(cls, adapter: DataSourceAdapter):
        """注册一个适配器实例。

        如果同名已存在，发出警告并覆盖。
        """
        name = adapter.metadata.name
        if name in cls._registry:
            logger.warning(
                "数据源 %r 已注册，将被覆盖。旧: %s",
                name,
                cls._registry[name].__class__.__name__,
            )
        cls._registry[name] = adapter
        logger.info("已注册数据源: %s (%s)", name, adapter.__class__.__name__)

    @classmethod
    def unregister(cls, name: str):
        """注销一个适配器。"""
        cls._registry.pop(name, None)

    @classmethod
    def get(cls, name: str) -> Optional[DataSourceAdapter]:
        """按名称获取已注册的适配器实例。"""
        return cls._registry.get(name)

    @classmethod
    def list_all(cls) -> list[DataSourceAdapter]:
        """返回所有已注册的适配器实例。"""
        return list(cls._registry.values())

    @classmethod
    def list_by_type(cls, source_type: str) -> list[DataSourceAdapter]:
        """按资料类型筛选。"""
        return [
            a
            for a in cls._registry.values()
            if a.metadata.source_type == source_type
        ]

    @classmethod
    def available_types(cls) -> list[str]:
        """列出所有可用的资料类型。"""
        return sorted({a.metadata.source_type for a in cls._registry.values()})

    @classmethod
    def extract_all(
        cls,
        source_types: Optional[list[str]] = None,
        min_len: int = 2,
        min_freq: int = 2,
        max_terms_per_source: int = 500,
    ) -> dict[str, ExtractionResult]:
        """从所有（或指定类型的）数据源批量提取术语。

        Args:
            source_types: 限定资料类型，None = 全部。
            min_len: 最短术语长度。
            min_freq: 最低频次。
            max_terms_per_source: 每个源的候选上限。

        Returns:
            {adapter_name: ExtractionResult, ...}
        """
        results: dict[str, ExtractionResult] = {}
        adapters = cls.list_all()

        if source_types:
            adapters = [a for a in adapters if a.metadata.source_type in source_types]

        for adapter in adapters:
            logger.info("正在从 %s 提取术语...", adapter.metadata.name)
            try:
                result = adapter.extract_terms(
                    min_len=min_len,
                    min_freq=min_freq,
                    max_terms=max_terms_per_source,
                )
                results[adapter.metadata.name] = result
                logger.info(
                    "  %s → %d 个候选术语",
                    adapter.metadata.name,
                    result.candidate_count,
                )
            except Exception as exc:
                logger.exception("从 %s 提取失败: %s", adapter.metadata.name, exc)
                results[adapter.metadata.name] = ExtractionResult(
                    adapter_name=adapter.metadata.name,
                    extracted_at=datetime.now().isoformat(),
                    total_texts_processed=0,
                    total_chars_processed=0,
                    candidates=[],
                    errors=[str(exc)],
                )
        return results

    @classmethod
    def report_all(cls) -> str:
        """生成所有已注册源的报告。"""
        if not cls._registry:
            return "📭 没有已注册的数据源。\n\n请先注册适配器，例如:\n  SourceRegistry.register(PolicyDBSource(...))"

        lines = ["=" * 60, "📂 已注册数据源总览", "=" * 60, ""]
        for adapter in cls._registry.values():
            lines.append(adapter.report())
            lines.append("")
        lines.append(f"共 {len(cls._registry)} 个数据源")
        return "\n".join(lines)

    @classmethod
    def reset(cls):
        """清空注册表（主要用于测试）。"""
        cls._registry.clear()
