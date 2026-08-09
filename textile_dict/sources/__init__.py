"""数据源适配器 — 可插拔的资料源接入框架。

每个纺织领域资料库（政策、标准、产品、专利等）实现一个
DataSourceAdapter 子类，注册到 SourceRegistry，即可自动
接入本词典项目的 PMI 新词发现、术语校验和合并流程。

核心对象:
  SourceRegistry   — 全局适配器注册表，发现 + 调度
  DataSourceAdapter — 抽象基类，定义 extract_terms / iter_texts 接口

内建适配器:
  PolicyDBSource    — SQLite 政策数据库
  StandardSource    — GB/T 纺织标准 PDF + JSON
  ProductSource     — 专精特新企业案例和产品资料

用法:
    from textile_dict.sources import SourceRegistry
    from textile_dict.sources.adapters import PolicyDBSource

    # 注册适配器
    SourceRegistry.register(PolicyDBSource())

    # 查看所有源
    print(SourceRegistry.report_all())

    # 从所有源提取术语
    results = SourceRegistry.extract_all(min_freq=3)
"""

from textile_dict.sources.adapter import (
    DataSourceAdapter,
    ExtractionResult,
    SourceMetadata,
    SourceRegistry,
    TermCandidate,
)
from textile_dict.sources import shared

__all__ = [
    "DataSourceAdapter",
    "ExtractionResult",
    "SourceMetadata",
    "SourceRegistry",
    "TermCandidate",
    "shared",
]

