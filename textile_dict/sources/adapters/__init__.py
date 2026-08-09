"""具体数据源适配器实现。

当前已支持:
  - PolicyDBSource      — SQLite 政策数据库
  - StandardSource      — GB/T 纺织标准 PDF
  - ProductSource       — 专精特新产品/企业案例

未来扩展:
  - PatentSource        — 纺织专利文献
  - AcademicSource      — 学术论文
  - ReportSource        — 行业研报/白皮书
"""

from textile_dict.sources.adapters.policy_db import PolicyDBSource
from textile_dict.sources.adapters.standards import StandardSource
from textile_dict.sources.adapters.products import ProductSource

__all__ = [
    "PolicyDBSource",
    "StandardSource",
    "ProductSource",
]
