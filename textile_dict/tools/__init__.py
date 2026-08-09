"""工具模块 — 术语验证、分类、NER 标注和训练工具。

提供:
  - term_validator: M2/M3 术语覆盖率和歧义消除率验证
  - auto_classify: 基于规则和模型的术语自动分类
  - annotate: NER 标注辅助工具
  - train: sklearn-crfsuite NER 模型训练
"""

from .term_validator import main as validate_main
from .auto_classify import main as classify_main

__all__ = ["validate_main", "classify_main"]
