"""core/ — 内核模块，所有项目共享。

提供词典加载、查询、实体类型定义等基础能力。
"""

from .lexicon import Lexicon
from .entities import Term, Entity, Relation
from .loader import load_jieba_dict

__all__ = ["Lexicon", "Term", "Entity", "Relation", "load_jieba_dict"]
