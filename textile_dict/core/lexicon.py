"""词典加载器 — 从 YAML 文件加载七层词典并提供查询接口。

对应 data/lexicon_v2.yaml 的渐进式数据结构。

用法:
    from textile_dict.core import Lexicon

    lex = Lexicon()
    # 查询所有术语
    all_terms = lex.all_terms()
    # 查询某一层
    chain_terms = lex.get_layer("layer_3_textile_chain")
    # 搜索
    results = lex.search("染色")
"""

import re
from pathlib import Path
from typing import Iterator, Optional

import yaml


class Lexicon:
    """纺织行业 NLP 词典的加载和查询引擎。

    从 lexicon_v2.yaml 加载七层词典，提供统一查询接口。
    """

    def __init__(self, lexicon_path: Optional[str] = None):
        """初始化词典。

        Args:
            lexicon_path: 词典 YAML 文件路径。默认使用包内置的 v2.x 版本。
        """
        if lexicon_path is None:
            lexicon_path = (
                Path(__file__).parent.parent
                / "data"
                / "lexicon_v2.yaml"
            )

        self._path = Path(lexicon_path)
        if not self._path.exists():
            raise FileNotFoundError(f"词典文件不存在: {self._path}")

        with open(self._path, encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

        self.meta = self._data.get("meta", {})
        self.layers = self._data.get("layers", {})

        # 构建术语索引
        self._term_index: dict[str, list[dict]] = {}
        self._build_index()

    # ─── 公开 API ────────────────────────────────────────

    @property
    def version(self) -> str:
        """词典版本号。"""
        return self.meta.get("version", "unknown")

    @property
    def total_terms(self) -> int:
        """词条总数（声明值）。"""
        return self.meta.get("total_terms", 0)

    def all_terms(self) -> Iterator[str]:
        """迭代所有唯一术语。"""
        seen = set()
        for term, _ in self._term_index.items():
            if term not in seen:
                seen.add(term)
                yield term

    def get_layer(self, layer_key: str) -> dict:
        """获取某一层的术语数据。

        Args:
            layer_key: 如 "layer_3_textile_chain"

        Returns:
            该层的 terms 字典。
        """
        layer = self.layers.get(layer_key)
        if layer is None:
            valid = ", ".join(self.layers.keys())
            raise KeyError(f"层级不存在: {layer_key}。有效层级: {valid}")
        return layer.get("terms", {})

    def get_term(self, term: str) -> list[dict]:
        """查询某个术语的所有出现位置。

        Returns:
            [{"layer": "layer_3_textile_chain", "category": "纺纱工艺", "definition": "..."}, ...]
        """
        return self._term_index.get(term, [])

    def search(self, query: str, fuzzy: bool = False) -> list[dict]:
        """搜索包含关键词的术语。

        Args:
            query: 搜索关键词
            fuzzy: True 时用正则模糊匹配

        Returns:
            [{"term": "染色布", "layer": "layer_3_textile_chain", "category": "染整"}, ...]
        """
        results = []
        for term, entries in self._term_index.items():
            if fuzzy:
                if re.search(query, term):
                    for entry in entries:
                        results.append({"term": term, **entry})
            else:
                if query in term:
                    for entry in entries:
                        results.append({"term": term, **entry})
        return results

    def terms_by_category(self, layer_key: str, category: str) -> list[str]:
        """获取某一类别下的所有术语。

        Args:
            layer_key: 层级键名
            category: 分类路径，如 "1_原料端 / 天然纤维"

        Returns:
            术语列表
        """
        layer = self.get_layer(layer_key)
        if isinstance(layer, dict):
            for cat, terms in layer.items():
                if cat == category:
                    if isinstance(terms, list):
                        return [t for t in terms if isinstance(t, str)]
                    elif isinstance(terms, dict):
                        result = []
                        for sub_terms in terms.values():
                            if isinstance(sub_terms, list):
                                result.extend(
                                    t for t in sub_terms if isinstance(t, str)
                                )
                        return result
        return []

    # ─── 内部方法 ────────────────────────────────────────

    def _build_index(self):
        """构建 term → [(layer, category, ...)] 倒排索引。"""
        for layer_key, layer_data in self.layers.items():
            terms = layer_data.get("terms", {})
            self._index_terms(terms, layer_key, "")

    def _index_terms(self, obj, layer_key: str, category: str):
        """递归索引术语。

        跳过元数据键 (description, source, aliases 等)，
        其余键名和字符串值都视为术语。
        """
        META_KEYS = {
            "description", "source", "note", "产业定位", "政策类型",
            "policy_count", "cluster_info", "type", "aliases", "definition",
        }

        if isinstance(obj, str) and len(obj) >= 2:
            entry = {"layer": layer_key, "category": category.rstrip("/")}
            # Collect additional metadata from parent if available
            self._term_index.setdefault(obj, []).append(entry)

        elif isinstance(obj, list):
            for item in obj:
                self._index_terms(item, layer_key, category)

        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k in META_KEYS:
                    # 不索引元数据键名，但递归索引元数据值
                    if isinstance(v, (dict, list)):
                        self._index_terms(v, layer_key, category)
                else:
                    # 键名本身是术语
                    new_cat = f"{category}/{k}" if category else k
                    if len(k) >= 2:
                        entry = {"layer": layer_key, "category": category.rstrip("/")}
                        self._term_index.setdefault(k, []).append(entry)
                    # 递归索引值
                    self._index_terms(v, layer_key, new_cat)
