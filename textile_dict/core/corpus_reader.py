"""语料库读取器 — 遍历 corpus/ 下的文本文件并解析 YAML frontmatter 元数据。

corpus/ 目录结构:
    corpus/
    ├── policy/       # 纺织政策全文（中央+省级）
    ├── standard/     # GB/T 纺织标准文本
    ├── report/       # 行业研究报告、白皮书
    ├── academic/     # 学术论文全文（脱敏后）
    └── business/     # 企业案例、产品目录

每份 .txt 文件含 YAML frontmatter（以 `---` 开头和结尾的元数据块）。

用法:
    from textile_dict.core.corpus_reader import CorpusReader

    reader = CorpusReader()
    # 遍历所有政策文本
    for meta, body in reader.iter_category("policy"):
        print(meta["title"], len(body))

    # 搜索
    results = reader.search("涡流纺", categories=["policy", "standard"])

    # 获取统计
    stats = reader.stats()
    print(f"政策: {stats['policy']['count']} 篇, {stats['policy']['total_chars']} 字")
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import yaml

# 默认语料库路径（相对于本文件：textile_dict/core/ → 上级 → 上级 → corpus/）
_DEFAULT_CORPUS = Path(__file__).parent.parent.parent / "corpus"

# 支持的语料分类
VALID_CATEGORIES = ("policy", "standard", "report", "academic", "business")


@dataclass
class CorpusDocument:
    """语料库中的一份文档。"""

    meta: dict
    body: str
    file_path: str
    category: str

    @property
    def title(self) -> str:
        return self.meta.get("title", "")

    @property
    def publish_date(self) -> str:
        return self.meta.get("publish_date", "")

    @property
    def char_count(self) -> int:
        return len(self.body)


class CorpusReader:
    """语料库读取器 —— 遍历、搜索、统计 corpus/ 下的文本。

    每个 .txt 文件格式:
        ---
        id: corpus_0001
        source: 国务院政策文件库
        title: 纺织工业数字化转型行动方案
        publish_date: 2026-03-15
        author: 工业和信息化部
        topic: [纺织工业, 数字化转型]
        textile_sectors: [制造端, 智能制造端]
        ---
        （正文...）
    """

    def __init__(self, corpus_dir: Optional[str | Path] = None):

        self._root = Path(corpus_dir) if corpus_dir else _DEFAULT_CORPUS
        if not self._root.exists():
            raise FileNotFoundError(
                f"语料库目录不存在: {self._root}\n"
                f"请检查 corpus/ 目录是否已建立，或传入自定义路径。"
            )

    # ── 核心迭代 ──

    def iter_category(self, category: str) -> Iterator[tuple[dict, str]]:
        """遍历某一分类下的所有文档。

        Args:
            category: 分类名（policy/standard/report/academic/business）

        Yields:
            (metadata_dict, body_text) 元组。
        """
        if category not in VALID_CATEGORIES:
            raise ValueError(f"无效分类: {category}。有效值: {VALID_CATEGORIES}")

        cat_dir = self._root / category
        if not cat_dir.exists():
            return

        for txt_file in sorted(cat_dir.glob("*.txt")):
            meta, body = self._parse_file(txt_file)
            yield meta, body

    def iter_all(self) -> Iterator[tuple[str, dict, str]]:
        """遍历所有分类下的所有文档。

        Yields:
            (category, metadata_dict, body_text) 元组。
        """
        for cat in VALID_CATEGORIES:
            cat_dir = self._root / cat
            if not cat_dir.exists():
                continue
            for txt_file in sorted(cat_dir.glob("*.txt")):
                meta, body = self._parse_file(txt_file)
                yield cat, meta, body

    def documents(self) -> Iterator[CorpusDocument]:
        """遍历所有文档，返回 CorpusDocument 对象。"""
        for cat in VALID_CATEGORIES:
            cat_dir = self._root / cat
            if not cat_dir.exists():
                continue
            for txt_file in sorted(cat_dir.glob("*.txt")):
                meta, body = self._parse_file(txt_file)
                yield CorpusDocument(
                    meta=meta,
                    body=body,
                    file_path=str(txt_file),
                    category=cat,
                )

    # ── 搜索 ──

    def search(
        self,
        query: str,
        categories: Optional[list[str]] = None,
        fuzzy: bool = False,
    ) -> list[dict]:
        """搜索包含关键词的文档。

        Args:
            query: 搜索词。
            categories: 限定分类，None = 所有分类。
            fuzzy: True 启用正则匹配。

        Returns:
            [{category, file, title, match_snippet, ...}, ...]
        """
        results: list[dict] = []
        cats = categories or list(VALID_CATEGORIES)

        for cat in cats:
            if cat not in VALID_CATEGORIES:
                continue
            for meta, body in self.iter_category(cat):
                if fuzzy:
                    matches = list(re.finditer(query, body))
                else:
                    matches = [
                        m for m in re.finditer(re.escape(query), body)
                    ]

                if not matches:
                    continue

                for m in matches[:5]:  # 每篇最多 5 处
                    start = max(0, m.start() - 30)
                    end = min(len(body), m.end() + 30)
                    snippet = body[start:end].replace("\n", " ")
                    results.append(
                        {
                            "category": cat,
                            "title": meta.get("title", ""),
                            "file": meta.get("id", ""),
                            "match_position": m.start(),
                            "snippet": f"...{snippet}...",
                        }
                    )

        return results

    # ── 统计 ──

    def stats(self) -> dict:
        """返回语料库统计信息。

        Returns:
            {category: {"count": N, "total_chars": N, "date_range": (最早, 最晚)}}
        """
        result: dict[str, dict] = {}
        for cat in VALID_CATEGORIES:
            count = 0
            total_chars = 0
            dates: list[str] = []
            for meta, body in self.iter_category(cat):
                count += 1
                total_chars += len(body)
                d = meta.get("publish_date", "")
                if d:
                    dates.append(d)
            result[cat] = {
                "count": count,
                "total_chars": total_chars,
                "date_range": (min(dates), max(dates)) if dates else ("", ""),
            }
        return result

    def total_documents(self) -> int:
        """语料库中的文档总数。"""
        return sum(1 for _ in self.iter_all())

    def total_chars(self) -> int:
        """语料库中的总字符数。"""
        return sum(len(body) for _, _, body in self.iter_all())

    def report(self) -> str:
        """生成语料库人类可读报告。"""
        s = self.stats()
        lines = ["=" * 50, "📚 语料库状态", "=" * 50, ""]
        total_docs = 0
        total_chars = 0
        for cat in VALID_CATEGORIES:
            info = s[cat]
            if info["count"] == 0:
                lines.append(f"  📭 {cat:12s} — 空")
            else:
                d_from, d_to = info["date_range"]
                lines.append(
                    f"  📄 {cat:12s} {info['count']:>4} 篇  "
                    f"{info['total_chars']:>10,} 字  "
                    f"{d_from} ~ {d_to}"
                )
                total_docs += info["count"]
                total_chars += info["total_chars"]
        lines.append("")
        lines.append(f"  合计: {total_docs} 篇, {total_chars:,} 字")
        return "\n".join(lines)

    # ── 内部 ──

    @staticmethod
    def _parse_file(file_path: Path) -> tuple[dict, str]:
        """解析单个 .txt 文件，返回 (metadata, body)。"""
        raw = file_path.read_text(encoding="utf-8")

        meta: dict = {}
        body = raw

        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    meta = {}
                body = parts[2].strip()

        return meta, body
