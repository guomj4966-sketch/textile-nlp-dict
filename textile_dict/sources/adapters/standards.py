"""纺织标准规范适配器 — 从 GB/T 标准 PDF 和术语 JSON 提取术语+定义。

数据源类型: standard
数据源位置:
  - 标准 PDF: ../分类和术语标准规范/GBT+*.pdf
  - 提取结果: textile_dict/data/GB-T_*_terms.json

标准 PDF 分四类:
  🟢 UTF-8     → fitz 直接提取
  🟡 全角拉丁   → fitz + _fw2hw 解码
  🔴 乱码(OCR)  → Tesseract OCR 回退
  🟠 纯图片     → 跳过
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from textile_dict.sources.adapter import (
    DataSourceAdapter,
    ExtractionResult,
    SourceMetadata,
    TermCandidate,
)
from textile_dict.sources.shared import classify_candidate

# GB/T 标准目录（相对于 OneDrive 根目录）
_STANDARD_ROOT = Path(__file__).parent.parent.parent.parent / "分类和术语标准规范"
# 已提取的 JSON 术语文件
_EXTRACTED_DIR = Path(__file__).parent.parent / "data"

# 标准 → 领域映射
_STANDARD_REGISTRY: list[tuple[str, str, str]] = [
    # (filename, domain, encoding)
    ("GBT+30558-2025.pdf", "产业用纺织品", "utf8"),
    ("GBT+30420.4-2025.pdf", "缝制机械", "utf8"),
    ("GBT+46947-2025.pdf", "棉纤维", "utf8"),
    ("GBT+47198-2026.pdf", "针织横机", "utf8"),
    ("GBT+47771-2026.pdf", "聚丙烯腈纤维生产装备", "utf8"),
    ("GBT+44870-2024.pdf", "纤维碳化生产装备", "utf8"),
    ("GBT+38136-2019.pdf", "化学纤维分类", "utf8"),
    ("GBT+33278-2016.pdf", "粘扣带", "utf8"),
    ("GBT+6002.17-2025.pdf", "环锭捻线机", "utf8"),
    ("GBT+5705-2018.pdf", "棉纺织产品", "fullwidth"),
    ("GBT+26380-2022.pdf", "丝绸", "fullwidth"),
    ("GBT+4146.2-2017.pdf", "化学纤维产品", "fullwidth"),
    ("GBT+38111-2019.pdf", "玄武岩纤维", "fullwidth"),
    ("GBT+15557-2008.pdf", "服装", "garbled"),
    ("GBT+42693-2023.pdf", "应急产业用纺织品", "garbled"),
    ("GB 50514-2009 非织造布工厂设计规范.pdf", "非织造布工厂", "garbled"),
]


class StandardSource(DataSourceAdapter):
    """纺织 GB/T 标准术语适配器。

    从两个方面读取术语:
      1. 已 OCR/提取的 JSON 文件（textile_dict/data/GB-T_*_terms.json）
      2. 标准 PDF 文件名中隐含的术语关键词

    优先使用预提取的 JSON 文件，因为它包含完整的 term + definition + english。
    """

    name = "standards"

    def __init__(
        self,
        standards_dir: Optional[str | Path] = None,
        extracted_dir: Optional[str | Path] = None,
    ):
        self._standards_dir = Path(standards_dir) if standards_dir else _STANDARD_ROOT
        self._extracted_dir = (
            Path(extracted_dir) if extracted_dir else _EXTRACTED_DIR
        )
        self._meta: Optional[SourceMetadata] = None
        self._json_files: list[Path] = []

    # ── 元信息 ──

    @property
    def metadata(self) -> SourceMetadata:
        if self._meta is not None:
            return self._meta

        # 统计 PDF 和 JSON 文件
        n_pdf = len([
            fn for fn, _, _ in _STANDARD_REGISTRY
            if (self._standards_dir / fn).exists()
        ])
        json_files = sorted(self._extracted_dir.glob("GB-T_*_terms.json"))
        self._json_files = json_files

        total_terms = 0
        for jf in json_files:
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    total_terms += len(data)
                elif isinstance(data, dict):
                    total_terms += len(data.get("terms", data))
            except Exception:
                pass

        self._meta = SourceMetadata(
            name="standards",
            display_name="GB/T 纺织标准术语库",
            source_type="standard",
            version="1.0",
            description=f"15 份纺织 GB/T 标准，含 {total_terms} 条结构化术语定义",
            location=str(self._standards_dir),
            item_count=len(json_files),
            estimated_total_chars=total_terms * 80,  # rough estimate
            tags=["纺织", "GB/T", "术语", "定义", "中英对照"],
        )
        return self._meta

    # ── 连接 ──

    def validate_connection(self) -> bool:
        return any(self._extracted_dir.glob("GB-T_*_terms.json"))

    # ── 文本迭代 ──

    def iter_texts(self) -> Iterator[str]:
        """产出每个标准术语的定义文本（用于分词发现新术语）。"""
        json_files = sorted(self._extracted_dir.glob("GB-T_*_terms.json"))
        for jf in json_files:
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else data.get("terms", data) if isinstance(data, dict) else []
                if isinstance(items, dict):
                    items = list(items.values())
                for item in items:
                    if isinstance(item, dict):
                        definition = item.get("definition", "")
                        term = item.get("term", item.get("cn_term", ""))
                        text = f"{term}：{definition}"
                        if len(text) > 10:
                            yield text
                    elif isinstance(item, str) and len(item) > 10:
                        yield item
            except Exception:
                continue

    # ── 术语提取 ──

    def extract_terms(
        self,
        texts: Optional[list[str]] = None,
        min_len: int = 2,
        min_freq: int = 1,
        max_terms: int = 2000,
    ) -> ExtractionResult:
        """从标准 JSON 直接提取结构化术语，辅以分词发现。"""
        started_at = datetime.now().isoformat()
        candidates: list[TermCandidate] = []
        processed_count = 0

        # ── 直接从 JSON 提取结构化术语 ──
        json_files = sorted(self._extracted_dir.glob("GB-T_*_terms.json"))
        for jf in json_files:
            processed_count += 1
            source_name = jf.stem.replace("_terms", "")
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            items = data if isinstance(data, list) else data.get("terms", data) if isinstance(data, dict) else []
            if isinstance(items, dict):
                items = list(items.values())

            for item in items:
                if not isinstance(item, dict):
                    continue

                term = item.get("term", item.get("cn_term", ""))
                if not term or len(term) < 2:
                    continue

                definition = item.get("definition", "")
                english = item.get("english", item.get("en_term", ""))
                domain = item.get("category", item.get("domain", ""))

                layer, category = classify_candidate(term)
                if not category:
                    category = domain

                candidates.append(
                    TermCandidate(
                        term=term,
                        source_name=self.name,
                        layer=layer,
                        category=category,
                        definition=definition if len(definition) > 5 else None,
                        english=english,
                        confidence_score=0.85 if definition else 0.70,
                    )
                )

        # ── 从定义文本中再发现词组分词典未收录的术语 ──
        if len(candidates) < max_terms:
            raw_texts = texts or list(self.iter_texts())
            total_chars = sum(len(t) for t in raw_texts)

            # 用简单的中文词组提取（非 jieba，因为标准术语定义是高度结构化的）
            from collections import Counter
            from textile_dict.sources.shared import STOP_WORDS

            term_counter: Counter = Counter()
            for text in raw_texts:
                # 提取 2-6 字中文片段
                parts = re.findall(r"[一-鿿]{2,6}", text)
                for p in parts:
                    if p not in STOP_WORDS and len(p) >= min_len:
                        term_counter[p] += 1

            for term, freq in term_counter.most_common():
                if freq < min_freq:
                    break
                # 跳过已在候选中的
                if any(c.term == term for c in candidates):
                    continue
                if len(candidates) >= max_terms:
                    break

                layer, category = classify_candidate(term)
                candidates.append(
                    TermCandidate(
                        term=term,
                        source_name=self.name,
                        layer=layer,
                        category=category,
                        frequency=freq,
                        confidence_score=round(min(1.0, 0.2 + 0.05 * freq), 2),
                    )
                )
        else:
            total_chars = 0

        return ExtractionResult(
            adapter_name=self.name,
            extracted_at=started_at,
            total_texts_processed=processed_count,
            total_chars_processed=total_chars if "total_chars" in dir() else 0,
            candidates=candidates,
            stats={
                "json_files": len(json_files),
                "structured_terms": sum(
                    1 for c in candidates if c.definition
                ),
                "discovered_terms": sum(
                    1 for c in candidates if not c.definition
                ),
            },
        )

    def extract_definitions(
        self,
        texts: Optional[list[str]] = None,
    ) -> list[TermCandidate]:
        """提取包含定义的术语对（标准源的专属能力）。

        Returns:
            仅 definition 非空的候选术语。
        """
        result = self.extract_terms(texts=texts, max_terms=5000)
        return [c for c in result.candidates if c.definition]
