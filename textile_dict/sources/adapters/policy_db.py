"""纺织政策数据库适配器 — 从 policy.db 提取术语。

数据源类型: policy
数据源位置: ../collector/policy.db (相对于 OneDrive 根目录)
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from textile_dict.sources.adapter import (
    DataSourceAdapter,
    ExtractionResult,
    SourceMetadata,
    TermCandidate,
)
from textile_dict.sources.shared import (
    classify_candidate,
    discover_terms_from_texts,
    is_textile_relevant,
)


class PolicyDBSource(DataSourceAdapter):
    """纺织政策数据库（SQLite）适配器。

    从 collector/policy.db 读取审核通过、与纺织直接或间接相关的
    政策原文全文，用 jieba 分词发现候选纺织术语。
    """

    name = "policy_db"

    def __init__(self, db_path: Optional[str | Path] = None):

        if db_path is None:
            # 默认路径：项目根目录下 collector/policy.db
            root = Path(__file__).parent.parent.parent.parent
            db_path = root / "collector" / "policy.db"
        self._db_path = Path(db_path)

        # 延迟连接
        self._conn: Optional[sqlite3.Connection] = None
        self._meta: Optional[SourceMetadata] = None

    # ── 元信息 ──

    @property
    def metadata(self) -> SourceMetadata:
        if self._meta is not None:
            return self._meta

        exists = self._db_path.exists()
        item_count = 0
        last_updated = ""
        if exists:
            try:
                conn = self._get_conn()
                row = conn.execute(
                    "SELECT COUNT(*), MAX(scraped_at) FROM policy "
                    "WHERE review_status = '通过' "
                    "AND (textile_relevance = '直接' OR textile_relevance = '间接') "
                    "AND doc_type = '政策原文'"
                ).fetchone()
                if row:
                    item_count = row[0] or 0
                    last_updated = row[1] or ""
            except Exception:
                pass

        self._meta = SourceMetadata(
            name="policy_db",
            display_name="纺织政策数据库",
            source_type="policy",
            version="1.0",
            description="中央+省级纺织相关政策全文，经审核标注（textile_relevance + review_status）",
            location=str(self._db_path),
            item_count=item_count,
            last_updated=last_updated,
            tags=["纺织", "政策", "中央部委", "省级厅局", "2024-2026"],
        )
        return self._meta

    # ── 连接 ──

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self._db_path.exists():
                raise FileNotFoundError(f"政策数据库不存在: {self._db_path}")
            self._conn = sqlite3.connect(
                f"file:{self._db_path}?mode=ro", uri=True
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def validate_connection(self) -> bool:
        if not self._db_path.exists():
            return False
        try:
            conn = self._get_conn()
            conn.execute("SELECT 1 FROM policy LIMIT 1")
            return True
        except Exception:
            return False

    # ── 文本迭代 ──

    def iter_texts(self) -> Iterator[str]:
        """逐条产出政策「标题 + 正文」合并后的纯文本。"""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT title, full_text
            FROM policy
            WHERE review_status = '通过'
              AND (textile_relevance = '直接' OR textile_relevance = '间接')
              AND doc_type = '政策原文'
              AND full_text IS NOT NULL AND full_text != ''
            ORDER BY publish_date DESC
        """)

        for row in rows:
            title = row["title"] or ""
            body = row["full_text"] or ""
            # 清理 HTML 标签和链接
            body = re.sub(r"<[^>]+>", "", body)
            body = re.sub(r"https?://\S+", "", body)
            body = re.sub(r"\s+", " ", body)
            combined = title + "。" + body
            if len(combined) > 50:
                yield combined

    def iter_texts_with_meta(self) -> Iterator[dict]:
        """带元数据的文本迭代。"""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT policy_id, title, full_text, publish_date, issuing_authority,
                   textile_relevance, source_url
            FROM policy
            WHERE review_status = '通过'
              AND (textile_relevance = '直接' OR textile_relevance = '间接')
              AND doc_type = '政策原文'
              AND full_text IS NOT NULL AND full_text != ''
            ORDER BY publish_date DESC
        """)

        for row in rows:
            title = row["title"] or ""
            body = row["full_text"] or ""
            body = re.sub(r"<[^>]+>", "", body)
            body = re.sub(r"https?://\S+", "", body)
            body = re.sub(r"\s+", " ", body)
            combined = title + "。" + body
            if len(combined) < 50:
                continue
            yield {
                "text": combined,
                "title": title,
                "source_id": row["policy_id"],
                "publish_date": row["publish_date"],
                "issuing_authority": row["issuing_authority"],
                "textile_relevance": row["textile_relevance"],
                "source_url": row["source_url"],
            }

    # ── 术语提取 ──

    def extract_terms(
        self,
        texts: Optional[list[str]] = None,
        min_len: int = 2,
        min_freq: int = 2,
        max_terms: int = 500,
    ) -> ExtractionResult:
        started_at = datetime.now().isoformat()

        if texts is None:
            texts = list(self.iter_texts())

        total_chars = sum(len(t) for t in texts)

        # 用共享提取引擎
        discovered = discover_terms_from_texts(
            texts,
            min_len=min_len,
            min_freq=min_freq,
            max_terms=max_terms,
        )

        candidates: list[TermCandidate] = []
        for term, freq in discovered.most_common():
            if len(candidates) >= max_terms:
                break

            # 只保留与纺织相关的
            if not is_textile_relevant(term):
                continue

            layer, category = classify_candidate(term)
            # 简单的置信度估计（基于词频倒数 + 相关性）
            confidence = min(1.0, 0.3 + 0.05 * freq)

            candidates.append(
                TermCandidate(
                    term=term,
                    source_name=self.name,
                    layer=layer,
                    category=category,
                    frequency=freq,
                    confidence_score=round(confidence, 2),
                )
            )

        return ExtractionResult(
            adapter_name=self.name,
            extracted_at=started_at,
            total_texts_processed=len(texts),
            total_chars_processed=total_chars,
            candidates=candidates,
            stats={
                "raw_discovered": len(discovered),
                "textile_relevant": len(candidates),
                "discarded_non_textile": len(discovered) - len(candidates),
            },
        )

    def close(self):
        """关闭数据库连接。"""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
