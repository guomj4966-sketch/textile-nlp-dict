# Changelog

## [v2.8] — 2026-08-09

### 变更
- 独立为 standalone Python 包 (`textile-nlp-dict`)
- 数据路径修复：data/ 打包进 textile_dict/ 包内
- NER 工具从 scripts/nlp_dict/ner/ 迁移至 textile_dict/tools/
- 构建工具从 scripts/nlp_dict/ 迁移至 tools/build/
- 版本号统一为 YAML meta.version 单一来源
- 配置 Gitee + GitHub 双远程仓库
- 数据集和语料库目录骨架初始化

### 词典数据（本次未变更）
- 2,545 词条，七层结构
- CRF NER F1=0.93
- M1 分词修复率 51.1%

---

## [v2.0 — v2.7] — 2026-07 ~ 2026-08-05

详见政策研报项目 journal/2026-W31.md 和 journal/2026-W32-NLP词典.md。

- v2.0 (777 词) → v2.6 (1,604 词) → v2.8 (2,545 词)
- Phase 0 质量收敛：jieba 校验 / PMI 重分类 / 外部源扩充 / M1 基线
- Phase 1A 子包化：pip install -e . 生效
- 外部资料源接入：HS Code Ch50-63 (285 词) + 专精特新产品 (133 词) + GB/T 标准名 (51 词)
- CRF NER 模型训练完成 (F1=0.93)
- GB/T PDF 文本提取 v3（四类 PDF 编码分类）
