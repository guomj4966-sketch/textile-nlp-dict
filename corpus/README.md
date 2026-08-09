# 领域语料库

> 精选纺织行业文本，按来源-主题-时期三重标注。

## 目录结构

```
corpus/
├── policy/       # 纺织政策全文（中央 + 省级）
├── standard/     # GB/T 纺织标准文本
├── report/       # 行业研究报告、白皮书
├── academic/     # 学术论文全文（脱敏后）
└── business/     # 企业案例、产品目录
```

## 数据格式

每份文本文件头部包含标准元数据（YAML frontmatter）：

```yaml
---
id: corpus_0001
source: 国务院政策文件库
source_url: https://...
publish_date: 2026-03-15
author: 工业和信息化部
topic: [纺织工业, 数字化转型]
textile_sectors: [制造端, 智能制造端]
language: zh-CN
quality: raw  # raw | cleaned | annotated
---
```

## 当前状态

⬜ 框架已建立，内容待填充。

## 入库标准

- 文本可读（非扫描件，或已有 OCR 文本）
- 与纺织行业直接相关（参见 docs/语料库入库标准.md）
- 版权允许研究用途使用
