# 标注/训练数据集

> 版本化管理的 NLP 训练和评估数据集。

## 目录结构

```
datasets/
├── ner/              # NER 命名实体识别训练/评估集
├── m2_coverage/      # M2 术语覆盖率人工标注样本 (20篇政策全文)
├── bilingual/        # 中英术语对齐数据集
├── classification/   # 文本分类数据集（待建）
└── README.md
```

## 数据集清单

| 数据集 | 格式 | 规模 | 用途 | 状态 |
|:--|:--|:--|:--|:--|
| ner_train.jsonl | JSONL | ~158 KB | NER 模型训练 | ✅ |
| ner_eval.json | JSON | ~0.5 KB | NER 模型评估 | ✅ |
| m2_coverage/ | TXT + CSV | 20 份政策全文 | M2 术语覆盖率人工标注 | ✅ |
| bilingual/ | — | 0 | 中英术语对齐 | ⬜ 待建 |

## 数据格式

### NER 训练数据 (JSONL)

每行一条标注样本：
```json
{"text": "...", "entities": [{"start": 0, "end": 5, "label": "AGENCY"}, ...]}
```

### 术语对齐 (CSV)

```csv
zh_term,en_term,source,confidence
涡流纺纱,air-jet spinning,GB/T 5705-2018,high
```
