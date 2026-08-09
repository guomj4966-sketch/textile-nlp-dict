# 纺织行业中文 NLP 领域词典

> 面向纺织行业的政策分析、市场研究、供应链调研、ESG 合规等领域的中文 NLP 词典与术语库。

[![Version](https://img.shields.io/badge/词典-v2.12-blue)](textile_dict/data/lexicon_v2.yaml)
[![Terms](https://img.shields.io/badge/词条-4,701-green)](textile_dict/data/lexicon_v2.yaml)
[![Python](https://img.shields.io/badge/Python-≥3.10-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Gitee](https://img.shields.io/badge/Gitee-tom--hj%2Ftextile--nlp--dict-red)](https://gitee.com/tom-hj/textile-nlp-dict)

---

## 项目定位

纺织行业中文 NLP 的**领域知识基座**，经过九轮迭代建设，涵盖 4,701 个纺织产业链术语。

### 核心能力

- **jul 分词增强**：加载后 jieba 将识别纺织领域术语（如"莱赛尔/纤维/涡流纺/纱"而非"莱/赛尔/纤维/涡流/纺纱"）
- **七层结构化分类**：机构 → 文书类型 → 产业链 → 政策语义 → 交叉领域 → 地理 → 时间
- **CRF NER 模型**：命名实体识别 F1=0.93
- **术语搜索**：通过 Lexicon 引擎进行中英文、模糊匹配查询

---

## 快速开始

### 安装

```bash
pip install git+https://gitee.com/ghj123456/textile-nlp-dict.git
# 或
pip install git+https://github.com/ghj123456/textile-nlp-dict.git
```

安装 NER 可选依赖：

```bash
pip install "textile-nlp-dict[ner]"
```

### 使用

```python
# 1. 加载 jieba 自定义词典（一次调用）
from textile_dict import load_jieba_dict
load_jieba_dict()

import jieba
words = jieba.lcut("工业和信息化部发布纺织工业数字化转型行动方案")
# 工业和信息化部/发布/纺织工业/数字化转型/行动方案

# 2. 查询词典
from textile_dict.core import Lexicon
lex = Lexicon()

# 搜索术语
lex.search("涡流纺")  # → [{term, layer, category}, ...]

# 按层查询
lex.get_layer("layer_3_textile_chain")  # → 产业链全部术语

# 按类别查询
lex.terms_by_category("layer_3_textile_chain", "3_织造")
```

---

## 词典版本

| 指标 | 数值 |
|:--|:--|
| 词典版本 | v2.12 |
| 唯一词条 | 4,701 词 |
| jieba 词条 | 2,544 条 |
| 分层结构 | 7 层 |
| 有定义术语 | 1,287 条（39%） |
| 中英对照术语 | 609 条 |
| CRF NER 模型 | F1=0.93 |
| M1 分词修复率 | 51.1% |

### 七层架构

| Layer | 名称 | 词条数 |
|:--|:--|:--|
| Layer 1 | 发文机关 | 中央部委 + 省级厅局 |
| Layer 2 | 文书类型 | 解读/新闻/视频/行政文书 |
| Layer 3 | 纺织产业链 | 八段：原料→纺纱→织造→染整→成衣→终端品类→智能制造→绿色低碳循环 |
| Layer 4 | 政策语义 | 约束/排除/激励方向 |
| Layer 5 | 交叉领域 | 绿色合规 / 贸易经济 / 数字经济 / 通用 |
| Layer 6 | 地理/产业集群 | 省份 + 产业集群 |
| Layer 7 | 时间表达 | 政策时间标记 |

---

## 消费项目

- **纺织行业政策研报系统**：月报/学术写作中的分词和术语识别
- **专精特新产品新技术智汇服务系统**（计划中）：企业案例的自动分类和关键词提取

---

## 贡献新词

如果你的项目发现了纺织领域未被词典收录的术语，欢迎贡献：

```python
# 通过 Issue / PR 提交
# - 术语名
# - 所属层级和分类
# - 来源（PDF 文件、政策文本、企业案例等）
# - 英文对应词（如有）
```

---

## 项目结构

```
textile-nlp-dict/
├── textile_dict/          # Python 库（pip install 安装的内容）
│   ├── core/              # Lexicon 查询引擎
│   ├── domains/           # 领域子集（政策 / 产业链 / 绿色合规）
│   ├── tools/             # NER 训练 / 术语验证 / 自动分类
│   ├── models/            # 预训练模型
│   ├── sources/           # 外部资料源适配器
│   └── data/              # 数据文件（打包发布）
├── tools/                 # 构建工具（不打包，仅开发使用）
├── corpus/                # 语料库
├── datasets/              # 标注/训练数据集
└── docs/                  # 文档
```

---

## 路线图

- [x] v2.12 七层词典 + 4,701 词（行业分类+HS编码+数字经济术语）
- [x] v3.0 术语定义覆盖率从 2.6% 提升至 39%（已超额完成）
- [ ] v3.1 中英双语术语对齐数据集
- [ ] v3.2 领域语料库（政策全文 + 标准文本 + 行业报告）
- [ ] v4.0 纺织专用 NLP 模型

---

## 许可

MIT License
