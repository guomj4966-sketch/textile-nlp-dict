"""基于词典自动标注数据，训练轻量级纺织领域 NER 模型。

训练策略：
  1. 加载 ner_train.jsonl（字符级 BIO 标注）
  2. 转为词级 BIO（jieba 分词 → 词级标签 = 词内字符标签的众数）
  3. 使用 bert-base-chinese 微调 token classification
  4. LoRA 轻量化适配（可选 --lora）
  5. 导出模型到 models/ner_model/

运行方式:
    # 自动降级：GPU → CPU → LoRA → 规则模型（sklearn CRF）
    python scripts/nlp_dict/ner/train.py
    python scripts/nlp_dict/ner/train.py --data scripts/nlp_dict/data/ner_train.jsonl
    python scripts/nlp_dict/ner/train.py --epochs 8 --batch-size 8 --lora

依赖:
    pip install torch transformers datasets seqeval accelerate peft
    # 或者（如果上述安装失败）：
    pip install sklearn-crfsuite  # 备选方案，仅需 ~2MB

输出:
    - models/ner_model/              ← 模型权重 + config
    - scripts/nlp_dict/data/ner_eval.json  ← 评估结果

数据源:
    - scripts/nlp_dict/data/ner_train.jsonl  ← annotate.py 产出
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import jieba

jieba.setLogLevel(20)

# ============================================================
# 配置
# ============================================================

DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_PATH = DATA_DIR / "ner_train.jsonl"
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models" / "ner_model"

LABELS = ["O", "B-AGENCY", "I-AGENCY", "B-DOC_TYPE", "I-DOC_TYPE",
          "B-TEXTILE", "I-TEXTILE", "B-POLICY", "I-POLICY",
          "B-GEO", "I-GEO", "B-TIME", "I-TIME"]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


# ============================================================
# 数据加载 & 字符→词级转换
# ============================================================


def load_char_data(path):
    """加载字符级 BIO 标注数据。"""
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


def char_to_word_bio(tokens, labels):
    """将字符级 BIO 转为词级 BIO（用 jieba 分词）。

    词级标签 = 词内字符标签的众数（B/I → B 优先）。
    """
    text = "".join(tokens)
    words = jieba.lcut(text)

    word_labels = []
    pos = 0
    for word in words:
        wlen = len(word)
        char_labels = labels[pos:pos + wlen]

        # 取众数标签（排除 O）
        non_o = [l for l in char_labels if l != "O"]
        if not non_o:
            word_labels.append("O")
        else:
            # 有 B- 则取第一个 B-
            b_labels = [l for l in non_o if l.startswith("B-")]
            if b_labels:
                word_labels.append(b_labels[0])
            else:
                # 全部是 I-，取最常见的
                most_common = Counter(non_o).most_common(1)[0][0]
                word_labels.append(most_common)

        pos += wlen

    # 修复：连续的 I- 前面如果没有 B-，转为 O
    for i in range(len(word_labels)):
        if word_labels[i].startswith("I-"):
            entity = word_labels[i][2:]
            if i == 0 or not word_labels[i - 1].endswith(entity):
                word_labels[i] = "O"

    return words, word_labels


def prepare_datasets(data, test_ratio=0.2):
    """准备训练/验证数据集（词级）。"""
    all_words = []
    all_labels = []

    for item in data:
        tokens = item["tokens"]
        labels = item["labels"]
        words, word_labels = char_to_word_bio(tokens, labels)
        if words:  # 确保非空
            all_words.append(words)
            all_labels.append(word_labels)

    # 切分
    split = int(len(all_words) * (1 - test_ratio))
    train_words = all_words[:split]
    train_labels = all_labels[:split]
    val_words = all_words[split:]
    val_labels = all_labels[split:]

    return train_words, train_labels, val_words, val_labels


# ============================================================
# 训练方案 A: Transformer (bert-base-chinese)
# ============================================================


def train_transformer(train_words, train_labels, val_words, val_labels, args):
    """使用 bert-base-chinese 微调 token classification。"""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, DataCollatorForTokenClassification,
        EarlyStoppingCallback,
    )
    import numpy as np
    from seqeval.metrics import classification_report, f1_score

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # 数据类
    class NERDataset(Dataset):
        def __init__(self, words_list, labels_list, tokenizer, max_len=128):
            self.encodings = []
            self.labels_enc = []

            for words, labels in zip(words_list, labels_list):
                # Tokenize with word alignment
                tokenized = tokenizer(
                    words,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=max_len,
                    padding=False,
                )
                word_ids = tokenized.word_ids()

                # Align labels to subword tokens
                aligned = []
                prev_word = None
                for wid in word_ids:
                    if wid is None:
                        aligned.append(-100)  # special tokens
                    elif wid != prev_word:
                        aligned.append(LABEL2ID.get(labels[wid], 0))
                    else:
                        aligned.append(LABEL2ID.get(labels[wid], 0))
                    prev_word = wid

                self.encodings.append({k: torch.tensor(v) for k, v in tokenized.items()})
                self.labels_enc.append(torch.tensor(aligned))

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            item = {k: v for k, v in self.encodings[idx].items()}
            item["labels"] = self.labels_enc[idx]
            return item

    train_dataset = NERDataset(train_words, train_labels, tokenizer)
    val_dataset = NERDataset(val_words, val_labels, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 模型
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # 评估指标
    def compute_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)
        true_labels = [[ID2LABEL[l] for l in label if l != -100] for label in labels]
        pred_labels = [
            [ID2LABEL[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return {"f1": f1_score(true_labels, pred_labels)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # 训练
    print(f"\n🔧 开始训练 (device={'cuda' if torch.cuda.is_available() else 'cpu'})...")
    trainer.train()

    # 保存
    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"\n✅ 模型已保存: {MODEL_DIR}")

    # 评估
    eval_results = trainer.evaluate()
    return eval_results


# ============================================================
# 训练方案 B: CRF (sklearn-crfsuite) — 轻量备选
# ============================================================


def train_crf(train_words, train_labels, val_words, val_labels, args):
    """使用 CRF 训练词级 NER（不依赖 PyTorch/transformers）。

    适用于数据量小（<500 句）的场景，效果可能比大小模型更好。
    """
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics

    # 特征函数
    def word2features(sent, i):
        word = sent[i][0] if isinstance(sent[i], tuple) else sent[i]
        features = {
            'bias': 1.0,
            'word': word,
            'word[-3:]': word[-3:] if len(word) >= 3 else word,
            'word[-2:]': word[-2:] if len(word) >= 2 else word,
            'is_digit': word.isdigit(),
            'is_alpha': word.isalpha(),
            'len': len(word),
        }
        if i > 0:
            prev = sent[i - 1][0] if isinstance(sent[i - 1], tuple) else sent[i - 1]
            features.update({
                '-1:word': prev,
                '-1:word[-2:]': prev[-2:] if len(prev) >= 2 else prev,
            })
        else:
            features['BOS'] = True

        if i > 1:
            pp = sent[i - 2][0] if isinstance(sent[i - 2], tuple) else sent[i - 2]
            features['-2:word'] = pp

        if i < len(sent) - 1:
            nxt = sent[i + 1][0] if isinstance(sent[i + 1], tuple) else sent[i + 1]
            features['+1:word'] = nxt
            features['+1:word[-2:]'] = nxt[-2:] if len(nxt) >= 2 else nxt
        else:
            features['EOS'] = True

        return features

    def sent2features(sent):
        return [word2features(sent, i) for i in range(len(sent))]

    def sent2labels(sent):
        return [label for _, label in sent] if isinstance(sent[0], tuple) else []

    def sent2tokens(sent):
        return [w if isinstance(w, str) else w[0] for w in sent]

    # 构建 (token, label) 格式
    train_data = [list(zip(w, l)) for w, l in zip(train_words, train_labels)]
    val_data = [list(zip(w, l)) for w, l in zip(val_words, val_labels)]

    X_train = [sent2features(s) for s in train_data]
    y_train = [sent2labels(s) for s in train_data]
    X_val = [sent2features(s) for s in val_data]
    y_val = [sent2labels(s) for s in val_data]

    # 训练 CRF
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )

    print(f"\n🔧 训练 CRF 模型...")
    crf.fit(X_train, y_train)

    # 评估
    y_pred = crf.predict(X_val)
    f1 = metrics.flat_f1_score(y_val, y_pred, average='weighted',
                               labels=list(LABEL2ID.keys()))

    print(f"\n✅ CRF 模型训练完成 (F1={f1:.4f})")

    # 保存 CRF 模型
    import pickle
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "crf_model.pkl", "wb") as f:
        pickle.dump(crf, f)
    # 保存标签映射
    with open(MODEL_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, ensure_ascii=False)

    print(f"💾 模型已保存: {MODEL_DIR}")
    return {"f1": f1, "model_type": "CRF"}


# ============================================================
# 主流程
# ============================================================


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="纺织领域 NER 模型训练")
    parser.add_argument("--data", default=str(TRAIN_PATH), help="训练数据 JSONL 路径")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--lora", action="store_true", help="使用 LoRA (暂未实现，预留)")
    parser.add_argument("--force-crf", action="store_true", help="强制使用 CRF（即使有 PyTorch）")
    args = parser.parse_args()

    # 1. 加载数据
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ 训练数据不存在: {data_path}")
        print("   请先运行: python scripts/nlp_dict/ner/annotate.py")
        sys.exit(1)

    print(f"📖 加载训练数据: {data_path}")
    data = load_char_data(data_path)
    print(f"   共 {len(data)} 句")

    train_words, train_labels, val_words, val_labels = prepare_datasets(data)
    print(f"   训练集: {len(train_words)} 句, 验证集: {len(val_words)} 句")

    # 统计标签分布
    label_dist = Counter()
    for labels in train_labels + val_labels:
        for l in labels:
            if l != "O":
                label_dist[l] += 1
    print(f"   实体标签分布: {dict(label_dist)}")

    # 2. 选择训练方案
    use_transformer = False
    if not args.force_crf:
        try:
            import torch
            import transformers
            use_transformer = True
            print(f"\n✅ PyTorch {torch.__version__}, Transformers {transformers.__version__}")
        except ImportError:
            print("\n⚠️  PyTorch/Transformers 未安装")

    if use_transformer:
        print("🔧 使用 Transformer 方案 (bert-base-chinese)")
        print("   首次运行将下载模型 (~400MB)")
        results = train_transformer(train_words, train_labels, val_words, val_labels, args)
    else:
        print("🔧 使用 CRF 方案 (sklearn-crfsuite, ~2MB)")
        print("   pip install sklearn-crfsuite")
        try:
            results = train_crf(train_words, train_labels, val_words, val_labels, args)
        except ImportError:
            print("\n❌ sklearn-crfsuite 也未安装")
            print("   请选择: pip install torch transformers  (方案A)")
            print("   或:     pip install sklearn-crfsuite     (方案B, 更轻量)")
            sys.exit(1)

    # 3. 保存评估结果
    eval_path = DATA_DIR / "ner_eval.json"
    eval_results = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_path": str(data_path),
        "train_sentences": len(train_words),
        "val_sentences": len(val_words),
        "model_type": results.get("model_type", "transformer"),
        "metrics": results,
    }
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    print(f"\n📄 评估结果: {eval_path}")
    print("\n下一步:")
    print("  1. 用 evaluate.py 在新政策文本上测试模型效果")
    print("  2. 对错误标注交互式修正后重新训练")
    print("  3. 将模型集成到 auto_classify.py 中替代纯词典匹配")


if __name__ == "__main__":
    main()
