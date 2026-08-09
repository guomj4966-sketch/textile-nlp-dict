"""纺织行业中文 NLP 领域词典 — Python 包入口。

三支柱架构之"支柱三 NLP 词典"，面向纺织行业的政策分析、市场研究、
供应链调研、ESG 合规等主题项目，提供统一的分词和术语能力。

用法:
    # 加载 jieba 自定义词典
    from textile_dict import load_jieba_dict
    load_jieba_dict()  # jieba 将识别纺织领域术语

    # 查询词典
    from textile_dict import Lexicon
    lex = Lexicon()
    print(lex.get_layer("layer_3_textile_chain"))

    # 领域子集
    from textile_dict.domains import policy, industry_chain
    from textile_dict.tools import term_validator
"""

from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

_PACKAGE_DIR = Path(__file__).parent
DATA_DIR = _PACKAGE_DIR / "data"
JIEBA_DICT = DATA_DIR / "jieba_dict.txt"
LEXICON_PATH = DATA_DIR / "lexicon_v2.yaml"

# 版本号从 YAML meta.version 读取（单一事实源）
__version__ = None  # 延迟读取，见 get_version()


def load_jieba_dict():
    """加载词典到 jieba，后续分词将识别纺织领域术语。

    示例:
        import jieba
        from textile_dict import load_jieba_dict
        load_jieba_dict()
        words = jieba.lcut("工业和信息化部发布纺织工业数字化转型行动方案")
        # 工业和信息化部/发布/纺织工业/数字化转型/行动方案
    """
    import jieba

    if not JIEBA_DICT.exists():
        raise FileNotFoundError(
            f"jieba 词典文件不存在: {JIEBA_DICT}\n"
            f"请运行: python tools/build/merge_lexicon.py --export-jieba-only"
        )

    jieba.load_userdict(str(JIEBA_DICT))
    return True


def get_version():
    """返回词典版本号（从 lexicon_v2.yaml meta.version 读取，单一事实源）。"""
    import yaml

    if LEXICON_PATH.exists():
        with open(LEXICON_PATH, encoding="utf-8") as f:
            lex = yaml.safe_load(f)
        return lex.get("meta", {}).get("version", "v?.?")
    return "v?.?"
