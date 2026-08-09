"""jieba 加载器 — 将词典注册为 jieba 用户词典。

从 data/jieba_dict.txt 加载，也可直接从 lexicon_v2.yaml 动态生成。
"""

from pathlib import Path

import jieba


def load_jieba_dict(path: str | None = None):
    """加载 jieba 自定义词典。

    调用后，jieba 分词将识别纺织行业术语。

    Args:
        path: jieba_dict.txt 的路径。默认使用包内置版本。

    Returns:
        True 表示加载成功。

    Raises:
        FileNotFoundError: 词典文件不存在。
    """
    if path is None:
        data_dir = Path(__file__).parent.parent / "data"
        path = str(data_dir / "jieba_dict.txt")

    dict_path = Path(path)
    if not dict_path.exists():
        raise FileNotFoundError(
            f"jieba 词典文件不存在: {dict_path}\n"
            f"运行 python scripts/merge_lexicon.py --export-jieba-only 生成。"
        )

    jieba.load_userdict(str(dict_path))
    return True
