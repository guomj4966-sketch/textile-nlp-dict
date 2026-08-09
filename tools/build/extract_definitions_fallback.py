"""
从国民经济行业分类注释 + H6.json 提取定义和双语术语。

这是 GB/T PDF 无法自动解码后的替代方案：
  1. 行业分类注释: 提取 17xx/18xx 大类的详细定义
  2. H6.json: 963 条纺织 HS 分类中英文, 做双语术语对齐的基础

运行:
    PYTHONIOENCODING=utf-8 python scripts/nlp_dict/extract_definitions_fallback.py
"""

import sys, io, re, csv, json, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import yaml

DATA_DIR = Path(__file__).parent / "data"
LEXICON_PATH = DATA_DIR / "lexicon_v2.yaml"

# 从脚本位置向上4级到 OneDrive 根目录，再进入"分类和术语标准规范"
_ONEDRIVE = Path(__file__).parent.parent.parent.parent
_CLASSIFICATION_DIR = _ONEDRIVE / "分类和术语标准规范"

# ============================================================
# Source 1: H6.json → 双语术语对齐
# ============================================================

def extract_h6_bilingual():
    """从 H6.json 提取纺织章节 (Ch50-63) 的中英对照。"""
    h6_path = _CLASSIFICATION_DIR / "H6.json"
    with open(h6_path, encoding='utf-8') as f:
        h6 = json.load(f)

    results = h6['results']
    textile = []
    for r in results:
        text = r.get('text', '')
        code = r.get('id', '')
        # Filter to textile chapters: 50-63
        if re.match(r'^(5\d|6[0-3])', text):
            # Parse: "500100 - Silk; silk-worm cocoons suitable for reeling"
            m = re.match(r'^(\d{6})\s*-\s*(.+)', text)
            if m:
                hs_code = m.group(1)
                en_name = m.group(2).strip()
                textile.append((hs_code, en_name))

    print(f"   H6.json: {len(textile)} 条纺织 HS 6位码 (中英)")
    # Show 10 samples
    for code, name in textile[:10]:
        print(f"     {code}: {name}")
    return textile


# ============================================================
# Source 2: 国民经济行业分类注释 → 纺织大类定义
# ============================================================

def extract_industry_defs():
    """从行业分类注释提取纺织相关的定义文本。"""
    import openpyxl
    wb = openpyxl.load_workbook(
        _CLASSIFICATION_DIR / "《2017国民经济行业分类注释》（按第1号修改单修订）.xlsx",
        data_only=True)
    ws = wb['Sheet1']

    results = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        code = str(row[0]).strip() if row[0] else ''
        name = str(row[1]).strip() if len(row) > 1 and row[1] else ''
        note = str(row[2]).strip() if len(row) > 2 and row[2] else ''
        full_desc = str(row[3]).strip() if len(row) > 3 and row[3] else ''

        # Textile sectors: 17 (textile), 18 (apparel), 28 (chemical fiber)
        if code.startswith(('17', '18', '28')) and len(name) >= 2:
            # Clean: skip pure "excluding" entries
            if '不包含' in name and len(name) <= 6:
                continue
            definition = full_desc if full_desc else note
            if len(definition) >= 10:
                results.append((code, name, definition))

    wb.close()

    print(f"\n   行业分类注释: {len(results)} 条纺织相关定义")
    for code, name, defn in results[:15]:
        print(f"     {code} {name}: {defn[:100]}...")
    return results


# ============================================================
# 匹配到词典
# ============================================================

def update_lexicon_with_defs(lex, definitions_list, output_path):
    """将提取到的定义匹配到词典中并写入。

    definitions_list: [(code, term, definition), ...] from industry classification
    """
    # Build term → definition index
    def_map = {}
    for _, name, defn in definitions_list:
        # Try to match to lexicon terms
        if name not in def_map or len(defn) > len(def_map[name]):
            def_map[name] = defn

    updated = 0
    def update_obj(obj):
        nonlocal updated
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if k in ('description','source','note','产业定位','政策类型',
                        'policy_count','cluster_info','type','aliases','definition'):
                    if isinstance(v, (dict, list)):
                        update_obj(v)
                    continue
                # k is a term — does it have a definition?
                if k in def_map and 'definition' not in obj:
                    obj['definition'] = def_map[k]
                    updated += 1
                if isinstance(v, dict):
                    update_obj(v)
                elif isinstance(v, list):
                    for item in v: update_obj(item)
        elif isinstance(obj, list):
            for item in obj: update_obj(item)

    for lk, ld in lex.get('layers', {}).items():
        update_obj(ld.get('terms', {}))

    if updated > 0:
        lex['meta']['version'] = 'v2.4'
        lex['meta']['generated'] = f'{lex["meta"].get("generated","")}, T0.6 行业分类定义 {datetime.now().strftime("%Y-%m-%d")}'
        lex['meta']['note'] = f'v2.4: 从国民经济行业分类注释补充{updated}条定义 (maturity→termbase)'

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(lex, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

    return updated


# ============================================================
# 主流程
# ============================================================

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("📖 替代数据源: 行业分类定义 + H6 双语对齐")
    print("=" * 60)

    # 1. H6 Bilingual
    print("\n📦 Source 1: H6.json 双语术语对齐")
    h6_items = extract_h6_bilingual()

    # Save H6 bilingual reference
    with open(DATA_DIR / "h6_textile_bilingual.csv", 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hs6_code', 'en_name'])
        writer.writerows(h6_items)
    print(f"   💾 h6_textile_bilingual.csv ({len(h6_items)} 条)")

    # 2. Industry classification
    print(f"\n📦 Source 2: 国民经济行业分类注释")
    industry_defs = extract_industry_defs()

    # 3. Update lexicon
    print(f"\n📝 更新词典...")
    with open(LEXICON_PATH, encoding='utf-8') as f:
        lex = yaml.safe_load(f)

    updated = update_lexicon_with_defs(lex, industry_defs, LEXICON_PATH)
    print(f"   ✅ {updated} 个词条补充了行业分类定义")

    # 4. Export jieba
    import subprocess
    result = subprocess.run(['python', 'scripts/nlp_dict/merge_lexicon.py', '--export-jieba-only'],
                           capture_output=True, text=True, timeout=30)
    # Show last line
    lines = [l for l in result.stdout.split('\n') if l.strip()]
    if lines:
        print(f"   📖 {lines[-1]}")

    print(f"\n{'='*60}")
    print(f"📊 提取摘要")
    print(f"{'='*60}")
    print(f"  H6 双语条目: {len(h6_items)} 条")
    print(f"  行业分类定义: {len(industry_defs)} 条")
    print(f"  词典补充定义: {updated} 条 (v2.4)")


if __name__ == "__main__":
    main()
