"""Step 5: Merge new terms into lexicon_v2.yaml.
Adds HS Code 2026 terms (layer_3_textile_chain) and GBT 38923-2020 terms (layer_5_cross_domain).
"""
import sys, io, yaml, copy
from collections import OrderedDict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from datetime import datetime

BASE = Path(r'D:\GHJ\OneDrive\AI-Textile-policy-brief')
LEXICON = BASE / 'data' / 'lexicon_v2.yaml'
HS_TERMS = BASE / 'scripts' / 'nlp_dict' / 'data' / 'hs2026_filtered_terms.txt'
GBT_TERMS = BASE / 'scripts' / 'nlp_dict' / 'data' / 'gbt38923_terms.txt'
JIEBA_DICT = BASE / 'data' / 'jieba_dict.txt'
JIEBA_DICT_ALT = BASE / 'scripts' / 'nlp_dict' / 'data' / 'jieba_dict.txt'

# ==============================
# Load current lexicon
# ==============================
with open(LEXICON, 'r', encoding='utf-8') as f:
    lex = yaml.safe_load(f)

# Count existing terms properly
def count_terms(lex_data):
    count = 0
    terms_set = set()
    for lname, ldata in lex_data['layers'].items():
        if 'terms' not in ldata:
            continue
        for cat_name, term_list in ldata['terms'].items():
            for t in term_list:
                if isinstance(t, str):
                    terms_set.add(t)
                    count += 1
                elif isinstance(t, dict):
                    terms_set.add(t.get('term', ''))
                    count += 1
    return count, len(terms_set)

old_count, old_unique = count_terms(lex)
print(f'Old lexicon: {old_count} entries, {old_unique} unique terms')

# ==============================
# Load new terms
# ==============================
with open(HS_TERMS, 'r', encoding='utf-8') as f:
    hs_terms = [l.strip() for l in f if l.strip()]
with open(GBT_TERMS, 'r', encoding='utf-8') as f:
    gbt_terms = [l.strip() for l in f if l.strip()]

print(f'HS Code 2026 terms: {len(hs_terms)}')
print(f'GBT 38923-2020 terms: {len(gbt_terms)}')

# ==============================
# Get existing terms for dedup
# ==============================
existing_terms = set()
layer3_terms = lex['layers']['layer_3_textile_chain']['terms']

for cat_name, term_list in layer3_terms.items():
    for t in term_list:
        if isinstance(t, str):
            existing_terms.add(t)
        elif isinstance(t, dict):
            existing_terms.add(t.get('term', ''))

# Also collect from other layers for cross-layer dedup
for lname, ldata in lex['layers'].items():
    if 'terms' not in ldata:
        continue
    for cat_name, term_list in ldata['terms'].items():
        for t in term_list:
            if isinstance(t, str):
                existing_terms.add(t)
            elif isinstance(t, dict):
                existing_terms.add(t.get('term', ''))

# ==============================
# Classify HS terms into layer_3 sub-categories
# ==============================
# Categories in layer_3_textile_chain (from existing lexicon):
# 1_原料端 / 天然纤维, 1_原料端 / 再生/循环纤维, ...
# We need to map HS terms to the right sub-categories
# Most HS terms are products (categories 5-11) or materials (1-4)

MATERIAL_KEYWORDS = {
    '1_原料端 / 天然纤维': ['棉', '毛', '麻', '丝', '蚕', '羊绒', '羊', '驼', '兔', '羽绒', '羽毛',
                      '亚麻', '黄麻', '苎麻', '大麻', '椰壳', '蕉麻', '龙舌兰'],
    '1_原料端 / 合成纤维': ['聚酯', '涤纶', '锦纶', '尼龙', '腈纶', '氨纶', '维纶', '丙纶',
                      '聚酰胺', '聚丙烯腈', '聚丙烯', '弹性纤维', '合纤'],
    '1_原料端 / 再生纤维素纤维': ['粘胶', '莫代尔', '莱赛尔', '醋酸', '铜氨', '再生纤维', '人造纤维'],
    '1_原料端 / 再生/循环纤维': ['废', '再生', '回收', '循环'],
}

PRODUCT_KEYWORDS = {
    '5_终端品类 / 服装': ['服装', '衣', '裤', '裙', '衬衫', '夹克', '西服', '套装', '便服',
                    '睡衣', '泳装', '内衣', '袜', '手套', '帽', '领带', '围巾', '披肩',
                    '运动服', '工作服', '防护服', '校服', '制服', '婴儿', '童装',
                    'T恤', '背心', '马甲', '外套', '大衣', '斗篷', '浴衣', '晨衣'],
    '5_终端品类 / 家纺': ['床', '被', '枕', '毯', '巾', '帘', '垫', '褥', '桌布', '餐具',
                    '帷幔', '挂毯', '蚊帐', '睡袋', '毛巾被', '床罩', '靠垫'],
    '5_终端品类 / 产业用纺织品': ['帐篷', '帆布', '绳', '索', '缆', '网', '过滤布', '传送带',
                          '土工', '防水', '阻燃', '遮阳', '油苫布', '天篷'],
}

FABRIC_KEYWORDS = {
    '3_织造端 / 梭织': ['梭织', '机织', '机织物', '府绸', '卡其', '牛仔', '帆布', '灯芯绒'],
    '3_织造端 / 针织': ['针织', '针织物', '经编', '纬编', '钩编', '钩编织物'],
    '3_织造端 / 非织造': ['非织造', '无纺', '毡', '絮', '衬'],
    '3_织造端 / 编织与其他': ['编织', '编带', '花边', '蕾丝', '刺绣', '标签', '徽章',
                       '流苏', '绒球', '饰带', '网眼', '薄纱'],
}

YARN_KEYWORDS = {
    '2_纺纱端 / 纱线品类': ['纱', '线', '丝', '单纱', '股线', '缆线', '缝纫线', '绣花线',
                      '弹力丝', '变形丝', '高强力纱', '扁条'],
}

# Classify each HS term
classified = {}
unclassified = []

for term in hs_terms:
    assigned = False
    for cat, kws in {**MATERIAL_KEYWORDS, **PRODUCT_KEYWORDS, **FABRIC_KEYWORDS, **YARN_KEYWORDS}.items():
        if any(kw in term for kw in kws):
            if cat not in classified:
                classified[cat] = []
            classified[cat].append(term)
            assigned = True
            break
    if not assigned:
        unclassified.append(term)

print(f'\nHS term classification:')
for cat in sorted(classified.keys()):
    print(f'  {cat}: {len(classified[cat])} terms')
print(f'  Unclassified: {len(unclassified)} terms')

# Show unclassified to check for miscategorization
print(f'\nUnclassified terms ({len(unclassified)}):')
for t in unclassified[:30]:
    print(f'  {t}')
if len(unclassified) > 30:
    print(f'  ... and {len(unclassified) - 30} more')

# ==============================
# Merge into lexicon
# ==============================
# Make a deep copy of the old lex
new_lex = copy.deepcopy(lex)

# Update layer_3 with new terms
# Ensure all category keys exist in the new layer_3 terms
for cat in sorted(classified.keys()):
    if cat not in new_lex['layers']['layer_3_textile_chain']['terms']:
        new_lex['layers']['layer_3_textile_chain']['terms'][cat] = []

    existing_in_cat = set(new_lex['layers']['layer_3_textile_chain']['terms'][cat])
    new_for_cat = [t for t in classified[cat] if t not in existing_in_cat]
    new_lex['layers']['layer_3_textile_chain']['terms'][cat].extend(new_for_cat)
    print(f'Merged {len(new_for_cat)} new terms into {cat}')

# Add unclassified terms to an "HS Code 2026" sub-category under layer_3
if unclassified:
    uc_cat = 'HS Code 2026 / 未分类纺织商品'
    # Filter ones that are already in some category
    all_classified = set()
    for cat_terms in classified.values():
        all_classified.update(cat_terms)
    truly_unclassified = [t for t in unclassified if t not in all_classified]

    if truly_unclassified:
        new_lex['layers']['layer_3_textile_chain']['terms'][uc_cat] = truly_unclassified
        print(f'Added {len(truly_unclassified)} terms to {uc_cat}')

# Add GBT 38923 terms to layer_5_cross_domain (green/recycling)
gbt_cat = '绿色低碳 / 废旧纺织品分类'
new_lex['layers']['layer_5_cross_domain']['terms'][gbt_cat] = gbt_terms
print(f'Added {len(gbt_terms)} GBT terms to layer_5_cross_domain/{gbt_cat}')

# ==============================
# Update meta
# ==============================
total, unique = count_terms(new_lex)
new_lex['meta']['version'] = 'v2.7'
new_lex['meta']['generated'] = f'{datetime.now().strftime("%Y-%m-%d")} (新增 HS Code 2026 + GBT 38923-2020)'
new_lex['meta']['total_terms'] = unique
new_lex['meta']['note'] = (
    f'v2.7: 从2026版HS编码表(纺织Ch50-63)提取{len(hs_terms)}个商品术语, '
    f'从GBT 38923-2020(废旧纺织品分类与代码)提取{len(gbt_terms)}个分类术语'
)

# ==============================
# Save
# ==============================
# Custom YAML dumper to preserve format
class CustomDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(CustomDumper, self).increase_indent(flow, False)

def str_representer(dumper, data):
    """Use literal block scalar for multi-line strings."""
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

# Don't override default string representation, just use standard dump
# with allow_unicode and default_flow_style=False
with open(LEXICON, 'w', encoding='utf-8') as f:
    yaml.dump(new_lex, f, allow_unicode=True, default_flow_style=False,
              sort_keys=False, width=200, Dumper=CustomDumper)

print(f'\n=== LEXICON SAVED ===')
print(f'Old: {old_unique} unique terms')
print(f'New: {unique} unique terms')
print(f'Added: {unique - old_unique} terms')
print(f'Version: {new_lex["meta"]["version"]}')
