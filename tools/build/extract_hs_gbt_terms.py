"""Step 2-3 v2: Better sub-term extraction + stricter filtering for HS Code 2026."""
import sys, io, re, yaml
import difflib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

BASE = Path(r'D:\GHJ\OneDrive\AI-Textile-policy-brief')
LEXICON = BASE / 'data' / 'lexicon_v2.yaml'
HS_RAW = BASE / 'scripts' / 'nlp_dict' / 'data' / 'hs2026_raw_terms.txt'
HS_OUT = BASE / 'scripts' / 'nlp_dict' / 'data' / 'hs2026_filtered_terms.txt'
GBT_OUT = BASE / 'scripts' / 'nlp_dict' / 'data' / 'gbt38923_terms.txt'

# ==============================
# Load existing lexicon
# ==============================
with open(LEXICON, 'r', encoding='utf-8') as f:
    lex = yaml.safe_load(f)

existing_chinese = set()
for lname, ldata in lex['layers'].items():
    if 'terms' not in ldata:
        continue
    for cat_name, term_list in ldata['terms'].items():
        for t in term_list:
            if isinstance(t, str):
                existing_chinese.add(t)
            elif isinstance(t, dict):
                existing_chinese.add(t.get('term', ''))

print(f'Existing terms: {len(existing_chinese)}')

# ==============================
# Source 1: HS Code 2026 — Better sub-term extraction
# ==============================
with open(HS_RAW, 'r', encoding='utf-8') as f:
    raw_terms = [l.strip() for l in f if l.strip()]

# Identify the KEY textile sub-terms from HS names
# Strategy: strip modifiers, keep core material + product nouns
# Pattern: [material]制/用 [product type] [qualifiers]

def extract_clean_terms(name):
    """Extract clean 2-6 character textile-relevant sub-terms from a HS name."""
    results = set()

    # Remove parenthetical content, measurements
    name = re.sub(r'[（(][^)）]*[)）]', '', name)
    name = re.sub(r'[>=<>=<]\s*\d+[.\d]*\s*[%%]?', '', name)
    name = re.sub(r'\d+[.\d]*\s*(?:分特|转/米|千克|分米|厘米|毫米|克)', '', name)
    name = re.sub(r'每平方米[^,,\s]+', '', name)

    # Remove non-textile chapters (Ch 64-67 footwear/headgear/umbrellas, Ch 66 walking-sticks)
    # These are in the source but aren't textile terms per se

    # Split on key separators
    name = re.sub(r'^(未列名|其他|其他非|其他未|其他不)', '', name)
    name = re.sub(r'(供零售用|非供零售用|非零售用|仿手工|非手工|手工|非绣制|绣制)', '', name)
    name = re.sub(r'(未漂白|漂白|染色|印花|色织|色)|(及类似品|及类似物)', '', name)
    name = re.sub(r'(完全|部分|全部)', '', name)
    name = name.strip('  , .  , ').strip('，。')

    if len(name) <= 2 or len(name) > 25:
        return results

    # Strategy A: Extract core product category (last segment)
    # e.g. "丝及绢丝制男式西服套装" -> keep "男式西服套装"
    # Better: extract BOTH material and product as separate terms

    # Strategy B: Extract textile material terms (prefix before 制/用)
    material_match = re.match(r'(.+?)(?:制|用|纺制的)', name)
    if material_match:
        material = material_match.group(1).strip()
        if 2 <= len(material) <= 10 and material not in ('按', '各种'):
            # Further split material
            for m in re.split(r'[及或和、]', material):
                m = m.strip()
                if 2 <= len(m) <= 8:
                    results.add(m)

    # Strategy C: Extract product/garment type (suffix after 制/用)
    product_match = re.search(r'(?:制|用)(.+)$', name)
    if product_match:
        product = product_match.group(1).strip()
        if 2 <= len(product) <= 12:
            results.add(product)

    # Strategy D: For names without 制/用, the whole thing may be a term
    if '制' not in name and '用' not in name:
        if 3 <= len(name) <= 12:
            results.add(name)

    # Strategy E: Extract specific textile processes/products by pattern
    textile_patterns = [
        r'(?:针织|梭织|机织|编织|钩编|手工编|非织造)(?:物|布|品)?',
        r'(?:男式|女式|男|女|婴儿)(?:\w{2,8})',
        r'(?:\w{2,6})(?:服装|内衣|睡衣|外衣|夹克|裤子|衬衫|裙子|套装)',
        r'(?:\w{2,4})(?:纱|线|布|织物|面料|纤维|丝|绒)',
        r'(?:\w{2,6})(?:地毯|毛巾|床单|被套|枕套|窗帘|桌布)',
        r'(?:\w{2,6})(?:花边|蕾丝|刺绣|标签|徽章|装饰)',
    ]
    for pat in textile_patterns:
        for m in re.finditer(pat, name):
            results.add(m.group())

    return {r for r in results if 2 <= len(r) <= 12 and not re.match(r'^[计第含]', r)}

# Extract sub-terms from all HS names
all_sub_terms = set()
for name in raw_terms:
    subs = extract_clean_terms(name)
    all_sub_terms.update(subs)

print(f'\n=== HS Code 2026 v2 ===')
print(f'Sub-terms extracted: {len(all_sub_terms)}')

# Show distribution
len_dist = {}
for t in all_sub_terms:
    l = len(t)
    len_dist[l] = len_dist.get(l, 0) + 1
print(f'Length distribution:')
for l in sorted(len_dist.keys())[:12]:
    print(f'  {l} chars: {len_dist.get(l, 0)} terms')

# Round 2: Remove noise
noise_words = {'除外', '包括', '其他', '其他不', '其他未', '含有', '非为', '标注应'}
filtered = {t for t in all_sub_terms if t not in noise_words}
print(f'Round 2 (noise): {len(filtered)} terms')

# Round 3: Dedup vs existing
new_terms_raw = filtered - existing_chinese
deduped = set()
for t in sorted(new_terms_raw):
    is_dup = False
    for e in existing_chinese:
        if len(t) >= 3 and len(e) >= 3:
            ratio = difflib.SequenceMatcher(None, t, e).ratio()
            if ratio > 0.80:
                is_dup = True
                break
    if not is_dup:
        deduped.add(t)
print(f'Round 3 (vs existing + fuzzy): {len(deduped)} new terms')
print(f'  Removed {len(new_terms_raw) - len(deduped)} near-duplicates')

# Round 4: Textile relevance scoring
textile_materials = [
    '丝', '棉', '毛', '麻', '纤维', '化纤', '聚酯', '涤纶', '锦纶', '腈纶', '氨纶',
    '维纶', '丙纶', '粘胶', '莫代尔', '莱赛尔', '醋酸', '桑蚕', '柞蚕', '绢丝',
    '羊绒', '羊毛', '驼毛', '兔毛', '蚕丝', '黄麻', '亚麻', '苎麻', '大麻',
]
textile_products = [
    '纱', '线', '布', '织物', '面料', '服装', '内衣', '睡衣', '衬衫', '裤子', '裙子',
    '套装', '夹克', '大衣', '袜', '手套', '帽', '围巾', '领带', '披肩', '地毯', '毛巾',
    '床单', '被套', '枕套', '窗帘', '桌布', '毯', '袋', '包', '箱', '绳', '索', '缆',
    '网', '帐篷', '帆布', '鞋', '花边', '蕾丝', '刺绣', '纽扣', '拉链', '羽绒',
]
textile_process = [
    '针织', '梭织', '机织', '编织', '钩编', '非织造', '无纺', '染色', '印花', '漂白',
    '涂层', '复合', '混纺', '交织', '捻', '股', '单纱', '股线', '缝纫', '绣',
]

scored = []
for t in deduped:
    score = 0
    for kw in textile_materials:
        if kw in t:
            score += 3
    for kw in textile_products:
        if kw in t:
            score += 3
    for kw in textile_process:
        if kw in t:
            score += 2
    # Bonus: well-formed term length
    if 4 <= len(t) <= 8:
        score += 1
    # Penalty: still has generic remnants
    if re.search(r'(?:未列名|按第|于第|包括|除外|本章|其他不)', t):
        score -= 5
    scored.append((t, score))

# Keep score >= 4
quality = {t for t, s in scored if s >= 4}
low_q = [(t, s) for t, s in scored if s < 4]
print(f'Round 4 (quality >= 4): {len(quality)} high-quality terms')
print(f'  Discarded: {len(low_q)} terms')

# Show sample
print(f'\nSample new terms:')
for t in sorted(quality)[:35]:
    print(f'  {t}')
print(f'  ...')
for t in sorted(quality)[-20:]:
    print(f'  {t}')

# Save
with open(HS_OUT, 'w', encoding='utf-8') as f:
    for t in sorted(quality):
        f.write(t + '\n')
print(f'\nSaved {len(quality)} terms to {HS_OUT}')

# ==============================
# Source 2: GBT 38923-2020
# ==============================
gbt_terms = [
    '废旧纺织品',
    '消费前废旧纺织品',
    '消费后废旧纺织品',
    '棉类废旧纺织品',
    '毛类废旧纺织品',
    '聚酯类废旧纺织品',
    '聚酰胺类废旧纺织品',
    '聚丙烯腈类废旧纺织品',
    '其他类废旧纺织品',
    '复合类废旧纺织品',
    '废旧纺织品不合格品',
    '废旧纺织品不合格品含量',
]

gbt_new = set()
for t in gbt_terms:
    if t not in existing_chinese:
        dup = any(difflib.SequenceMatcher(None, t, e).ratio() > 0.85
                  for e in existing_chinese if len(e) >= 3)
        if not dup:
            gbt_new.add(t)

print(f'\n=== GBT 38923-2020 ===')
print(f'New terms: {len(gbt_new)}')
for t in sorted(gbt_new):
    print(f'  {t}')

with open(GBT_OUT, 'w', encoding='utf-8') as f:
    for t in sorted(gbt_new):
        f.write(t + '\n')

# ==============================
# Summary
# ==============================
print(f'\n=== SUMMARY ===')
print(f'HS Code 2026 new terms: {len(quality)}')
print(f'GBT 38923-2020 new terms: {len(gbt_new)}')
print(f'TOTAL: {len(quality) + len(gbt_new)}')
print(f'Existing: {len(existing_chinese)}')
print(f'Estimated new total: {len(existing_chinese) + len(quality) + len(gbt_new)}')
