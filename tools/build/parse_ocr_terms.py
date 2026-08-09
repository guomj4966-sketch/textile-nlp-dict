"""OCR Term Parser v5 — Working.
Key insight: Don't require '术语和定义' heading. Just parse ALL numbered entries
from the document that have: section number + Chinese name + definition text.
"""
import json, re, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

RESULTS = Path(__file__).parent / 'data' / 'ocr_results'

def fix_ocr_spaces(text):
    """Remove OCR-added spaces between Chinese characters."""
    result = []
    chars = list(text)
    i = 0
    n = len(chars)
    while i < n:
        result.append(chars[i])
        if (i + 2 < n and '一' <= chars[i] <= '鿿' and
            chars[i+1] == ' ' and '一' <= chars[i+2] <= '鿿'):
            i += 1
        i += 1
    return ''.join(result)


def parse_all_terms_from_ocr(text, std_num, desc):
    """Parse ALL numbered term entries from OCR text.
    Works regardless of whether there's a explicit '术语和定义' heading.
    """
    lines = text.split('\n')
    terms = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip blank lines, page headers
        if not line or len(line) < 3:
            i += 1
            continue
        if re.match(r'^(GB/T|ICS|CCS|GB\s*/?\s*T)', line):
            i += 1
            continue
        if re.match(r'^\d{1,2}$', line.strip()):  # standalone page number
            i += 1
            continue

        # Normalize: "3. 22" -> "3.22" (OCR adds spaces in section numbers)
        line_normalized = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', line)

        # ============================================
        # Pattern A: Solo section number on its own line
        #   "2.6" or "3. 22" (OCR space)
        #   "地 弄 花 opening waste"
        # ============================================
        solo_num_m = re.match(r'^(\d+(?:\.\d+){0,3})$', line_normalized)
        if solo_num_m:
            sec_num = solo_num_m.group(1)
            # Skip single-digit section headers (1, 2, 3, 4, 5)
            if '.' not in sec_num and int(sec_num) <= 5:
                i += 1
                continue

            # Next non-blank line
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break

            term_line = lines[i].strip()

            # Parse term line
            term_m = re.match(
                r'^([一-鿿][一-鿿\s·\-（）Ⅰ-Ⅸ]{1,50}?)'  # Chinese term
                r'\s*'                                    # space
                r'([a-zA-ZöüäÖÜÄßⅠ-Ⅸ][a-zA-Z\s\-;,/öüäÖÜÄß·Ⅰ-Ⅸ（）().]{0,80})?'  # optional English
                r'$',
                term_line
            )

            if not term_m:
                # Could be just '术语名' without English
                term_m2 = re.match(r'^([一-鿿][一-鿿\s·\-]{2,30})$', term_line)
                if term_m2:
                    ch_name = term_m2.group(1)
                    en_name = ''
                    i += 1
                else:
                    i += 1
                    continue
            else:
                ch_name = term_m.group(1)
                en_name = (term_m.group(2) or '').strip()
                i += 1

        # ============================================
        # Pattern B: Section number + term on same line
        #   "3.1.9 桑 蚕 彩 色 茧 colored cocoon"
        # ============================================
        else:
            inline_m = re.match(
                r'^(\d+(?:\.\d+){0,3})\s+'          # section number
                r'([一-鿿][一-鿿\s·\-（）Ⅰ-Ⅸ]{1,50}?)' # Chinese term
                r'\s*'                               # space
                r'([a-zA-ZöüäÖÜÄßⅠ-Ⅸ][a-zA-Z\s\-;,/öüäÖÜÄß·Ⅰ-Ⅸ（）().]{0,80})?'  # optional English
                r'$',
                line
            )
            if not inline_m:
                i += 1
                continue

            sec_num = inline_m.group(1)
            if '.' not in sec_num and int(sec_num) <= 5:
                i += 1
                continue

            ch_name = inline_m.group(2)
            en_name = (inline_m.group(3) or '').strip()
            i += 1

        # Clean Chinese name
        ch_name_clean = fix_ocr_spaces(ch_name.strip())

        # Quality filter: must be 2+ CJK chars, not a section header
        cjk_in_name = sum(1 for c in ch_name_clean if '一' <= c <= '鿿')
        if cjk_in_name < 2:
            continue

        # Skip section headers and non-terms
        skip_set = {
            '术语和定义', '规范性引用文件', '术语', '定义', '分类',
            '要求', '方法', '通则', '原则', '试验方法',
            '检测', '测定', '原理', '仪器', '步骤', '结果', '结论',
            '前言', '范围', '附录', '参考文献', '类别',
            '评定', '试验', '检验', '判定', '符号', '代号',
            '通用', '总则', '概述', '基本要求',
            '准备', '操作', '计算', '报告', '说明',
            '试样', '设备', '装置', '试剂', '材料', '样品',
            '半制品', '产品分类', '产品品种', '品种',
            '文件', '规定', '标准', '规范',
            '原料', '精练', '练染整', '成品',
        }
        if ch_name_clean in skip_set:
            continue
        # Skip if term looks like a category heading (no substantive meaning alone)
        if len(ch_name_clean) <= 2 and not en_name:
            continue

        # ============================================
        # Collect definition text
        # ============================================
        def_lines = []

        while i < len(lines):
            def_line = lines[i].strip()

            if not def_line:
                i += 1
                continue

            # Next solo number (stop)
            if re.match(r'^\d+(?:\.\d+){0,3}$', def_line):
                break

            # Next inline term (stop)
            if re.match(r'^\d+(?:\.\d+){0,3}\s+[一-鿿]', def_line):
                break

            # Page number / header
            if re.match(r'^(GB/T|ICS|CCS|犌犅)', def_line):
                i += 1
                continue

            # Major section boundary
            if re.match(r'^\s*[4-9][\.\s、．]', def_line):
                break

            # Another term pattern (stop)
            if re.match(r'^[一-鿿][一-鿿\s]{1,30}\s+[a-zA-Z]{3,}', def_line):
                break

            def_lines.append(def_line)
            i += 1

        def_text = ''.join(def_lines)
        def_text = fix_ocr_spaces(def_text)

        # Quality: definition must have enough CJK
        cjk = sum(1 for c in def_text if '一' <= c <= '鿿')
        if cjk >= 6:
            terms.append({
                'term': ch_name_clean,
                'english': en_name,
                'definition': def_text[:500],
                'source': std_num,
                'category': desc,
            })

    return terms


# ============================================================
# PROCESS ALL STANDARDS
# ============================================================
PDFS = [
    ('GBT+15557-2008.pdf', '服装术语', 'GB/T 15557-2008'),
    ('GBT+26380-2022.pdf', '丝绸术语', 'GB/T 26380-2022'),
    ('GBT+4146.2-2017.pdf', '化学纤维术语', 'GB/T 4146.2-2017'),
    ('GBT+5705-2018.pdf', '棉纤维术语', 'GB/T 5705-2018'),
    ('GBT+38111-2019.pdf', '静电性能术语', 'GB/T 38111-2019'),
    ('GBT+38923-2020.pdf', '废旧纺织品分类', 'GB/T 38923-2020'),
    ('GBT+42693-2023.pdf', '产业用纺织品术语', 'GB/T 42693-2023'),
    ('GBT+46947-2025.pdf', '可持续性术语', 'GB/T 46947-2025'),
    ('GBT+47198-2026.pdf', '生物基纤维术语', 'GB/T 47198-2026'),
    ('GBT+47771-2026.pdf', '功能性纺织品术语', 'GB/T 47771-2026'),
    ('GBT+30420.4-2025.pdf', '纤维含量标识', 'GB/T 30420.4-2025'),
    ('GBT+30558-2025.pdf', '再生纤维素纤维', 'GB/T 30558-2025'),
    ('GBT+33278-2016.pdf', '色牢度试验', 'GB/T 33278-2016'),
    ('GBT+38136-2019.pdf', '遮热性能', 'GB/T 38136-2019'),
    ('GBT+6002.17-2025.pdf', '纺织机械术语', 'GB/T 6002.17-2025'),
]

all_terms = []
stats = {}

print('=== OCR TERM PARSING v5 (solo number + inline number) ===')
print()

for fn, desc, std_num in PDFS:
    stem = Path(fn).stem
    txt_files = sorted(RESULTS.glob(f'{stem}_p*.txt'))
    if not txt_files:
        continue

    full_text = ''
    for tf in txt_files:
        with open(tf, 'r', encoding='utf-8') as f:
            full_text += f.read() + '\n\n'

    terms = parse_all_terms_from_ocr(full_text, std_num, desc)

    # Dedup within standard
    seen = set()
    unique = []
    for t in terms:
        if t['term'] not in seen:
            seen.add(t['term'])
            unique.append(t)

    stats[std_num] = len(unique)
    all_terms.extend(unique)

    sample = [t['term'] for t in unique[:5]]
    sep = ' / '
    print(f'  [{std_num:20s}] {desc:12s}: {len(unique):>3d} terms | {sep.join(sample)}')

    # Save
    out_fn = std_num.replace('/', '-').replace(' ', '_') + '_terms.json'
    with open(RESULTS / out_fn, 'w', encoding='utf-8') as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

total = sum(stats.values())
unique_total = len(set(t['term'] for t in all_terms))
zero_stds = [k for k, v in stats.items() if v == 0]
print()
print(f'SUMMARY: {total} total, {unique_total} unique terms from {sum(1 for v in stats.values() if v > 0)}/15 standards')
if zero_stds:
    print(f'Zero-yield standards ({len(zero_stds)}): {zero_stds}')

with open(RESULTS / 'all_extracted_terms.json', 'w', encoding='utf-8') as f:
    json.dump(all_terms, f, ensure_ascii=False, indent=2)
print(f'Saved: all_extracted_terms.json')
