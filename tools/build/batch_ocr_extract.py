"""
Batch OCR Pipeline: Extract term+definition pairs from GB/T textile standard PDFs.

Strategy (optimized after openstd test):
1. Try pdf.get_text() first — some newer PDFs (2025+) are UTF-8 readable
2. Fall back to OCR (render page → Tesseract chi_sim) — all PDFs
3. Parse extracted text for term/definition patterns
4. Dedup vs existing lexicon
5. Output structured results

Time: ~15-20 minutes for ~400 pages across 16 PDFs
"""
import fitz, sys, io, os, re, subprocess, json, yaml
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# CONFIG
# ============================================================
TESSERACT = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSDATA = r'C:\Users\ghj\AppData\Local\Temp'
# PDF_DIR: auto-detect from OneDrive (cross-machine compatible)
_ONEDRIVE_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
PDF_DIR = _ONEDRIVE_ROOT / '分类和术语标准规范'
OUT_DIR = Path(__file__).parent / 'data' / 'ocr_results'
MATRIX = fitz.Matrix(300/72, 300/72)

# Only the textile definition-relevant PDFs
# Exclude: 44870 (carbon fiber equipment, not general textile), 50514 (factory design)
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
    # GBT+44870-2024 is carbon fiber equipment, not textile terminology
]

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CORE FUNCTIONS
# ============================================================

def ocr_page(page_img_path, output_txt_path):
    """Run Tesseract OCR on a single page image. Returns True on success."""
    try:
        subprocess.run([
            TESSERACT, '--tessdata-dir', TESSDATA,
            '-l', 'chi_sim',
            str(page_img_path),
            str(output_txt_path).replace('.txt', '')  # tesseract adds .txt
        ], capture_output=True, timeout=30, check=True)
        return True
    except subprocess.TimeoutExpired:
        print(f'    TIMEOUT: {page_img_path.name}')
        return False
    except subprocess.CalledProcessError as e:
        print(f'    ERROR: {page_img_path.name}: {e}')
        return False


def render_page_to_image(doc, page_idx, out_path):
    """Render a PDF page to 300 DPI PNG."""
    page = doc[page_idx]
    pix = page.get_pixmap(matrix=MATRIX)
    pix.save(str(out_path))
    return out_path


def extract_text_from_pdf(pdf_path, use_ocr=True):
    """Get text from a PDF. If use_ocr=True, render and OCR every page.
    If use_ocr=False, try pdf.get_text() first (useful for newer UTF-8 PDFs)."""
    doc = fitz.open(str(pdf_path))

    # First try direct extraction
    direct_text = ''
    for i in range(len(doc)):
        direct_text += doc[i].get_text()

    # Check if direct text is usable (>500 readable CJK chars)
    readable_cjk = sum(1 for c in direct_text if 0x4e00 <= ord(c) <= 0x9fff)
    if readable_cjk > 500 and not use_ocr:
        doc.close()
        return direct_text

    # Fall back to OCR
    ocr_text = ''
    for i in range(len(doc)):
        img_path = OUT_DIR / f'{pdf_path.stem}_p{i+1:02d}.png'
        txt_path = OUT_DIR / f'{pdf_path.stem}_p{i+1:02d}.txt'

        if not img_path.exists():
            render_page_to_image(doc, i, img_path)

        if not txt_path.exists():
            success = ocr_page(img_path, txt_path)
            if not success:
                continue

        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                ocr_text += f.read() + '\n'

    doc.close()
    return ocr_text


def parse_term_definitions(text, std_num, desc):
    """Parse OCR/extracted text to find term-definition pairs.

    GB/T standard structure for 术语和定义 section:
    X.Y  术语名 [英文名]
    术语定义文本。

    Patterns vary but typically:
    - Numbered: 3.1.1  术语名  EnglishName
    - Definition follows (often next line or paragraph)
    """
    terms = []

    # Pattern 1: Numbered term with possible English name
    # e.g. "3.1  棉网  web" or "3.1.1  棉网  cotton web"
    # Definition follows until next numbered item or empty line

    # Pattern 2: Term bold in table (older standards)
    # e.g. "开清棉  opening and cleaning"

    # First, find the 术语和定义 section
    term_section_start = -1
    for marker in ['术语和定义', '3  术语', '3 术语', '3、术语', '3．术语']:
        idx = text.find(marker)
        if idx >= 0:
            term_section_start = idx
            break

    if term_section_start < 0:
        # Try to find by section number pattern
        m = re.search(r'\n3[\.\s、．]+术语', text)
        if m:
            term_section_start = m.start()

    if term_section_start < 0:
        return terms  # No term section found

    section_text = text[term_section_start:term_section_start + 50000]  # Next 50K chars

    # Strategy A: Find numbered term entries with English names
    # Pattern: X.Y.Z  ChineseName EnglishName(s)
    # This is the most common pattern in GB/T standards
    pattern_a = re.findall(
        r'(?:^|\n)\s*(\d+(?:\.\d+)+)\s+'
        r'([^\n]{2,60}?)\s*'  # Chinese term
        r'(?:（[^）]*）)?\s*'  # Optional Chinese parenthetical
        r'(?:([a-zA-Z][a-zA-Z\s\-;,/]{3,60}))?\s*'  # English name (optional)
        r'(?:\n|$)',
        section_text, re.MULTILINE
    )

    for num, ch_name, en_name in pattern_a:
        ch_name = ch_name.strip()
        en_name = en_name.strip() if en_name else ''

        # Filter: must contain Chinese chars
        if not re.search(r'[\u4e00-\u9fff]', ch_name):
            continue

        # Filter: skip if it's a section header (not a term)
        if any(kw in ch_name for kw in ['术语', '定义', '分类', '要求', '方法',
                                          '通则', '原则', '试验', '检测', '测定',
                                          '原理', '仪器', '步骤', '结果', '通则']):
            continue

        # Find the definition text (after the term, before next term or section)
        idx = section_text.find(num + ' ' + ch_name)
        if idx < 0:
            idx = section_text.find(f'{num}  {ch_name}')
        if idx < 0:
            continue

        # The definition starts after the term line
        def_start = section_text.find('\n', idx) + 1
        if def_start <= 0:
            continue

        # Next section or term boundary
        next_boundary = len(section_text)
        # Next numbered item
        next_num = re.search(r'\n\s*\d+(?:\.\d+)+\s+', section_text[def_start:def_start+2000])
        if next_num:
            next_boundary = def_start + next_num.start()

        # Or next major section (4, 5, 6...)
        next_section = re.search(r'\n\s*[4-9]\s', section_text[def_start:def_start+2000])
        if next_section and next_section.start() < (next_boundary - def_start):
            next_boundary = def_start + next_section.start()

        def_text = section_text[def_start:next_boundary].strip()
        # Clean: remove page numbers, excessive whitespace
        def_text = re.sub(r'\n\s*\d+\s*\n', '\n', def_text)  # standalone page numbers
        def_text = re.sub(r'\s+', ' ', def_text)
        def_text = re.sub(r'GB/T \d+[\.\d]*—?\d*', '', def_text)

        if len(def_text) >= 10 and len(def_text) <= 500:
            terms.append({
                'term': ch_name,
                'english': en_name,
                'definition': def_text,
                'source': std_num,
                'category': desc,
            })

    # Strategy B: Table-structured terms — for older standards like GB/T 15557-2008
    # These often have no numbering; terms are in a list with English equivalents
    if len(terms) < 5:
        # Look for patterns like: 中文名  EnglishName  定义
        # Often separated by significant whitespace or formatting
        pattern_b = re.findall(
            r'(?:^|\n)\s*([\u4e00-\u9fff][\u4e00-\u9fff（）、\s]{1,30})\s+'
            r'([a-zA-Z][a-zA-Z\s\-;,/（）]{2,50})\s*\n'
            r'([^\n]{15,300})',
            section_text, re.MULTILINE
        )
        for ch_name, en_name, def_text in pattern_b:
            ch_name = ch_name.strip()
            if len(ch_name) >= 2 and len(ch_name) <= 20:
                terms.append({
                    'term': ch_name,
                    'english': en_name.strip(),
                    'definition': def_text.strip(),
                    'source': std_num,
                    'category': desc,
                })

    return terms


# ============================================================
# MAIN — Process each PDF
# ============================================================

print('=' * 60)
print('BATCH OCR: 16 GB/T Textile Standard PDFs')
print('=' * 60)
print()

all_terms = []
stats = {}

for fn, desc, std_num in PDFS:
    pdf_path = PDF_DIR / fn
    if not pdf_path.exists():
        print(f'[{std_num}] SKIP: File not found')
        continue

    print(f'[{std_num}] {desc}... ', end='', flush=True)

    # Extract text (all PDFs need OCR)
    text = extract_text_from_pdf(pdf_path, use_ocr=True)

    # Parse terms
    terms = parse_term_definitions(text, std_num, desc)

    # Dedup within this standard
    seen = set()
    unique_terms = []
    for t in terms:
        if t['term'] not in seen:
            seen.add(t['term'])
            unique_terms.append(t)

    stats[std_num] = {'pages': len(text.split('\n')), 'terms': len(unique_terms)}
    all_terms.extend(unique_terms)

    print(f'{len(unique_terms)} terms')

    # Save per-standard results
    result_path = OUT_DIR / f'{std_num.replace("/", "-").replace(" ", "_")}_terms.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(unique_terms, f, ensure_ascii=False, indent=2)

print()
print('=' * 60)
print('RESULTS SUMMARY')
print('=' * 60)

total = 0
for std_num, s in stats.items():
    print(f'  {std_num:20s}: {s["terms"]:>4d} terms')
    total += s['terms']

print(f'  {"TOTAL":20s}: {total:>4d} terms')

# Save combined results
combined_path = OUT_DIR / 'all_extracted_terms.json'
with open(combined_path, 'w', encoding='utf-8') as f:
    json.dump(all_terms, f, ensure_ascii=False, indent=2)

print(f'\nSaved: {combined_path}')
print(f'Total unique terms (cross-standard): {len(set(t["term"] for t in all_terms))}')
