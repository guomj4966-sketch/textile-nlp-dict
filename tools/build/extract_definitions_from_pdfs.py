"""
从 GB/T 纺织标准 PDF 提取术语和定义 —— v3。

核心策略（两遍扫描）:
  第一遍：识别所有 编号+术语 位置
  第二遍：在每个条目的两个编号之间收集定义文本

v3 改进:
  1. 断行编号合并: 3.\n1.\n14 → 3.1.14
  2. 全角数字归一化: 全角→半角
  3. PDF 四类分类: UTF-8 / 全角拉丁 / 乱码(OCR) / 纯图片
  4. 补充 15557 等 PDF

运行:
    PYTHONIOENCODING=utf-8 python scripts/nlp_dict/extract_definitions_from_pdfs.py
    PYTHONIOENCODING=utf-8 python scripts/nlp_dict/extract_definitions_from_pdfs.py --ocr
"""

import sys, io, re, csv, fitz, argparse, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import yaml

DATA_DIR = Path(__file__).parent / "data"
LEXICON_PATH = DATA_DIR / "lexicon_v2.yaml"
_ONEDRIVE = Path(__file__).parent.parent.parent.parent
STD_DIR = _ONEDRIVE / "分类和术语标准规范"

# ── PDF 分类清单 ──

GROUP_UTF8 = [
    ("GBT+30558-2025.pdf",        "产业用纺织品"),
    ("GBT+30420.4-2025.pdf",      "缝制机械"),
    ("GBT+46947-2025.pdf",        "棉纤维"),
    ("GBT+47198-2026.pdf",        "针织横机"),
    ("GBT+47771-2026.pdf",        "聚丙烯腈纤维生产装备"),
    ("GBT+44870-2024.pdf",        "纤维碳化生产装备"),
    ("GBT+38136-2019.pdf",        "化学纤维分类"),
    ("GBT+33278-2016.pdf",        "粘扣带"),
    ("GBT+6002.17-2025.pdf",      "环锭捻线机"),
]

GROUP_FULLWIDTH = [
    ("GBT+5705-2018.pdf",         "棉纺织产品"),
    ("GBT+26380-2022.pdf",        "丝绸"),
    ("GBT+4146.2-2017.pdf",       "化学纤维产品"),
    ("GBT+38111-2019.pdf",        "玄武岩纤维"),
]

GROUP_GARBLED = [
    ("GBT+15557-2008.pdf",        "服装"),  # 自定义字体编码编号
    ("GBT+42693-2023.pdf",        "应急产业用纺织品"),
    ("GB 50514-2009 非织造布工厂设计规范.pdf", "非织造布工厂"),
]

GROUP_IMAGE_ONLY = [
    ("GBT+44870-2024(EN).pdf",    "纤维碳化装备(英文版)"),
]

GROUP_DUP = [
    ("GBT+30558-2025 (1).pdf",    "产业用纺织品(重复)"),
]

ALL_GROUPS = [
    ("🟢 UTF-8",    GROUP_UTF8),
    ("🟡 全角拉丁",  GROUP_FULLWIDTH),
    ("🔴 乱码(OCR)", GROUP_GARBLED),
    ("🟠 纯图片",    GROUP_IMAGE_ONLY),
    ("⏭️ 重复",     GROUP_DUP),
]


# ── 字符工具 ──

def _fw2hw(text):
    """全角→半角归一化。"""
    result = []
    for ch in text:
        cp = ord(ch)
        if 0xFF10 <= cp <= 0xFF19:      result.append(chr(cp - 0xFF10 + ord('0')))
        elif 0xFF21 <= cp <= 0xFF3A:    result.append(chr(cp - 0xFF21 + ord('A')))
        elif 0xFF41 <= cp <= 0xFF5A:    result.append(chr(cp - 0xFF41 + ord('a')))
        elif cp == 0x3000:              result.append(' ')
        elif cp == 0xFF0E:              result.append('.')
        elif cp == 0xFF0C:              result.append(',')
        elif cp == 0xFF1A:              result.append(':')
        elif cp == 0xFF08:              result.append('(')
        elif cp == 0xFF09:              result.append(')')
        elif cp == 0xFF0F:              result.append('/')
        elif cp == 0xFF0D:              result.append('-')
        elif cp == 0xFF0B:              result.append('+')
        elif cp == 0xFF1D:              result.append('=')
        elif cp == 0xFF05:              result.append('%')
        elif cp == 0xFF3B:              result.append('[')
        elif cp == 0xFF3D:              result.append(']')
        elif 0xFF01 <= cp <= 0xFF5E:    result.append(chr(cp - 0xFF01 + ord('!')))
        else:                           result.append(ch)
    return ''.join(result)


def _decode_fw_english(text):
    """仅解码全角拉丁字母/数字。"""
    result = []; has_fw = False
    for ch in text:
        cp = ord(ch)
        if 0xFF41 <= cp <= 0xFF5A:
            result.append(chr(cp - 0xFF41 + ord('a'))); has_fw = True
        elif 0xFF21 <= cp <= 0xFF3A:
            result.append(chr(cp - 0xFF21 + ord('A'))); has_fw = True
        elif 0xFF10 <= cp <= 0xFF19:
            result.append(chr(cp - 0xFF10 + ord('0'))); has_fw = True
        elif cp == 0x3000:
            result.append(' ')
        else:
            result.append(ch)
    return ''.join(result), has_fw


# ── 行预处理（编号断行合并） ──

# 编号片段: "3.", "1." = 片段开头; "14" = 纯数字片段（编号尾段）
# 注意: 仅当上下文存在时纯数字才作为编号片段
_RE_SEG = re.compile(r'^(\d+)\.$')
_RE_SEG_LOOSE = re.compile(r'^(\d+)$')  # 无句点的纯数字，仅当上下文已有片段时匹配
_RE_NUM_LINE = re.compile(r'^(\d+\.\d+(?:\.\d+)*(?:\([^)]*\)[-\w]*)?)\s*(.*)$')
_RE_FOOTER = re.compile(r'^[犌犅／犜\d\s]*$|^GB/T|^ICS')

def _is_term_line(text):
    """判断是否术语行: 含中文 + 含英文/全角拉丁。"""
    return bool(re.search(r'[一-鿿]', _fw2hw(text))) and bool(re.search(r'[a-zA-ZＡ-ｚ]', text))

def _parse_term(text, encoding):
    """从术语行提取 cn_term, en_term。"""
    if encoding == 'fullwidth':
        decoded, _ = _decode_fw_english(text)
    else:
        decoded = text
    decoded = re.sub(r'^[A-Za-z0-9]+[:：]\s*', '', decoded)
    cn_parts = re.findall(r'[一-鿿（）·\u00b2\u00b3]+', decoded)
    en_parts = re.findall(r'[a-zA-Z][a-zA-Z\s;/,.+\-]*[a-zA-Z]', decoded)
    return (cn_parts[0] if cn_parts else ''), ('; '.join(w.strip() for w in en_parts).lower() if en_parts else '')


def collect_number_lines(page_text, page_num):
    """第一遍扫描：从单页文本中收集所有 (line_idx, section_num, term_line)。"""
    entries = []
    lines = page_text.split('\n')
    n = len(lines)
    i = 0
    max_iter = n * 3 + 10  # 安全守卫

    while i < n:
        max_iter -= 1
        if max_iter <= 0:
            print(f"      ⚠️ collect_number_lines: max_iter at page {page_num+1}", flush=True)
            break

        raw = lines[i].strip()
        s = _fw2hw(raw)

        if not s or _RE_FOOTER.match(s):
            i += 1
            continue

        # 场景1: 编号片段链（仅当以 "N." 格式开头时触发）
        m_seg = _RE_SEG.match(s)
        if m_seg:
            parts = [m_seg.group(1)]
            j = i + 1
            while j < n and j < i + 8:
                nxt = lines[j].strip()
                nxt_n = _fw2hw(nxt)
                m2_dot = _RE_SEG.match(nxt_n)     # "N." 格式
                m2_raw = _RE_SEG_LOOSE.match(nxt_n)  # "N" 格式——仅当已有片段时才采纳
                if m2_dot:
                    parts.append(m2_dot.group(1))
                    j += 1
                elif m2_raw and len(parts) >= 1:
                    # 纯数字，作为编号尾段 "14"
                    parts.append(m2_raw.group(1))
                    j += 1
                elif re.match(r'^(\d+)\s+(.+)', nxt_n):
                    mm = re.match(r'^(\d+)\s+(.+)', nxt_n)
                    if mm and mm.group(2):
                        parts.append(mm.group(1))
                        sn = '.'.join(parts)
                        entries.append((j, sn, mm.group(2).strip()))
                        i = j + 1
                        break
                    j += 1
                elif nxt_n == '':
                    j += 1
                else:
                    break
            else:
                if j < n and len(parts) >= 2:
                    nxt = lines[j].strip()
                    if re.search(r'[一-鿿]', _fw2hw(nxt)) and re.search(r'[a-zA-ZＡ-ｚ]', nxt):
                        sn = '.'.join(parts)
                        entries.append((j, sn, nxt))
                        i = j + 1
                    else:
                        i = j
                else:
                    i = j
            continue

        # 场景2: 编号在一行开头
        m_num = _RE_NUM_LINE.match(s)
        if m_num:
            sn = m_num.group(1)
            rest = m_num.group(2).strip()
            if rest and len(rest) >= 2:
                entries.append((i, sn, rest))
            elif i + 1 < n:
                nxt = lines[i + 1].strip()
                nxt_n = _fw2hw(nxt)
                if nxt and not _RE_NUM_LINE.match(nxt_n) and not _RE_FOOTER.match(nxt_n):
                    entries.append((i + 1, sn, nxt))
            i += 1
            continue

        i += 1

    return entries


def extract_terms_from_doc(doc, domain, source_name, encoding='utf8'):
    """综合提取：两遍扫描。"""
    results = []
    page_guard = 0

    for page_num in range(len(doc)):
        page_guard += 1
        if page_guard > 500:
            print(f"      ⚠️ extract_terms_from_doc: page_guard exceeded", flush=True)
            break

    for page_num in range(len(doc)):
        raw_text = doc[page_num].get_text()
        if len(raw_text) < 20:
            continue

        number_entries = collect_number_lines(raw_text, page_num)
        if not number_entries:
            continue

        lines = raw_text.split('\n')

        for k in range(len(number_entries)):
            li, section_num, term_line = number_entries[k]

            # 确定下一编号行位置
            if k + 1 < len(number_entries):
                end_li = number_entries[k + 1][0]
            else:
                end_li = len(lines)

            # 收集定义行（术语行之后到下一编号之前）
            def_lines = []
            for d in range(li + 1, min(end_li, len(lines))):
                dl = lines[d].strip()
                dl_n = _fw2hw(dl)
                if _RE_FOOTER.match(dl_n):
                    continue
                # 跳过看起来像编号行的单行（不能太严格以免丢失内容）
                if _RE_NUM_LINE.match(dl_n) and len(dl_n) <= 15:
                    continue
                if dl:
                    def_lines.append(dl)

            if not def_lines:
                continue

            cn_term, en_term = _parse_term(term_line, encoding)
            if not en_term or len(en_term) < 2:
                continue

            # 清理定义
            if encoding == 'fullwidth':
                clean = []
                for d in def_lines:
                    dec, _ = _decode_fw_english(_fw2hw(d))
                    clean.append(dec)
                raw_def = ''.join(clean)
                ok = [ch for ch in raw_def
                      if (0x4E00 <= ord(ch) <= 0x9FFF) or (32 <= ord(ch) < 127)]
                full_def = ''.join(ok)
            else:
                full_def = ''.join(def_lines)

            if len(full_def) >= 8:
                results.append({
                    'cn_term': cn_term,
                    'en_term': en_term,
                    'cn_definition': full_def,
                    'page': page_num + 1,
                    'section_num': section_num,
                    'domain': domain,
                    'source': source_name,
                    'encoding': encoding,
                })

    return results


# ── PDF 分类器 ──

def classify_pdf(pdf_path):
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return 'image_only'
    tcjk = tfw = ttxt = 0
    for p in range(min(8, len(doc))):
        text = doc[p].get_text()
        ttxt += len(text)
        tcjk += sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
        tfw += sum(1 for c in text if 0xFF21 <= ord(c) <= 0xFF5A or 0xFF41 <= ord(c) <= 0xFF7A)
    doc.close()
    if ttxt == 0: return 'image_only'
    if tcjk < 20 and tfw < 20: return 'garbled'
    if tfw > tcjk * 0.5 and tfw > 50: return 'fullwidth'
    return 'utf8'


# ── OCR 回退 ──

def _check_ocr():
    try:
        import pytesseract
        try: pytesseract.get_tesseract_version()
        except Exception:
            for loc in [r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe']:
                if os.path.exists(loc): pytesseract.pytesseract.tesseract_cmd = loc; break
            else: return False, "tesseract not found"
        return True, "ready"
    except ImportError: return False, "pytesseract not installed"

def _extract_ocr(pdf_path, domain, source_name):
    ok, msg = _check_ocr()
    if not ok:
        print(f"      ⚠️ OCR 不可用: {msg}")
        return []
    import pytesseract
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    results = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        mode = 'RGB' if pix.n == 3 else ('RGBA' if pix.n == 4 else 'L')
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        if mode == 'RGBA': img = img.convert('RGB')
        try:
            ocr_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        except Exception:
            continue
        # Wrap in a minimal page-like object
        class FakePage:
            def get_text(self): return ocr_text
        class FakeDoc:
            def __getitem__(s, i): return FakePage()
            def __len__(s): return 1
        res = extract_terms_from_doc(FakeDoc(), domain, source_name, encoding='utf8')
        for r in res:
            r['page'] = page_num + 1
            r['encoding'] = 'ocr'
        results.extend(res)
    doc.close()
    return results


# ── 调度 ──

def extract_from_pdf(pdf_path, domain, pdf_type=None, use_ocr=False):
    sname = pdf_path.name
    if pdf_type is None: pdf_type = classify_pdf(pdf_path)
    if pdf_type in ('utf8', 'fullwidth'):
        doc = fitz.open(str(pdf_path))
        enc = 'fullwidth' if pdf_type == 'fullwidth' else 'utf8'
        res = extract_terms_from_doc(doc, domain, sname, encoding=enc)
        doc.close()
        return res
    elif pdf_type == 'garbled':
        if use_ocr: return _extract_ocr(pdf_path, domain, sname)
        else: print(f"      ⏭️ 乱码字体PDF，跳过（用 --ocr 启用OCR）"); return []
    elif pdf_type == 'image_only':
        print(f"      ⏭️ 纯图片PDF，跳过"); return []
    return []


# ── 匹配 ──

def extract_chinese_terms(text):
    return re.findall(r'[一-鿿]{2,6}', text)

def match_to_lexicon(extracted, existing_terms):
    """精确+模糊匹配。"""
    cn_index = set(existing_terms)
    matched, unmatched = [], []
    for r in extracted:
        cn_def = r.get('cn_definition', '')
        cn_term = r.get('cn_term', '')
        best, best_len = None, 0
        if cn_term and cn_term in cn_index:
            best, best_len = cn_term, len(cn_term)
        if not best:
            for seg in extract_chinese_terms(cn_def):
                if seg in cn_index and len(seg) > best_len:
                    best, best_len = seg, len(seg)
        if best:
            r['matched_term'] = best; r['match_type'] = 'cn_in_lexicon'
            matched.append(r)
        else:
            unmatched.append(r)
    # fuzzy pass
    still = []
    for r in unmatched:
        best_f = None
        for seg in extract_chinese_terms(r.get('cn_definition', '')):
            if len(seg) < 2: continue
            for term in existing_terms:
                if len(term) >= 3 and (seg in term or term in seg):
                    if not best_f or len(term) > len(best_f): best_f = term
        if best_f:
            r['matched_term'] = best_f; r['match_type'] = 'fuzzy'
            matched.append(r)
        else:
            still.append(r)
    return matched, still


# ── 词典更新 ──

def update_lexicon(lex, matched, output_path):
    updated = 0
    defs = {}
    for r in matched:
        t = r.get('matched_term', '')
        d = r.get('cn_definition', '')
        if t and (t not in defs or len(d) > len(defs[t])):
            defs[t] = d
    def add_def(obj):
        nonlocal updated
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if k in ('description', 'source', 'note', '产业定位', '政策类型',
                         'policy_count', 'cluster_info', 'type', 'aliases', 'definition'):
                    if isinstance(v, (dict, list)): add_def(v)
                    continue
                if isinstance(v, str) and len(v) >= 2 and v in defs:
                    obj['definition'] = defs[v]; updated += 1
                elif isinstance(v, dict):
                    if k in defs and 'definition' not in v: v['definition'] = defs[k]; updated += 1
                    add_def(v)
                elif isinstance(v, list): add_def(v)
        elif isinstance(obj, list):
            for item in obj: add_def(item)
    for lk, ld in lex.get('layers', {}).items():
        add_def(ld.get('terms', {}))
    lex['meta']['version'] = 'v2.7'
    lex['meta']['note'] = 'v2.7: GB/T标准PDF v3提取'
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(lex, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)
    return updated


# ── main ──

def main():
    parser = argparse.ArgumentParser(description='GB/T PDF 术语提取 v3')
    parser.add_argument('--ocr', action='store_true',
                        help='启用OCR回退（需 pytesseract + tesseract-ocr chi_sim）')
    args = parser.parse_args()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("📖 GB/T 标准 PDF 术语定义提取 (v3)")
    print("=" * 60)

    with open(LEXICON_PATH, encoding='utf-8') as f:
        lex = yaml.safe_load(f)

    all_terms = set()
    def collect(v):
        if isinstance(v, str) and len(v) >= 2: all_terms.add(v)
        elif isinstance(v, list):
            for i in v: collect(i)
        elif isinstance(v, dict):
            for k, vv in v.items():
                if k not in ('description', 'source', 'note', '产业定位', '政策类型',
                             'policy_count', 'cluster_info'):
                    collect(k); collect(vv)
    for lk, ld in lex.get('layers', {}).items():
        collect(ld.get('terms', {}))
    print(f"   词典词条: {len(all_terms)}")

    all_extracted = []; total_by_group = {}
    type_map = {'🟢':'utf8','🟡':'fullwidth','🔴':'garbled','🟠':'image_only'}

    for glabel, plist in ALL_GROUPS:
        print(f"\n{'─'*50}\n  {glabel} ({len(plist)} 个文件)")
        gc = 0
        for pname, domain in plist:
            ppath = STD_DIR / pname
            if not ppath.exists():
                print(f"    ⚠️ 文件不存在: {pname}")
                continue
            if glabel.startswith('⏭'):
                print(f"    ⏭️ 重复文件，跳过: {pname}")
                continue
            ptype = type_map.get(glabel[0], classify_pdf(ppath))
            print(f"    📄 {pname} [{domain}]...", end=' ', flush=True)
            res = extract_from_pdf(ppath, domain, ptype, use_ocr=args.ocr)
            print(f"→ {len(res)} 条", flush=True)
            all_extracted.extend(res); gc += len(res)
        total_by_group[glabel] = gc

    print(f"\n{'='*60}\n📊 汇总")
    for gl, cnt in total_by_group.items(): print(f"  {gl}: {cnt} 条")
    print(f"  总计: {len(all_extracted)} 条")

    if not all_extracted:
        print("\n⚠️ 未提取到任何条目"); return

    unique = {}
    for r in all_extracted:
        key = (r['en_term'], r.get('domain', ''))
        if key not in unique or len(r.get('cn_definition', '')) > len(unique[key].get('cn_definition', '')):
            unique[key] = r
    print(f"  去重后: {len(unique)} 条")

    print(f"\n🔗 匹配到词典...")
    matched, unmatched = match_to_lexicon(list(unique.values()), all_terms)
    print(f"   匹配: {len(matched)}  未匹配: {len(unmatched)}")

    if matched:
        print(f"\n📋 匹配样本 (前15):")
        for r in matched[:15]:
            cn = r.get('cn_term', r.get('matched_term', '?'))
            print(f"  [{r['domain']}][{r.get('encoding','?')}] en={r['en_term']} cn={cn} → {r['matched_term']}")
            print(f"    {r['cn_definition'][:100]}...")

    # 保存
    out = DATA_DIR / "extracted_definitions_v3.csv"
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['en_term','cn_term','matched_term','match_type',
                                          'cn_definition','domain','source','encoding','section_num','page'],
                           extrasaction='ignore')
        w.writeheader()
        for r in matched: w.writerow(r)
        for r in unmatched: w.writerow(r)
    print(f"\n💾 {out}")

    if matched:
        u = update_lexicon(lex, matched, LEXICON_PATH)
        print(f"📝 词典更新: {u} 条定义 → {LEXICON_PATH}")

    if unmatched:
        up = DATA_DIR / "unmatched_terms_v3.csv"
        with open(up, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['en_term','cn_definition','domain','source','encoding'])
            w.writeheader()
            for r in unmatched:
                w.writerow({k: r.get(k, '') for k in ['en_term','cn_definition','domain','source','encoding']})
        print(f"📋 未匹配清单: {up}")


if __name__ == "__main__":
    main()
