#!/usr/bin/env python3
"""词典反馈与审核工具 — 生成待审核清单、标记审批、记录修改历史。

用法:
    # 生成待审核清单
    python textile_dict/tools/term_reviewer.py --report

    # 导出 CSV 供人工标注
    python textile_dict/tools/term_reviewer.py --export-csv

    # 根据人工标注的 CSV 更新词典
    python textile_dict/tools/term_reviewer.py --apply review_result.csv

    # 查看词典修改历史
    python textile_dict/tools/term_reviewer.py --history
"""

import sys, io, csv, yaml, json, hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
LEX_PATH = PACKAGE_ROOT / "data" / "lexicon_v2.yaml"
HISTORY_PATH = PACKAGE_ROOT / "data" / "review_history.jsonl"

META_KEYS = {
    'description', 'source', 'note', '产业定位', '政策类型',
    'policy_count', 'cluster_info', 'type', 'aliases', 'definition',
}


def load_lexicon():
    with open(LEX_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_lexicon(lex):
    with open(LEX_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(lex, f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False, width=120)


def iter_definitions(lex: dict):
    """遍历所有含默认的术语条目，yield (term, entry_dict, layer, quality_notes)。"""
    def walk(obj, layer=''):
        if isinstance(obj, dict):
            if 'term' in obj and 'definition' in obj:
                src = obj.get('source', '')
                notes = _quality_notes(obj, src)
                yield obj['term'], obj, layer, notes
            for k, v in obj.items():
                if k in META_KEYS:
                    continue
                if isinstance(v, dict) and 'definition' in v:
                    src = v.get('source', '')
                    notes = _quality_notes(v, src)
                    yield k, v, layer, notes
                new_layer = layer
                if k.startswith('layer_'):
                    new_layer = k
                yield from walk(v, new_layer)
        elif isinstance(obj, list):
            for item in obj:
                yield from walk(item, layer)

    for lk, ld in lex['layers'].items():
        yield from walk(ld.get('terms', {}), lk)


def _quality_notes(entry: dict, source: str) -> str:
    """自动评估定义质量并给出备注。"""
    d = entry.get('definition', '')
    issues = []

    if 'OCR' in source:
        # 检查噪声
        noise_chars = sum(1 for c in d if c in '…≡≈≤≥°±¾□■')
        if noise_chars > 0:
            issues.append(f'含{noise_chars}个非标准字符')
        if len(d) < 15:
            issues.append('定义过短(<15字)')
        if any(kw in d for kw in ['目uH', '悲[PEE', '矢豆子', '】军', '_work']):
            issues.append('含已知OCR噪声标记')
        issues.append('OCR提取，需人工核验')

    if 'HS' in source or '海关' in source:
        issues.append('自动生成定义（频次统计），需人工核实')
        if len(d) < 30:
            issues.append('定义信息量低')

    if '人工编录' in source or '知识图谱' in source:
        # 需要标记 confidence
        pass  # 不添加 issues

    return '; '.join(issues) if issues else ''


def generate_report():
    """生成待审核清单报告。"""
    lex = load_lexicon()
    entries = list(iter_definitions(lex))

    # 按风险等级分类
    high_risk = []  # OCR噪声 + HS自动生成
    medium_risk = []  # OCR无噪声但未验证
    low_risk = []  # 静态编录

    for term, entry, layer, notes in entries:
        src = entry.get('source', '')
        if 'OCR' in src and notes:
            high_risk.append((term, entry, layer, notes))
        elif 'HS' in src or '海关' in src:
            high_risk.append((term, entry, layer, notes))
        elif 'OCR' in src:
            medium_risk.append((term, entry, layer, notes))
        else:
            low_risk.append((term, entry, layer, notes))

    print("=" * 60)
    print("  纺织词典术语审核清单")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print()
    print(f"  总计: {len(entries)} 条定义")
    print(f"  高风险（需人工审核）: {len(high_risk)} 条")
    print(f"    - OCR含噪声: {sum(1 for _,e,_,n in high_risk if 'OCR' in e.get('source',''))} 条")
    print(f"    - HS自动生成: {sum(1 for _,e,_,n in high_risk if 'HS' in e.get('source','') or '海关' in e.get('source',''))} 条")
    print(f"  中风险（OCR未验证）: {len(medium_risk)} 条")
    print(f"  低风险（人工编录）: {len(low_risk)} 条")
    print()

    if high_risk:
        print("高风险条目示例 (前20):")
        print(f"  {'术语':20s} {'来源':30s} {'问题':40s}")
        print(f"  {'-'*20} {'-'*30} {'-'*40}")
        for term, entry, layer, notes in high_risk[:20]:
            src = entry.get('source', '')[:30]
            print(f"  {term:20s} {src:30s} {notes[:40]}")

    return entries


def export_review_csv(output_path: str = None):
    """导出待审核 CSV。"""
    if output_path is None:
        output_path = str(PACKAGE_ROOT / "data" / "review_checklist.csv")

    lex = load_lexicon()
    entries = list(iter_definitions(lex))

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'term', 'layer', 'layer_name', 'definition', 'source',
            'auto_quality_notes', 'reviewer_verdict', 'corrected_definition',
            'reviewer_comment', 'review_date'
        ])

        for term, entry, layer, notes in entries:
            layer_name = lex['layers'].get(layer, {}).get('description', layer) if layer else ''
            writer.writerow([
                term,
                layer,
                layer_name,
                entry.get('definition', ''),
                entry.get('source', ''),
                notes,
                '',  # reviewer_verdict (待填: ✅/❌/修正)
                '',  # corrected_definition (如有修正)
                '',  # reviewer_comment
                '',  # review_date
            ])

    print(f"  导出: {output_path}")
    print(f"  共 {len(entries)} 条待审核")
    print()
    print("  请在 'reviewer_verdict' 列填写:")
    print("    ✅   = 通过（定义正确）")
    print("    修正  = 请在 'corrected_definition' 列填写修正后的定义")
    print("    ❌   = 拒绝（此条目不纳入词典）")
    return output_path


def apply_review(review_csv: str):
    """读取人工审核结果并更新词典。"""
    lex = load_lexicon()

    approved = 0
    corrected = 0
    rejected = 0
    records = []

    with open(review_csv, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            verdict = (row.get('reviewer_verdict') or '').strip()
            term = (row.get('term') or '').strip()
            if not term or not verdict:
                continue

            if verdict == '✅':
                approved += 1
            elif verdict == '修正':
                new_def = (row.get('corrected_definition') or '').strip()
                if new_def and term:
                    _update_definition(lex, term, new_def)
                    corrected += 1
            elif verdict == '❌':
                rejected += 1

            records.append({
                'term': term,
                'verdict': verdict,
                'corrected_definition': row.get('corrected_definition', ''),
                'comment': row.get('reviewer_comment', ''),
                'review_date': row.get('review_date', datetime.now().strftime('%Y-%m-%d')),
            })

    # 记录到历史
    _log_review(records)

    # 更新词典 meta
    lex['meta']['last_reviewed'] = datetime.now().strftime('%Y-%m-%d')
    review_note = f'审核: {approved}通过/{corrected}修正/{rejected}拒绝'
    lex['meta']['note'] = lex['meta'].get('note', '') + '; ' + review_note
    save_lexicon(lex)

    print(f"  审核结果已应用:")
    print(f"    ✅ 通过: {approved}")
    print(f"    修正: {corrected}")
    print(f"    ❌ 拒绝: {rejected}")
    print(f"  历史记录: {HISTORY_PATH}")


def _update_definition(lex: dict, term: str, new_definition: str):
    """在词典中查找并更新术语定义。"""
    def walk(obj):
        if isinstance(obj, dict):
            if 'term' in obj and obj['term'] == term:
                obj['definition'] = new_definition
                obj['reviewed'] = True
                return True
            for k, v in obj.items():
                if k in META_KEYS:
                    continue
                if k == term and isinstance(v, dict) and 'definition' in v:
                    v['definition'] = new_definition
                    v['reviewed'] = True
                    return True
                if walk(v):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if walk(item):
                    return True
        return False

    for ld in lex['layers'].values():
        if walk(ld.get('terms', {})):
            return


def _log_review(records: list):
    """将审核记录写入 JSONL 历史文件。"""
    with open(HISTORY_PATH, 'a', encoding='utf-8') as f:
        for r in records:
            r['timestamp'] = datetime.now().isoformat()
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def show_history():
    """显示最近的修改历史。"""
    if not HISTORY_PATH.exists():
        print("  暂无审核历史记录。")
        return

    with open(HISTORY_PATH, encoding='utf-8') as f:
        lines = f.readlines()

    print(f"  共 {len(lines)} 条审核记录")
    print()
    for line in lines[-20:]:
        r = json.loads(line)
        verdict = r.get('verdict', '?')
        term = r.get('term', '?')
        ts = r.get('timestamp', '')[:16]
        print(f"  [{ts}] {verdict:4s} {term}")


def show_summary():
    """展示词典质量总览。"""
    lex = load_lexicon()
    entries = list(iter_definitions(lex))

    print("=" * 60)
    print("  纺织词典 — 术语定义质量报告")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print()
    print(f"  版本: {lex['meta']['version']}")
    print(f"  总词条: {lex['meta']['total_terms']}")
    print(f"  有定义: {len(entries)} 条")
    print()

    # 按来源质量分组
    from collections import Counter
    by_source = Counter()
    by_layer = Counter()
    for term, entry, layer, notes in entries:
        src = entry.get('source', '')
        if 'OCR' in src:
            by_source['OCR提取(需核验)'] += 1
        elif 'HS' in src or '海关' in src:
            by_source['HS自动提取(需核实)'] += 1
        elif '产业布局' in src or '规则引擎' in src:
            by_source['知识图谱编录(可信)'] += 1
        elif '行业分类' in src or '数字经济' in src:
            by_source['官方分类标准(可信)'] += 1
        elif '人工编录' in src:
            by_source['人工编录(可信)'] += 1
        else:
            by_source['其他'] += 1

        if layer:
            by_layer[layer] += 1

    print("  定义来源分布:")
    for src, cnt in by_source.most_common():
        print(f"    {src:30s}: {cnt:>4} 条")

    print()
    print("  按层分布:")
    for lk in sorted(by_layer.keys()):
        name = lex['layers'].get(lk, {}).get('description', lk)
        print(f"    {name:35s}: {by_layer[lk]:>4} 条")

    # 审核状态
    if HISTORY_PATH.exists():
        reviewed = set()
        with open(HISTORY_PATH, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                reviewed.add(r.get('term', ''))
        print(f"\n  已审核: {len(reviewed)} 条")
        print(f"  待审核: {len(entries) - len(reviewed)} 条")
    else:
        print(f"\n  待审核: {len(entries)} 条 (尚未开始审核)")


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='纺织词典术语审核与反馈工具')
    parser.add_argument('--report', action='store_true', help='生成待审核清单')
    parser.add_argument('--export-csv', nargs='?', const=None, help='导出审核CSV')
    parser.add_argument('--apply', metavar='CSV', help='应用审核结果')
    parser.add_argument('--history', action='store_true', help='查看修改历史')
    parser.add_argument('--summary', action='store_true', help='查看质量总览')
    args = parser.parse_args()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    if args.report:
        generate_report()
    elif args.export_csv is not None or '--export-csv' in sys.argv:
        export_review_csv(args.export_csv if args.export_csv != 'const' else None)
    elif args.apply:
        apply_review(args.apply)
    elif args.history:
        show_history()
    elif args.summary:
        show_summary()
    else:
        show_summary()


if __name__ == '__main__':
    main()
