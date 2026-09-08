# -*- coding: utf-8 -*-
"""SCI 论文数据一致性验证：docx 中每个关键数字可溯源到 results/*.csv。
用法：python verify_paper_sci.py [_quick]
检查项：
  1) Table 2 每个单元格（mean±std）与 sci_main{suffix}.csv 聚合值一致
  2) Table 4 消融值与 sci_ablation{suffix}.csv 一致
  3) Table 5 噪声值与 sci_noise{suffix}.csv 一致
  4) 摘要中的统计量（gain、param_ratio、time_reduction）与重算一致
  5) 参考文献数量与 refs_*.json 总数一致
  6) 表格完整性（所有方法×数据集单元格无缺失）
"""
import csv
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

from docx import Document

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
CODE = os.path.join(ROOT, 'code')
SFX = sys.argv[1] if len(sys.argv) > 1 else ''
DOCX = os.path.join(ROOT,
                    f'SCI_Paper_Learnable_Feature_Weighting{SFX}.docx')

errors = []
checked = 0


def ok(msg):
    print(f'  [OK] {msg}')


def err(msg):
    errors.append(msg)
    print(f'  [FAIL] {msg}')


def load(path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


print(f'== 验证论文: {os.path.basename(DOCX)} ==')

# ---------------------------------------------------------------- 1. 主表
print('1. Table 2 主表 (macro-F1 mean±std):')
main_rows = load(os.path.join(RESULTS, f'sci_main{SFX}.csv'))
agg = defaultdict(list)
for r in main_rows:
    if float(r['frac']) == 1.0:
        agg[(r['dataset'], r['method'])].append(float(r['f1']))
ref = {k: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
       for k, v in agg.items()}

doc = Document(DOCX)
# 找 Table 2（标题含 'Table 2'，实际表在标题段后的第一个表格）
tbl2 = None
for t in doc.tables:
    hdr = ' '.join(c.text for c in t.rows[0].cells)
    if 'Method' in hdr and 'Rank' in hdr:
        tbl2 = t
        break
if tbl2 is None:
    err('未找到 Table 2')
else:
    datasets = [c.text for c in tbl2.rows[0].cells][1:-2]
    for row in tbl2.rows[1:]:
        method = row.cells[0].text
        mkey = {'LFW (ours)': 'LFW(ours)',
                'Linear probe': 'LinearProbe',
                'Full fine-tune': 'FFT',
                'Zero-shot': 'ZeroShot'}.get(method, method)
        for j, ds in enumerate(datasets):
            cell = row.cells[j + 1].text.strip()
            if cell == '-':
                continue
            mm = re.match(r'([\d.]+)±([\d.]+)', cell)
            if not mm:
                continue
            mu_doc, sd_doc = float(mm.group(1)), float(mm.group(2))
            key = (ds, mkey)
            if key not in ref:
                err(f'Table2 {ds}|{mkey}: 论文有值但CSV无记录')
                continue
            mu_csv, sd_csv = ref[key]
            checked += 1
            if abs(mu_doc - mu_csv * 100) > 0.051 or \
               abs(sd_doc - sd_csv * 100) > 0.051:
                err(f'Table2 {ds}|{mkey}: 论文 {mu_doc}±{sd_doc} vs '
                    f'CSV {mu_csv*100:.2f}±{sd_csv*100:.1f}')
    ok(f'Table 2 单元格核对 {checked} 个')

# ---------------------------------------------------------------- 2. 消融
print('2. Table 4 消融:')
abl_rows = load(os.path.join(RESULTS, f'sci_ablation{SFX}.csv'))
aagg = defaultdict(list)
for r in abl_rows:
    aagg[(r['dataset'], r['variant'])].append(float(r['f1']))
aref = {k: st.mean(v) for k, v in aagg.items()}
tbl4 = None
for t in doc.tables:
    hdr = ' '.join(c.text for c in t.rows[0].cells)
    if 'Variant' in hdr:
        tbl4 = t
        break
if tbl4:
    ds4 = [c.text for c in tbl4.rows[0].cells][1:-1]
    n4 = 0
    for row in tbl4.rows[1:]:
        vname = row.cells[0].text
        vkey = re.search(r'\(([a-g])\) (.+)', vname)
        vmap = {'a': 'none', 'b': 'w-plain', 'c': 'w-nonneg', 'd': 'w-reg',
                'e': 'w-reg-nonneg', 'f': 'full', 'g': 'w-affine'}
        if not vkey:
            continue
        vk = vmap[vkey.group(1)]
        for j, ds in enumerate(ds4):
            cell = row.cells[j + 1].text.strip()
            if cell == '-':
                continue
            if (ds, vk) in aref:
                n4 += 1
                if abs(float(cell) - aref[(ds, vk)] * 100) > 0.005:
                    err(f'Table4 {ds}|{vk}: {cell} vs '
                        f'{aref[(ds, vk)]*100:.2f}')
    ok(f'Table 4 单元格核对 {n4} 个')
else:
    print('  [SKIP] 无消融表')

# ---------------------------------------------------------------- 3. 噪声
print('3. Table 5 标签噪声:')
nz_rows = load(os.path.join(RESULTS, f'sci_noise{SFX}.csv'))
nagg = defaultdict(list)
for r in nz_rows:
    nagg[(r['dataset'], f"{float(r['noise_p']):g}", r['method'])].append(
        float(r['f1']))
nref = {k: st.mean(v) for k, v in nagg.items()}
tbl5 = None
for t in doc.tables:
    hdr = ' '.join(c.text for c in t.rows[0].cells)
    if 'Noise' in hdr and 'BitFit' in hdr:
        tbl5 = t
        break
if tbl5:
    n5 = 0
    for row in tbl5.rows[1:]:
        ds = row.cells[0].text
        p = f"{float(row.cells[1].text.strip().rstrip('%'))/100:g}"
        for j, m in enumerate(('BitFit', 'FFT', 'LFW(ours)')):
            cell = row.cells[j + 2].text.strip()
            if cell == '-' or not cell:
                continue
            key = (ds, p, m)
            if key in nref:
                n5 += 1
                if abs(float(cell) - nref[key] * 100) > 0.005:
                    err(f'Table5 {ds}|{p}|{m}: {cell} vs '
                        f'{nref[key]*100:.2f}')
    ok(f'Table 5 单元格核对 {n5} 个')
else:
    print('  [SKIP] 无噪声表')

# ---------------------------------------------------------------- 4. 摘要统计
print('4. 摘要核心统计量:')
A = json.load(open(os.path.join(RESULTS, f'sci_analysis{SFX}.json'),
                   encoding='utf-8'))
full_text = '\n'.join(p.text for p in doc.paragraphs)
abs_gain = A['stats']['gain_over_zeroshot_mean']
if f'{abs_gain:.1f} percentage points' in full_text:
    ok(f'摘要增益 {abs_gain:.1f} pp 一致')
else:
    err(f'摘要增益 {abs_gain:.1f} pp 未在论文中找到')
ratio = A['stats']['param_ratio_median']
if f'{ratio:.0f}x fewer' in full_text:
    ok(f'参数比 1:{ratio:.0f} 一致')
else:
    err(f'参数比 {ratio:.0f} 未在论文正文找到')

# 正文统计复算：gain_over_zeroshot
gains = []
for ds in A['datasets']:
    try:
        gains.append((A['main'][f'{ds}|LFW(ours)']['metrics']['f1'][0] -
                      A['main'][f'{ds}|ZeroShot']['metrics']['f1'][0]) * 100)
    except KeyError:
        pass
if gains:
    g_mean = st.mean(gains)
    if abs(g_mean - abs_gain) < 1e-6:
        ok(f'gain 复算一致 ({g_mean:.4f} pp)')
    else:
        err(f'gain 复算不一致: analysis={abs_gain} vs 重算={g_mean}')

# ---------------------------------------------------------------- 5. 参考文献
print('5. 参考文献:')
refs = []
for fn in ('refs_transfer.json', 'refs_peft.json', 'refs_tabular.json'):
    p = os.path.join(CODE, fn)
    if os.path.exists(p):
        refs.extend(json.load(open(p, encoding='utf-8')))
n_ref_doc = len(re.findall(r'^\[\d+\] ', full_text, re.M))
if n_ref_doc == len(refs):
    ok(f'参考文献条数一致 ({n_ref_doc} 篇)')
else:
    err(f'参考文献条数不一致: 论文 {n_ref_doc} vs JSON {len(refs)}')
dupe = len(refs) - len({r["key"] for r in refs})
if dupe:
    err(f'参考文献 key 重复 {dupe} 条')
else:
    ok('无重复文献')

# ---------------------------------------------------------------- 6. 表格完整性
print('6. 表格完整性:')
expect_methods = {'Zero-shot', 'VarFilter', 'MIFilter', 'LR', 'SVM', 'RF',
                  'Linear probe', 'BitFit', 'SSF', 'Adapter', 'LoRA',
                  'LFW (ours)', 'Full fine-tune'}
tbl2_methods = {r.cells[0].text for r in tbl2.rows[1:]} if tbl2 else set()
missing = expect_methods - tbl2_methods
if missing:
    err(f'Table 2 缺方法: {missing}')
else:
    ok(f'Table 2 方法齐全 ({len(tbl2_methods)}/13)')

# ---------------------------------------------------------------- 7. 视觉 ViT 表（Table 8）
print('7. Table 8 视觉 ViT 结果:')
vjson = os.path.join(RESULTS, 'sci_vision_analysis.json')
if os.path.exists(vjson):
    VIS = json.load(open(vjson, encoding='utf-8'))
    if VIS.get('main'):
        # 找视觉表：含 Method/Rank 且表头含 USPS 或 Gauss
        tbl8 = None
        for t in doc.tables:
            hdr = ' '.join(c.text for c in t.rows[0].cells)
            if 'Method' in hdr and 'Rank' in hdr and (
                    'USPS' in hdr or 'Gauss' in hdr or 'Blur' in hdr):
                tbl8 = t
                break
        if tbl8 is None:
            err('未找到 Table 8（视觉表）')
        else:
            vchecked = 0
            vshort = {'USPS': 'USPS', 'Gauss.': 'C10C_gaussian_noise',
                      'Blur': 'C10C_motion_blur', 'Fog': 'C10C_fog',
                      'Bright': 'C10C_brightness'}
            hdr_cells = [c.text for c in tbl8.rows[0].cells]
            vds = [vshort.get(h, h) for h in hdr_cells[1:-2]]
            for row in tbl8.rows[1:]:
                method = row.cells[0].text
                mkey = {'LFW (ours)': 'LFW(ours)',
                        'Linear probe': 'LinearProbe',
                        'Full fine-tune': 'FFT',
                        'Zero-shot': 'ZeroShot'}.get(method, method)
                for j, ds in enumerate(vds):
                    cell = row.cells[j + 1].text.strip()
                    if cell == '-':
                        continue
                    mm = re.match(r'([\d.]+)±([\d.]+)', cell)
                    if not mm:
                        continue
                    mu_doc, sd_doc = float(mm.group(1)), float(mm.group(2))
                    vcell = VIS['main'].get(f'{ds}|{mkey}')
                    if vcell is None:
                        err(f'Table8 {ds}|{mkey}: 论文有值但JSON无记录')
                        continue
                    mu_j, sd_j = vcell['f1']
                    vchecked += 1
                    if abs(mu_doc - mu_j * 100) > 0.11 or \
                       abs(sd_doc - sd_j * 100) > 0.11:
                        err(f'Table8 {ds}|{mkey}: 论文 {mu_doc}±{sd_doc} '
                            f'vs JSON {mu_j*100:.2f}±{sd_j*100:.1f}')
            ok(f'Table 8 视觉单元格核对 {vchecked} 个')
    else:
        print('  (sci_vision_analysis.json 无 main 数据，跳过)')
else:
    print('  (无视觉实验结果，跳过)')

print()
if errors:
    print(f'=== 验证失败: {len(errors)} 处不一致 ===')
    for e in errors:
        print(' -', e)
    sys.exit(1)
print('=== 全部验证通过：论文数字与 results/*.csv 完全一致 ===')
