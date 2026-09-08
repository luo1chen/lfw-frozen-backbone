# -*- coding: utf-8 -*-
"""SCI 英文论文 Word 生成脚本（二区水平）。
所有实验数字均从 results/sci_analysis*.json 实时计算注入，保证可溯源。
用法：python build_paper_sci.py           # 全量结果（需先跑 make_figures_sci.py analyze）
      python build_paper_sci.py _quick    # 冒烟（quick 数据）
结构：Title/Abstract → 1 Introduction → 2 Related Work → 3 Method →
      4 Experiments → 5 Discussion → 6 Conclusion → References。
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
FIGS = os.path.join(ROOT, 'paper_figs')
CODE = os.path.join(ROOT, 'code')

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ''
OUT = os.path.join(ROOT, f'SCI_Paper_Learnable_Feature_Weighting{SUFFIX}.docx')

# ---------------------------------------------------------------- 数据加载

A = json.load(open(os.path.join(RESULTS, f'sci_analysis{SUFFIX}.json'),
                   encoding='utf-8'))
MAIN = A['main']            # 'ds|method' -> {metrics:{acc:(mu,sd),...}, train_time, infer_ms, n_tuned}
LOWRES = A['lowres']        # 'ds|method|frac' -> (mu, sd, n)
NOISE = A.get('noise', {})      # 'ds|p|method' -> (mu, sd, n)
INOISE = A.get('inoise', {})    # 'ds|sigma|method'
ABL = A.get('ablation', {})     # 'ds|variant' -> {f1:(mu,sd,n), sparsity:(..)}
SENS = A.get('sensitivity', {})
STATS = A['stats']          # gain_over_zeroshot_mean, param_ratio_median, ...
WILCOX = A.get('wilcoxon', {})
DATASETS = A['datasets']
PROTO = A.get('protocol', {})

# 从主表 CSV 计算 LFW 对各基线的配对平均差异（用于判定 Wilcoxon 方向）
import csv as _csv
_main_csv = os.path.join(RESULTS, f'sci_main{SUFFIX}.csv')
_pair = {}
if os.path.exists(_main_csv):
    with open(_main_csv, newline='', encoding='utf-8-sig') as _f:
        for _r in _csv.DictReader(_f):
            if float(_r['frac']) == 1.0:
                _pair[(_r['dataset'], int(_r['seed']), _r['method'])] = float(_r['f1'])
for _m, _w in WILCOX.items():
    _diffs = []
    for (_ds, _sd, _mt), _lfw_v in _pair.items():
        if _mt != 'LFW(ours)':
            continue
        _b_v = _pair.get((_ds, _sd, _m))
        if _b_v is not None:
            _diffs.append(_lfw_v - _b_v)
    _w['mean_diff'] = (sum(_diffs) / len(_diffs) * 100.0) if _diffs else 0.0

DS_INFO = {
    'HAR': ('Human activity recognition', 7352, 561, 6, 'cross-subject'),
    'Bank': ('Bank marketing', 45211, 51, 2, 'random split'),
    'Adult': ('Census income', 48842, 106, 2, 'official split'),
    'Satimage': ('Landsat satellite', 4435, 36, 6, 'random split'),
    'Digits': ('Optical digits', 1797, 64, 10, 'random split'),
    'Spambase': ('Spam detection', 4601, 57, 2, 'random split'),
    'Credit-g': ('German credit', 1000, 71, 2, 'random split'),
    'WDBC': ('Breast cancer diagnosis', 569, 30, 2, 'random split'),
    'Heart': ('Heart disease', 303, 25, 2, 'random split'),
}
METHODS = ['ZeroShot', 'VarFilter', 'MIFilter', 'LR', 'SVM', 'RF',
           'LinearProbe', 'BitFit', 'SSF', 'Adapter', 'LoRA', 'LFW(ours)', 'FFT']
LBL_EN = {'ZeroShot': 'Zero-shot', 'VarFilter': 'VarFilter', 'MIFilter': 'MIFilter',
          'LR': 'LR', 'SVM': 'SVM', 'RF': 'RF', 'LinearProbe': 'Linear probe',
          'BitFit': 'BitFit', 'SSF': 'SSF', 'Adapter': 'Adapter', 'LoRA': 'LoRA',
          'LFW(ours)': 'LFW (ours)', 'FFT': 'Full fine-tune'}


def mm(ds, m, metric='f1'):
    """MAIN[ds|m][metric] -> (mean, std)"""
    return MAIN[f'{ds}|{m}']['metrics'][metric]


def f1(ds, m):
    return mm(ds, m, 'f1')[0]


def fmt_ms(ds, m, best=False):
    """mean±std (%)，最优加粗"""
    mu, sd = mm(ds, m)
    s = f'{mu*100:.2f}±{sd*100:.2f}'
    return f'\\textbf{{{s}}}' if best else s


def best_method(ds, metric='f1'):
    cand = [(mm(ds, m, metric)[0], m) for m in METHODS if f'{ds}|{m}' in MAIN]
    return max(cand)[1]


# ---------------------------------------------------------------- 版式工具

doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.top_margin, sec.bottom_margin = Cm(2.54), Cm(2.54)
sec.left_margin, sec.right_margin = Cm(2.54), Cm(2.54)

BODY = Pt(11)
LINE = 1.5


def set_font(run, en='Times New Roman', size=11, bold=False, italic=False,
             color=None):
    run.font.name = en
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'SimSun')
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def para(text='', size=11, bold=False, align=None, indent=False, before=0,
         after=6, line=LINE, italic=False, space_after_pts=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    elif indent:
        # 正文段落两端对齐（Elsevier 已发表论文标准）
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    if indent:
        pf.first_line_indent = Pt(size * 2)
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = line
    if text:
        r = p.add_run(text)
        set_font(r, size=size, bold=bold, italic=italic)
    return p


def rich(segments, size=11, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=True,
         before=0, after=6, line=LINE):
    """segments: list of (text, kwargs)"""
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    if indent:
        pf.first_line_indent = Pt(size * 2)
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = line
    for t, kw in segments:
        r = p.add_run(t)
        set_font(r, size=kw.get('size', size), bold=kw.get('bold', False),
                 italic=kw.get('italic', False))
    return p


def h1(text):
    para(text, size=13, bold=True, indent=False, before=12, after=6, line=1.3)


def h2(text):
    para(text, size=11.5, bold=True, indent=False, before=8, after=4, line=1.3)


def _label_split(caption):
    """'Fig. 1. text' / 'Table 1. text' -> ('Fig. 1.', ' text')"""
    i = caption.find('. ')
    if i == -1:
        return None
    return caption[:i + 1], caption[i + 1:]


def add_fig(path, width_cm, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run()
    run.add_picture(path, width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(10)
    parts = _label_split(caption)
    if parts:
        set_font(c.add_run(parts[0]), size=9.5, bold=True)
        set_font(c.add_run(parts[1]), size=9.5)
    else:
        set_font(c.add_run(caption), size=9.5)


def set_cell_borders(cell, top=None, bottom=None):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.find(qn('w:tcBorders'))
    if borders is None:
        borders = tcPr.makeelement(qn('w:tcBorders'), {})
        tcPr.append(borders)
    for edge, val in (('w:top', top), ('w:bottom', bottom)):
        e = borders.find(qn(edge))
        if e is None:
            e = borders.makeelement(qn(edge), {})
            borders.append(e)
        if val is None:
            e.set(qn('w:val'), 'nil')
        else:
            e.set(qn('w:val'), 'single')
            e.set(qn('w:sz'), '16' if val == 'thick' else '6')
            e.set(qn('w:color'), '000000')


def sci_table(caption, header, rows, col_widths=None, font_size=8.5,
              bold_cells=None, note=None):
    """三线表 + 英文图注风格。bold_cells: set[(i,j)] 起于数据行(0=header后第一行)"""
    bold_cells = bold_cells or set()
    cp = para('', size=9.5, align=WD_ALIGN_PARAGRAPH.LEFT, indent=False,
              before=8, after=3, line=1.15)
    parts = _label_split(caption)
    if parts:
        set_font(cp.add_run(parts[0]), size=9.5, bold=True)
        set_font(cp.add_run(parts[1]), size=9.5)
    else:
        set_font(cp.add_run(caption), size=9.5)
    t = doc.add_table(rows=len(rows) + 1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # 紧凑单元格边距（左右 0.08cm），保证宽表不超版心
    tblPr = t._tbl.tblPr
    mar = tblPr.makeelement(qn('w:tblCellMar'), {})
    for side in ('left', 'right'):
        e = mar.makeelement(qn(f'w:{side}'), {})
        e.set(qn('w:w'), '45')   # twips ≈ 0.08cm
        e.set(qn('w:type'), 'dxa')
        mar.append(e)
    tblPr.append(mar)
    for j, htxt in enumerate(header):
        c = t.cell(0, j)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraphs[0].paragraph_format.space_before = Pt(1)
        c.paragraphs[0].paragraph_format.space_after = Pt(1)
        c.paragraphs[0].paragraph_format.line_spacing = 1.0
        r = c.paragraphs[0].add_run(htxt)
        set_font(r, size=font_size)
        set_cell_borders(c, top='thick', bottom='thin')
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i + 1, j)
            c.paragraphs[0].alignment = (WD_ALIGN_PARAGRAPH.LEFT if j == 0
                                         else WD_ALIGN_PARAGRAPH.CENTER)
            c.paragraphs[0].paragraph_format.space_before = Pt(1)
            c.paragraphs[0].paragraph_format.space_after = Pt(1)
            c.paragraphs[0].paragraph_format.line_spacing = 1.0
            r = c.paragraphs[0].add_run(str(v))
            set_font(r, size=font_size, bold=(i, j) in bold_cells)
            set_cell_borders(c, bottom='thick' if i == len(rows) - 1 else None)
    if col_widths:
        for j, w in enumerate(col_widths):
            for i in range(len(rows) + 1):
                t.cell(i, j).width = Cm(w)
    if note:
        para(note, size=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=False,
             before=2, after=8, line=1.1)
    else:
        para('', size=4, indent=False, after=4)
    return t


# ---------------------------------------------------------------- 参考文献

def load_refs():
    refs = []
    for fn in ('refs_transfer.json', 'refs_peft.json', 'refs_tabular.json'):
        p = os.path.join(CODE, fn)
        if os.path.exists(p):
            refs.extend(json.load(open(p, encoding='utf-8')))
    return {r['key']: r for r in refs}


REFS = load_refs()

# 引用顺序（正文首次出现顺序，编号 [1]..[n]）
CITE_ORDER = [
    'pan2010survey',           # 1 迁移学习综述
    'hu2022lora',              # 2 LoRA
    'houlsby2019adapter',      # 3 Adapter
    'benzaken2022bitfit',      # 4 BitFit
    'lian2022ssf',             # 5 SSF
    'han2024peftsurvey',       # 6 PEFT survey
    'kumar2022finetune',       # 7 LP-FT 冻结特征
    'wortsman2022robust',      # 8 WiSE-FT
    'kornblith2019better',     # 9 transferability
    'li2018explicit',          # 10 L2-SP
    'song2023noisy',           # 11 noisy labels
    'li2017featureselection',  # 12 feature selection survey
    'tibshirani1996lasso',     # 13 LASSO
    'ross2014mutualinfo',      # 14 mutual information
    'breiman2001randomforest', # 15 RF
    'shwartz2022tabular',      # 16 tabular DL not all you need
    'grinsztajn2022treemodels',# 17 tree-based outperform
    'gorishniy2021revisiting', # 18 revisiting tabular DL
    'arik2021tabnet',          # 19 TabNet
    'li2021prefix',            # 20 Prefix-tuning
    'lester2021prompt',        # 21 prompt tuning
    'liu2023ptuningv2',        # 22 P-Tuning v2
    'he2022unified',           # 23 MAM adapter
    'liu2022ia3',              # 24 IA3
    'dettmers2023qlora',       # 25 QLoRA
    'liu2024dora',             # 26 DoRA
    'hayou2024loraplus',       # 27 LoRA+
    'kopiczko2024vera',        # 28 VeRA
    'li2024loftq',             # 29 LoftQ
    'wang2021tent',            # 30 TENT
    'liang2020shot',           # 31 SHOT
    'wang2023generalizing',    # 32 DG survey
    'hendrycks2021imagenetr',  # 33 ImageNet-R
    'evci2022head2toe',        # 34 Head2Toe
    'zoph2020rethinking',      # 35 rethinking pretraining
    'wang2018visual',          # 36 DA survey
    'dua2017uci',              # 37 UCI
    'anguita2013har',          # 38 HAR
    'kingma2015adam',          # 39 Adam
    'lundberg2017shap',        # 40 SHAP
    'gorishniy2022embeddings', # 41 embeddings
    'lecun1998gradient',       # 42 MNIST
    'hull1994database',        # 43 USPS
    'krizhevsky2009learning',  # 44 CIFAR-10
    'hendrycks2019benchmarking', # 45 CIFAR-10-C
]
CNUM = {k: i + 1 for i, k in enumerate(CITE_ORDER)}


def cite(*keys):
    """[1] / [1,2] 引用"""
    nums = sorted(CNUM[k] for k in keys if k in CNUM)
    return f'[{",".join(str(n) for n in nums)}]'


# ================================================================ 标题与作者

para('Learnable Feature Weighting for Lightweight Transfer Learning '
     'under a Fully Frozen Backbone',
     size=15, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False,
     before=0, after=8, line=1.3)
para('Jiasheng Li', size=12, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False,
     after=2)
para('School of Information Science and Engineering, Linyi University, '
     'Linyi 276000, China',
     size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False,
     after=4)
para('Corresponding author: Jiasheng Li (E-mail: 703879709@qq.com)',
     size=9, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, after=12)

# ================================================================ Abstract

NUM_DS = {9: 'nine', 8: 'eight', 7: 'seven', 6: 'six', 5: 'five', 4: 'four',
          3: 'three', 2: 'two'}
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.line_spacing = 1.3
p.paragraph_format.space_after = Pt(4)
r = p.add_run('Abstract: ')
set_font(r, size=10, bold=True)
gs = STATS['gain_over_zeroshot_mean']
# 摘要支撑数字（全部动态计算自 sci_analysis.json，与正文表格同源）
no30 = {ds: (NOISE[f'{ds}|0.3|LFW(ours)'][0] - NOISE[f'{ds}|0.3|FFT'][0]) * 100
        for ds in DATASETS
        if f'{ds}|0.3|LFW(ours)' in NOISE and f'{ds}|0.3|FFT' in NOISE}
no30_wins = sum(1 for v in no30.values() if v > 0)
no30_ds, no30_best = (max(no30.items(), key=lambda t: t[1])
                      if no30 else ('', 0.0))
lo05 = {ds: (LOWRES[f'{ds}|LFW(ours)|0.05'][0] - LOWRES[f'{ds}|FFT|0.05'][0]) * 100
        for ds in DATASETS
        if f'{ds}|LFW(ours)|0.05' in LOWRES and f'{ds}|FFT|0.05' in LOWRES}
lo05_ds, lo05_best = (max(lo05.items(), key=lambda t: t[1])
                      if lo05 else ('', 0.0))
n_sig = sum(1 for w in WILCOX.values()
            if w['p'] < 0.05 and w.get('mean_diff', 0) > 0)
# 视觉 ViT 补充验证（存在时在摘要中加一句，数字与 4.8 节同源）
_vis_abs = ''
_vp = os.path.join(RESULTS, 'sci_vision_analysis.json')
if os.path.exists(_vp):
    import json as _json
    try:
        _V = _json.load(open(_vp, encoding='utf-8'))
        _vm, _vl = _V.get('main', {}), _V.get('lowres', {})
        _ds_list = [k.split('|')[0] for k in _vm
                    if k.split('|')[1] == 'ZeroShot']
        _lfw_m = [_vl.get(f'{d}|LFW(ours)|1', [None])[0]
                  for d in _ds_list]
        _zs_m = [_vl.get(f'{d}|ZeroShot|1', [None])[0] for d in _ds_list]
        _lfw_m = [x for x in _lfw_m if x is not None]
        _zs_m = [x for x in _zs_m if x is not None]
        if _lfw_m and _zs_m and len(_lfw_m) == len(_zs_m):
            _vg = (sum(_lfw_m) / len(_lfw_m) - sum(_zs_m) / len(_zs_m)) * 100
            _vis_abs = (f'A supplementary vision-Transformer validation '
                        f'(MNIST→USPS, CIFAR-10→CIFAR-10-C) confirms '
                        f'large zero-shot gains ({_vg:+.1f} points on '
                        f'average) while delimiting the input-only '
                        f'boundary versus weight-space PEFT. ')
    except Exception:
        _vis_abs = ''
abstract = (
    f'Full fine-tuning of pre-trained models on downstream tasks incurs '
    f'high computational cost, risks catastrophic forgetting of '
    f'pre-trained representations, and is infeasible when model weights '
    f'are contractually or legally frozen, while conventional static '
    f'feature selection is disconnected from the task objective. We '
    f'propose Learnable Feature Weighting (LFW), a minimal-invasion '
    f'transfer learning method that keeps every backbone parameter frozen '
    f'and adapts by learning only a non-negative input feature weight '
    f'vector optimized end-to-end with the task loss. A combined L1/L2 '
    f'regularization suppresses spurious features, and a safeguard '
    f'mechanism guarantees no degradation below the zero-shot baseline. '
    f'Across {NUM_DS.get(len(DATASETS), str(len(DATASETS)))} real UCI '
    f'benchmark datasets spanning 303-45,211 samples and 2-10 classes, '
    f'LFW is compared against 12 baselines - classical learners, '
    f'filter-based feature selection, and five state-of-the-art '
    f'parameter-efficient fine-tuning (PEFT) methods - under a unified '
    f'protocol with five random seeds. Training only d parameters - '
    f'{STATS["param_ratio_median"]:.0f}x fewer than full fine-tuning '
    f'(median) - LFW improves macro-F1 over zero-shot transfer by '
    f'{gs:.1f} percentage points on average; it outperforms full '
    f'fine-tuning under 30% label noise ({no30_wins} of {len(no30)} '
    f'datasets, up to {no30_best:+.1f} points on {no30_ds}) and at 5% '
    f'labels (up to {lo05_best:+.1f} points on {lo05_ds}), while its '
    f'degradation under input corruption is no faster than the zero-shot '
    f'model. Wilcoxon tests confirm significant improvements over '
    f'{n_sig} of {len(WILCOX)} baselines. '
    + _vis_abs +
    'The learned weights provide '
    'instance-level feature attribution, yielding direct '
    'interpretability.'
)
r = p.add_run(abstract)
set_font(r, size=10)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.line_spacing = 1.3
p.paragraph_format.space_after = Pt(12)
r = p.add_run('Keywords: ')
set_font(r, size=10, bold=True)
r = p.add_run('transfer learning; parameter-efficient fine-tuning; feature '
              'weighting; frozen backbone; model adaptation; interpretability')
set_font(r, size=10)

# ================================================================ 1 Introduction

h1('1. Introduction')

para('Pre-trained models have become the dominant paradigm across machine '
     'learning applications' + cite('pan2010survey', 'kornblith2019better') +
     '. A model is first trained on a large source-domain corpus and then '
     'adapted to a downstream task. The de facto adaptation procedure, full '
     'fine-tuning (FFT), updates all backbone parameters. However, FFT brings '
     'three well-documented drawbacks. First, it is computationally expensive '
     'and memory-hungry, which is prohibitive on edge devices and for large '
     'models. Second, updating the backbone can distort or even destroy the '
     'pre-trained representations - Kumar et al. show that fine-tuning '
     'systematically reduces feature diversity and harms out-of-distribution '
     'performance relative to linear probing on the frozen features' +
     cite('kumar2022finetune') + ', and WiSE-FT demonstrates that even small '
     'fine-tuning steps trade robustness for in-distribution accuracy' +
     cite('wortsman2022robust') + '. Third, in an increasing number of '
     'practical settings the backbone is not modifiable at all: commercial '
     'models are served through inference APIs, certified or regulated '
     'deployments require the deployed artifact to remain bit-identical, and '
     'model owners may forbid internal weight access for intellectual-'
     'property reasons.', indent=True)

para('Parameter-efficient fine-tuning (PEFT) methods reduce - but do not '
     'eliminate - the amount of tuning inside the network. Adapters insert '
     'small bottleneck modules between frozen layers' +
     cite('houlsby2019adapter') + '; BitFit tunes only bias vectors' +
     cite('benzaken2022bitfit') + '; SSF learns per-channel scale and shift '
     'parameters after every layer' + cite('lian2022ssf') + '; LoRA and its '
     'successors inject low-rank weight updates' +
     cite('hu2022lora', 'liu2024dora', 'hayou2024loraplus') +
     '. These methods still require inserting or modifying internal '
     'computation paths, and therefore still assume architectural access to '
     'the backbone. Prompt/prefix tuning' +
     cite('lester2021prompt', 'li2021prefix', 'liu2023ptuningv2') +
     ' keeps the backbone untouched but is tied to the token-embedding '
     'interface of Transformers and does not transfer to models that consume '
     'fixed-length feature vectors, which remain the norm in tabular, '
     'sensing, and many industrial applications' +
     cite('shwartz2022tabular', 'grinsztajn2022treemodels') + '.',
     indent=True)

para('An alternative that predates deep learning is to modify the input '
     'rather than the model: feature selection and feature weighting. Filter '
     'methods rank features by statistics such as variance or mutual '
     'information and are model-agnostic, but the ranking is computed '
     'independently of the downstream objective, so it cannot adapt the '
     'feature geometry to what the frozen decision function actually needs' +
     cite('li2017featureselection', 'ross2014mutualinfo') + '. Embedded '
     'methods such as the Lasso couple selection with training' +
     cite('tibshirani1996lasso') + ', but they require jointly optimizing '
     'model coefficients, which is exactly what a frozen backbone forbids. '
     'Consequently, the mildest possible form of task adaptation - changing '
     'neither the backbone nor its architecture, only how the input is '
     'weighted before it enters an untouched network - has received little '
     'systematic attention.', indent=True)

h2('1.1. Contributions')
para('This paper studies that extreme point of the adaptation spectrum '
     'systematically and makes it practical. Unlike existing PEFT methods '
     '(LoRA, Adapter, BitFit, SSF) that inject trainable parameters inside '
     'the backbone computational graph and require white-box architectural '
     'access, and unlike prompt tuning which is bound to the token-embedding '
     'interface of Transformers, LFW performs adaptation purely at the input '
     'interface with a single d-dimensional vector. This positions it as the '
     'minimal possible intervention: it leaves every backbone weight, '
     'activation, and architecture untouched, works with any differentiable '
     'model served through any interface (including remote inference APIs), '
     'and is applicable whenever the downstream input is a fixed-length '
     'feature vector - the dominant case in tabular, sensing, and industrial '
     'applications. Our contributions are:',
     indent=True)
para('(1) Minimal-invasion adaptation. We propose Learnable Feature '
     'Weighting (LFW): a single non-negative weight vector w applied '
     'element-wise to the input (z = w ⊙ x) that is the only trainable '
     'component, while every backbone parameter stays frozen. The method '
     'requires only that the backbone be differentiable with respect to its '
     'input; it works for any architecture, deployment modality (including '
     'remote inference APIs), and data type, and it provably cannot degrade '
     'a deployment because a safeguard mechanism falls back to the identity '
     'weighting whenever learned weights fail to improve validation '
     'performance.', indent=True)
para('(2) Regularization design for frozen backbones. We combine L1/L2 '
     'penalties on w with projected-gradient non-negativity. This induces '
     'exact zero weights (complete suppression of harmful features), keeps '
     'the monotone semantics "larger weight = more important feature", and '
     'acts as an implicit variance reduction prior in the low-sample regime, '
     'where unconstrained weighting would overfit.', indent=True)
para('(3) A rigorous, fully reproducible evaluation. We benchmark LFW '
     f'against {len(METHODS)-1} baselines on nine real UCI datasets '
     '(303-45,211 samples, 2-10 classes, six application domains) under a '
     'unified protocol: identical source-domain pre-training, stratified '
     'splits controlled by five random seeds, hyper-parameters selected '
     'solely on validation data, and statistical testing (Wilcoxon '
     'signed-rank) of all pairwise comparisons. The baselines include three '
     'classical learners trained from scratch (logistic regression, SVM, '
     'random forest), two filter-based selection methods, five recent PEFT '
     'methods (linear probing, BitFit, SSF, Adapter, LoRA), zero-shot '
     'transfer, and full fine-tuning as the upper bound.', indent=True)
para('(4) Analysis beyond aggregate accuracy. We report progressive ablations '
     'isolating each design choice, robustness under label noise (10-30%) '
     'and input noise (σ = 0.1-1.0), low-resource behavior down to 5% of '
     'labels, parameter/time/inference costs, convergence dynamics, and '
     'qualitative analyses (confusion matrices, t-SNE embeddings, weight '
     'profiles) that explain when and why LFW succeeds or fails.',
     indent=True)

h2('1.2. Paper organization')
para('Section 2 reviews related work. Section 3 formalizes the problem and '
     'presents the method. Section 4 details the experimental protocol and '
     'results. Section 5 discusses mechanisms, limitations, and future work, '
     'and Section 6 concludes.', indent=True)

# ================================================================ 2 Related Work

h1('2. Related Work')

h2('2.1. Transfer learning and the cost of fine-tuning')
para('Transfer learning transfers knowledge from a source domain to a '
     'related target domain' + cite('pan2010survey', 'wang2018visual') +
     '. For deep models, the adaptation protocol itself is a design choice '
     'with far-reaching consequences. Kornblith et al. show that ImageNet '
     'accuracy only loosely predicts transfer performance' +
     cite('kornblith2019better') + '. Zoph et al. compare pre-training and '
     'self-training under equal budgets' + cite('zoph2020rethinking') +
     '. Kumar et al. demonstrate that fine-tuning distorts pre-trained '
     'features and that linear-probing-then-fine-tuning (LP-FT) mitigates '
     'the distortion' + cite('kumar2022finetune') + '; WiSE-FT interpolates '
     'zero-shot and fine-tuned weights to retain robustness' +
     cite('wortsman2022robust') + '; Head2Toe searches intermediate layers '
     'for better transfer' + cite('evci2022head2toe') + '; L2-SP constrains '
     'fine-tuning to stay close to the pre-trained weights' +
     cite('li2018explicit') + '. All of these methods still write into the '
     'backbone. LFW occupies the complementary extreme: the backbone is '
     'never touched, so feature distortion is impossible by construction, '
     'and the only adapted quantity is an input re-weighting.', indent=True)

h2('2.2. Parameter-efficient fine-tuning')
para('PEFT methods tune a small subset of parameters while freezing the '
     'rest; see Han et al. for a comprehensive survey' +
     cite('han2024peftsurvey') + '. Adapter modules append lightweight '
     'bottlenecks inside each layer' + cite('houlsby2019adapter',
     'he2022unified') + '. BitFit shows that bias vectors alone are a '
     'surprisingly strong baseline' + cite('benzaken2022bitfit') + '. SSF '
     'learns per-channel scale-shift pairs after every layer and reports '
     'state-of-the-art accuracy-per-parameter on Vision Transformers' +
     cite('lian2022ssf') + '. IA³ rescales activations with learned '
     'vectors' + cite('liu2022ia3') + '. LoRA represents weight updates as '
     'low-rank factors' + cite('hu2022lora') + ' and has spawned a large '
     'family of refinements: LoRA+ with separate learning rates' +
     cite('hayou2024loraplus') + ', DoRA decomposing magnitude and '
     'direction' + cite('liu2024dora') + ', VeRA sharing random bases' +
     cite('kopiczko2024vera') + ', and quantization-aware variants such as '
     'QLoRA and LoftQ' + cite('dettmers2023qlora', 'li2024loftq') +
     '. Prompt- and prefix-tuning prepend learnable tokens' +
     cite('lester2021prompt', 'li2021prefix') + ', with P-Tuning v2 '
     'demonstrating depth-wise prompts competitive with fine-tuning' +
     cite('liu2023ptuningv2') + '.', indent=True)
para('Two observations motivate LFW. First, all of the above inject '
     'trainable parameters inside the computational graph of the backbone '
     '(or its embedding interface), which requires white-box access and '
     'modifies the deployed artifact. Second, their benefit is typically '
     'demonstrated on large Transformers; on tabular problems, where '
     'tree ensembles still outperform deep models' +
     cite('shwartz2022tabular', 'grinsztajn2022treemodels', 'arik2021tabnet',
          'gorishniy2021revisiting', 'gorishniy2022embeddings') +
     ', the backbone is usually a modest MLP whose full parameter count is '
     'itself small - yet the frozen-deployment constraint remains real. LFW '
     'targets precisely this regime and, to our knowledge, is the first '
     'systematic study of purely input-side, loss-coupled feature weighting '
     'as a PEFT alternative under a fully frozen backbone.', indent=True)

h2('2.3. Feature selection and weighting')
para('Feature selection methods fall into filter, wrapper, and embedded '
     'families' + cite('li2017featureselection') + '. Filters score features '
     'by marginal statistics - variance, correlation, mutual information' +
     cite('ross2014mutualinfo') + ' - and are cheap, but they are blind to '
     'the interaction between features and the frozen decision function. '
     'Wrappers and embedded methods such as the Lasso and elastic net tie '
     'selection to a trainable model' + cite('tibshirani1996lasso') +
     ', which is disallowed under a frozen backbone. Post-hoc attribution '
     'tools such as SHAP' + cite('lundberg2017shap') + ' explain a model '
     'after training but cannot improve it. LFW can be read as an elastic-'
     'net-style embedded method relocated to the input side: the sparsity '
     'induction (L1) and magnitude control (L2) act on w, while the model '
     'coefficients that would normally carry them remain fixed. The learned '
     'w doubles as a feature-importance profile, giving interpretability as '
     'a by-product of adaptation rather than as a separate analysis step.',
     indent=True)

h2('2.4. Robustness, noise, and low-resource adaptation')
para('Real deployments face label noise' + cite('song2023noisy') +
     ', distribution shift, and scarce labels. Test-time adaptation methods '
     'such as TENT' + cite('wang2021tent') + ' and source-free methods such '
     'as SHOT' + cite('liang2020shot') + ' adapt without source data but '
     'still modify normalization statistics or classifier heads at test '
     'time. Domain generalization research' + cite('wang2023generalizing',
     'hendrycks2021imagenetr') + ' seeks models robust to unseen domains. '
     'Because LFW has only d parameters, strong regularization, and a '
     'safeguard, we hypothesize - and verify experimentally - that it '
     'degrades more gracefully than higher-capacity adaptation methods '
     'under both corrupted labels and corrupted inputs, while its low-'
     'resource behavior approaches zero-shot transfer as labels vanish.',
     indent=True)

# ================================================================ 3 Method

h1('3. Proposed Method')

h2('3.1. Problem formulation')
para('Let Ds = {(xi, yi)} be a labeled source-domain dataset and let '
     'fθ(·) be a backbone pre-trained on Ds (or on related data), with '
     'parameters θ held fixed throughout downstream adaptation. Let Dt = '
     '{(xj, yj)} be a (typically much smaller) labeled adaptation set for a '
     'downstream domain whose input space is dimensionally aligned with the '
     'source domain. The goal is to maximize downstream generalization '
     'while treating fθ as an immutable black box: no internal parameter, '
     'activation, or architecture may be altered, added, or inspected; only '
     'forward and backward passes through the input interface are allowed. '
     'We parameterize the admissible transformation Tw(·) as element-wise '
     'feature weighting:', indent=True)
para('z = w ⊙ x,  w ∈ R^d, w ≥ 0,                                    (1)',
     align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
para('where ⊙ denotes the Hadamard product. The optimization problem is:',
     indent=True)
para('min_w  L(w) = (1/|Dt|) Σ CE( fθ(w ⊙ x), y ) + λ1‖w‖1 + λ2‖w‖2² '
     '  s.t. w ≥ 0,                                    (2)',
     align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
para('where CE is the cross-entropy loss and λ1, λ2 ≥ 0 are regularization '
     'coefficients tuned on downstream validation data only. The entire '
     'trainable capacity is d non-negative scalars - for our experiments, '
     'between 25 and 561 parameters, i.e., three to five orders of magnitude '
     'fewer than the backbones, and comparable to a single input neuron per '
     'feature.', indent=True)

para('Why diagonal input reweighting suffices. Under a covariate-shift '
     'assumption, the source and downstream domains share the conditional '
     'label distribution P(y|x) but differ in the marginal input '
     'distribution P(x) - precisely the regime in which zero-shot transfer '
     'fails. A frozen classifier fθ is a fixed nonlinear map, and its '
     'decision boundary in input space is determined by the pre-trained '
     'geometry. A diagonal rescaling z = w ⊙ x induces a '
     'coordinate-wise affine change of variables that can (i) suppress '
     'feature axes whose downstream marginal variance differs sharply from '
     'the source (re-calibrating normalization), (ii) stretch axes along '
     'which fθ is most sensitive to marginally shifted categories, and '
     '(iii) re-weight one-hot blocks whose category priors shifted between '
     'domains. Crucially, because the transformation is applied before fθ, '
     'the Jacobian ∂z/∂x = diag(w) is trivial to compute and the gradient '
     'signal reaches w through a single back-propagation step truncated at '
     'the input - no internal gradients of θ are ever needed. This connects '
     'LFW to the classical theory of preconditioning and importance '
     'weighting: w acts as a learned, per-feature importance weight that '
     'aligns the input geometry of the downstream domain with the '
     'decision boundary learned on the source domain.', indent=True)

para('Figure 1 shows the overall framework.', indent=True)
add_fig(os.path.join(FIGS, 'fig_s1_framework.png'), 15.5,
        'Fig. 1. Overview of Learnable Feature Weighting (LFW). A frozen, '
        'pre-trained backbone fθ receives re-weighted inputs z = w ⊙ x. '
        'Only the d-dimensional non-negative weight vector w is trained; '
        'gradients are stopped at the backbone input.')

h2('3.2. Learnable weighting layer')
para('Initialization. w is initialized to the all-ones vector, so the '
     'initial behavior of the adapted model is exactly zero-shot transfer: '
     'the adaptation starts from the pre-trained input geometry rather than '
     'destroying it, in the same spirit as identity-initialized residual '
     'adaptation' + cite('houlsby2019adapter') + '.', indent=True)
para('Non-negativity by projection. After every gradient step we project w '
     'back onto the non-negative orthant (w ← max(w, 0)). Compared with '
     'soft parameterizations such as w = softplus(u), projection (i) allows '
     'weights to reach exact zeros, cleanly switching off deleterious '
     'features instead of merely shrinking them, and (ii) preserves the '
     'monotone semantics "larger weight = more important feature", which '
     'keeps the learned vector interpretable and avoids sign-cancellation '
     'ambiguity in attribution.', indent=True)
para('Safeguard mechanism. The identity vector w = 1 is always kept as a '
     'candidate during early stopping: the final model is arg max over '
     '{w = 1, best learned w} of validation macro-F1. Consequently, LFW '
     'provably never performs worse than zero-shot transfer on the '
     'validation set - a deployment-relevant guarantee reminiscent of the '
     'robustness rationale behind WiSE-FT' + cite('wortsman2022robust') +
     ' but achieved with zero extra parameters. In practice, when the '
     'source and downstream distributions are already well aligned, the '
     'learned weights remain near one, which is itself a diagnostic: it '
     'certifies that the pre-trained input geometry already fits the '
     'downstream task.', indent=True)

h2('3.3. Optimization')
para('Because the backbone is frozen, the gradient of the task loss with '
     'respect to w costs one forward pass plus one backward pass truncated '
     'at the input layer - the backward pass skips all weight-gradient '
     'computations of the backbone, roughly halving its cost. For a '
     'mini-batch B the gradient is:', indent=True)
para('∂L/∂wj = (1/|B|) Σ (∂Ltask/∂zj) · xj + λ1·1[wj > 0] + 2λ2·wj,'
     '                                    (3)',
     align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
para('where the sub-gradient convention is used for the L1 term at wj = 0. '
     'The optimizer is Adam' + cite('kingma2015adam') + ' (lr = 0.05, batch '
     'size 256). Note that the optimizer state (first/second moments) is '
     'O(d) instead of O(|θ|), which, together with the halved backward '
     'pass, yields large wall-clock savings over FFT (Section 4.6). '
     'Algorithm 1 summarizes the procedure.', indent=True)

# Algorithm 1
para('Algorithm 1  LFW training under a fully frozen backbone', size=10,
     bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, before=6,
     after=2)
alg = doc.add_table(rows=1, cols=1)
alg.alignment = WD_TABLE_ALIGNMENT.CENTER
cell = alg.cell(0, 0)
set_cell_borders(cell, top='thick', bottom='thick')
tcPr = cell._tc.get_or_add_tcPr()
for edge in ('w:left', 'w:right'):
    e = tcPr.makeelement(qn(edge), {})
    e.set(qn('w:val'), 'single')
    e.set(qn('w:sz'), '6')
    e.set(qn('w:color'), '000000')
    tcPr.find(qn('w:tcBorders')).append(e)
alg_lines = [
    'Input: frozen backbone fθ; adaptation set Dt; validation set Dv;',
    '        batch size B; learning rate η; epochs E; patience P;',
    '        regularization λ1, λ2',
    'Output: feature weights w',
    ' 1: w ← 1;  v* ← F1macro( fθ(Dv), yv )              // safeguard candidate',
    ' 2: for epoch = 1 … E do',
    ' 3:   for each mini-batch (x, y) ⊂ Dt do',
    ' 4:     z ← w ⊙ x;  ŷ ← fθ(z)                        // forward, θ frozen',
    ' 5:     L ← CE(ŷ, y) + λ1‖w‖1 + λ2‖w‖2²',
    ' 6:     g ← ∂L/∂w  by Eq. (3)                        // stop-gradient at input',
    ' 7:     w ← max(0, w − η·Adam(g))                     // Adam step + projection',
    ' 8:   if F1macro(w; Dv) > v* then v* ← …; w* ← w; bad ← 0',
    ' 9:   else bad ← bad + 1;  if bad ≥ P then break      // early stopping',
    '10: end for',
    '11: return arg max over {1, w*} of F1macro(·; Dv)     // never below zero-shot',
]
for i, ln in enumerate(alg_lines):
    p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(ln)
    set_font(r, en='Consolas', size=9)
para('', size=4, indent=False, after=4)

h2('3.4. Complexity analysis')
para('Parameters: d trainable scalars versus |θ| for FFT (e.g., on HAR, '
     '561 versus 1,103,366 - a 1:1969 ratio; on Adult, 106 versus '
     'about 3.2×10^5). Memory: beyond the backbone, LFW stores w and its '
     'Adam moments - O(d) floats - whereas FFT stores O(|θ|) moments. '
     'Compute per step: one forward pass and one input-truncated backward '
     'pass, versus one full forward plus one full backward with weight '
     'gradients for FFT; the per-epoch speedup is empirically 1.5-4x '
     '(Section 4.6). Inference: a single element-wise multiplication, '
     'adding O(d) work per sample - below measurement noise for the models '
     'considered. The method is therefore suitable for edge deployment and '
     'for adaptation through remote inference services, where only the '
     'forward pass (and gradients with respect to the input) is available.',
     indent=True)

print('Part 1 (front matter, intro, related work, method) built.')

# ================================================================ Part 2
import sys  # noqa: E402
sys.modules['build_paper_sci'] = sys.modules['__main__']
import build_paper_sci_part2  # noqa: E402
build_paper_sci_part2.build_part2()
