# -*- coding: utf-8 -*-
"""生成 Elsevier (Neurocomputing) 投稿包：
1. manuscript: 连续行号 + 双倍行距 + CRediT 声明（内容与主稿完全一致）
2. title_page: 标题页（作者信息占位）
3. highlights: 5 条要点（每条 <=85 字符）
4. cover_letter: 投稿信（docx + txt，数字全部动态取自 results/*.json）
"""
import csv
import json
import os
import statistics as st

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'SCI_Paper_Learnable_Feature_Weighting.docx')
OUT = os.path.join(ROOT, 'submission_package')
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(OUT, exist_ok=True)

JOURNAL = 'Neurocomputing'
DATE = '7 September 2026'
REPO = 'https://github.com/luo1chen/lfw-frozen-backbone'
TITLE = ('Learnable Feature Weighting for Lightweight Transfer '
         'Learning under a Fully Frozen Backbone')
AUTHOR = 'Jiasheng Li'          # 李佳胜（拼音署名）
AFFILIATION = ('School of Information Science and Engineering, '
               'Linyi University, Linyi 276000, China')
EMAIL = '703879709@qq.com'
ORCID = ''                     # 无 ORCID，留空

# ------------------------------------------------ 数字（与论文同源计算）
A = json.load(open(os.path.join(RESULTS, 'sci_analysis.json'),
                   encoding='utf-8'))
V = json.load(open(os.path.join(RESULTS, 'sci_vision_analysis.json'),
                   encoding='utf-8'))
STATS = A['stats']
WILCOX = A['wilcoxon']

# mean_diff 与 build_paper_sci.py 完全同口径：sci_main.csv frac=1.0 配对差
_pair = {}
with open(os.path.join(RESULTS, 'sci_main.csv'), newline='',
          encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if float(r['frac']) == 1.0:
            _pair[(r['dataset'], int(r['seed']), r['method'])] = float(r['f1'])
for m, w in WILCOX.items():
    diffs = [v - _pair[(ds, sd, m)]
             for (ds, sd, mt), v in _pair.items()
             if mt == 'LFW(ours)' and (ds, sd, m) in _pair]
    w['mean_diff'] = (sum(diffs) / len(diffs) * 100.0) if diffs else 0.0

n_sig = sum(1 for w in WILCOX.values()
            if w['p'] < 0.05 and w.get('mean_diff', 0) > 0)
gain = STATS['gain_over_zeroshot_mean']
ratio = STATS['param_ratio_median']
NOISE = A['noise']
datasets = A['datasets']
no30 = [ds for ds in datasets
        if f'{ds}|0.3|LFW(ours)' in NOISE and f'{ds}|0.3|FFT' in NOISE
        and NOISE[f'{ds}|0.3|LFW(ours)'][0] > NOISE[f'{ds}|0.3|FFT'][0]]
no30_wins = len(no30)

# 视觉数字（与 4.8 节同源）
vmain = V['main']
vds = [k.split('|')[0] for k in vmain if k.split('|')[1] == 'ZeroShot']
v_gain = (st.mean(vmain[f'{d}|LFW(ours)']['f1'][0] for d in vds)
          - st.mean(vmain[f'{d}|ZeroShot']['f1'][0] for d in vds)) * 100
v_f005 = V['summary']['lfw_gain_over_best_peft_f005']

print(f'gain={gain:.1f}pp ratio=1:{ratio:.0f} n_sig={n_sig}/{len(WILCOX)} '
      f'noise30={no30_wins}/{len(datasets)} '
      f'vit_gain={v_gain:+.1f}pp vit_f005={v_f005*100:+.1f}pp')

# ------------------------------------------------ 1) Manuscript
doc = Document(SRC)


def add_line_numbers(document):
    """每个 section 加连续行号（符合 Elsevier 审稿要求）。"""
    for sect in document.sections:
        sectPr = sect._sectPr
        if sectPr.find(qn('w:lnNumType')) is not None:
            continue
        ln = OxmlElement('w:lnNumType')
        ln.set(qn('w:countBy'), '1')
        ln.set(qn('w:restart'), 'continuous')
        anchor = None
        for tag in ('w:pgNumType', 'w:cols', 'w:formProt', 'w:vAlign',
                    'w:titlePg', 'w:textDirection', 'w:docGrid',
                    'w:printerSettings', 'w:sectPrChange'):
            el = sectPr.find(qn(tag))
            if el is not None:
                anchor = el
                break
        if anchor is not None:
            anchor.addprevious(ln)
        else:
            sectPr.append(ln)


add_line_numbers(doc)

# 正文双倍行距（doc.paragraphs 不含表格内段落，表格保持单倍）
for p in doc.paragraphs:
    p.paragraph_format.line_spacing = 2.0

# 在 Conflict of Interest 前插入 CRediT（Elsevier 要求）
target = None
for p in doc.paragraphs:
    if p.text.strip() == 'Conflict of Interest':
        target = p
        break
if target is not None:
    h = target.insert_paragraph_before('')
    rh = h.add_run('CRediT authorship contribution statement')
    rh.bold = True
    rh.font.size = Pt(13)
    b = target.insert_paragraph_before('')
    b.paragraph_format.space_after = Pt(6)
    rb = b.add_run(
        f'{AUTHOR}: Conceptualization, Methodology, Software, '
        'Validation, Formal analysis, Investigation, Data curation, '
        'Writing - original draft, Writing - review & editing, '
        'Visualization.')
    rb.font.size = Pt(11)

manuscript_path = os.path.join(OUT, 'manuscript.docx')
doc.save(manuscript_path)
print('saved:', manuscript_path)

# ------------------------------------------------ 2) Title page
tp = Document()


def tp_para(text, size=11, bold=False, center=False, space=6):
    p = tp.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space)
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    return p


tp_para(TITLE, size=16, bold=True, center=True, space=12)
tp_para('')
tp_para(AUTHOR, size=13, bold=True, center=True)
tp_para(AFFILIATION, center=True)
tp_para('')
tp_para('Corresponding author:', bold=True, space=2)
tp_para(f'E-mail: {EMAIL}', space=2)
if ORCID:
    tp_para(f'ORCID: {ORCID}', space=12)
else:
    tp_para('', space=12)
tp_para(f'Manuscript type: Research article', space=2)
tp_para(f'Journal: {JOURNAL} (Elsevier)', space=2)
tp_para(f'Date: {DATE}', space=2)
tp_para(f'Word count (abstract): 239', space=2)
tp_para('Keywords: transfer learning; parameter-efficient fine-tuning; '
        'frozen backbone; feature weighting; tabular learning; '
        'vision Transformer', space=12)
title_path = os.path.join(OUT, 'title_page.docx')
tp.save(title_path)
print('saved:', title_path)

# ------------------------------------------------ 3) Highlights
HIGHLIGHTS = [
    'Input-only adaptation: one learnable non-negative weight per feature',
    'Backbone stays fully frozen, enabling API and compliance deployments',
    f'1:{ratio:.0f} fewer trainable parameters than full fine-tuning',
    f'+{gain:.1f} pp mean macro-F1 over zero-shot on nine UCI datasets',
    'ViT study delimits when input-only adaptation is sufficient',
]
for h in HIGHLIGHTS:
    assert len(h) <= 85, f'highlight too long ({len(h)}): {h}'
hl = Document()
hp = hl.add_paragraph()
rh = hp.add_run('Highlights')
rh.bold = True
rh.font.size = Pt(14)
for h in HIGHLIGHTS:
    p = hl.add_paragraph(style='List Bullet')
    r = p.add_run(h)
    r.font.size = Pt(11)
highlights_path = os.path.join(OUT, 'highlights.docx')
hl.save(highlights_path)
print('saved:', highlights_path)

# ------------------------------------------------ 4) Cover letter
cl_text = f"""{DATE}

The Editors
{JOURNAL}

Subject: Submission of the research article "{TITLE}"

Dear Editors,

We are pleased to submit our manuscript "{TITLE}" for consideration as a research article in {JOURNAL}.

The paper studies the minimal-invasion extreme of parameter-efficient transfer learning: the backbone remains fully frozen - architecturally untouched - and adaptation happens entirely in the input space, by learning a single non-negative multiplicative weight per input feature. This setting is motivated by deployment scenarios in which backbone updates are impossible or non-compliant: models served behind inference APIs, regulated medical or financial deployments that require model-integrity audits, and edge devices whose shipped weights must remain bit-identical. While LoRA, adapters, BitFit, and SSF achieve remarkable efficiency, they all require internal write access; we show that a simple, theoretically motivated diagonal re-weighting - {int(round(V["summary"]["lfw_params"])):,} parameters in our vision study and a 1:{ratio:.0f} parameter ratio versus full fine-tuning in the tabular setting - recovers a substantial part of the achievable improvement.

Key results: across nine UCI datasets, LFW improves mean macro-F1 by +{gain:.1f} pp over zero-shot transfer and is Wilcoxon-significantly better than {n_sig} of {len(WILCOX)} baselines (alpha = 0.05, n = 45); under 30% label noise it outperforms full fine-tuning on {no30_wins} of 9 datasets while training orders of magnitude fewer parameters; and a supplementary study on a compact vision Transformer (MNIST to USPS, CIFAR-10 to CIFAR-10-C) reports an average +{v_gain:.1f} pp gain over zero-shot and validates the formulation on the architecture family for which BitFit/SSF/Adapter/LoRA were designed.

We would like to highlight two aspects of the manuscript's rigor. First, all numbers derive from public datasets (UCI Machine Learning Repository; MNIST, USPS, CIFAR-10, CIFAR-10-C), five-seed protocols, and statistical tests; the complete code and raw per-seed results are openly available at {REPO} (MIT license), and a verification script re-checks every table cell against the raw CSVs. Second, we deliberately report the applicability boundary as observed rather than overstating generality: the vision study shows a residual gap to the strongest weight-space PEFT baseline at full label budgets, and a built-in safeguard reverts to zero-shot behavior when the learned weights do not improve validation macro-F1 - which also guarantees that input noise never degrades LFW below the zero-shot model. We believe this honest delineation of when input-only adaptation is sufficient (largely marginal shifts) versus when internal updates are required is itself a useful contribution for practitioners.

This manuscript is original, has not been published previously, and is not under consideration by any other journal. All authors have approved the submission and declare no conflicts of interest. This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

We believe the work fits the scope of {JOURNAL} in learning systems and efficient adaptation, and we hope it merits your consideration for review.

Sincerely,

{AUTHOR}
{AFFILIATION}
E-mail: {EMAIL}
On behalf of all authors
"""

cl = Document()
for line in cl_text.split('\n'):
    p = cl.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(line)
    r.font.size = Pt(11)
cover_path = os.path.join(OUT, 'cover_letter.docx')
cl.save(cover_path)
with open(os.path.join(OUT, 'cover_letter.txt'), 'w',
          encoding='utf-8') as f:
    f.write(cl_text)
print('saved:', cover_path)
print('saved: cover_letter.txt')
print('DONE - package in', OUT)
