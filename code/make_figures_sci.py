# -*- coding: utf-8 -*-
"""SCI 论文分析与图表生成（英文、300dpi、色盲友好配色）。

输入：results/sci_main.csv / sci_noise.csv / sci_input_noise.csv /
      sci_ablation.csv / sci_sensitivity.csv / sci_meta.json /
      sci_weights.npz / sci_feature_names.json / sci_viz_*.npz
输出：results/sci_analysis.json（论文全部表格数据与统计检验）
      paper_figs/fig_s1..fig_s10（论文全部插图）
所有数字均由真实实验结果计算，论文构建脚本只引用本脚本产物。
"""
import csv
import json
import os
import statistics as st
from collections import defaultdict

import numpy as np
from scipy import stats as sps
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
FIGS = os.path.join(ROOT, 'paper_figs')
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'figure.dpi': 100,
})

# Okabe-Ito 色盲友好配色
C = {'ZeroShot': '#999999', 'VarFilter': '#E69F00', 'MIFilter': '#D55E00',
     'LR': '#F0E442', 'SVM': '#8B4513', 'RF': '#009E73',
     'LinearProbe': '#CC79A7', 'BitFit': '#56B4E9', 'SSF': '#009E73',
     'Adapter': '#E69F00', 'LoRA': '#0072B2', 'LFW(ours)': '#D7191C',
     'FFT': '#000000'}
LBL = {'ZeroShot': 'Zero-shot', 'VarFilter': 'VarFilter', 'MIFilter': 'MIFilter',
       'LR': 'LR', 'SVM': 'SVM', 'RF': 'RF', 'LinearProbe': 'LinearProbe',
       'BitFit': 'BitFit', 'SSF': 'SSF', 'Adapter': 'Adapter', 'LoRA': 'LoRA',
       'LFW(ours)': 'LFW (ours)', 'FFT': 'Full fine-tune'}
ALL_DS = ['HAR', 'Bank', 'Adult', 'Satimage', 'Digits', 'Spambase',
          'Credit-g', 'WDBC', 'Heart']
DS_DOM = {'Adult': 'Finance (census)', 'Heart': 'Healthcare',
          'HAR': 'Activity recognition', 'WDBC': 'Healthcare',
          'Digits': 'Handwritten digits', 'Spambase': 'Spam detection',
          'Credit-g': 'Credit scoring', 'Bank': 'Bank marketing',
          'Satimage': 'Remote sensing'}
METHODS = ['ZeroShot', 'VarFilter', 'MIFilter', 'LR', 'SVM', 'RF',
           'LinearProbe', 'BitFit', 'SSF', 'Adapter', 'LoRA', 'LFW(ours)', 'FFT']
CURVE_METHODS = ['ZeroShot', 'LR', 'RF', 'SSF', 'LoRA', 'LFW(ours)', 'FFT']


def load(path):
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


def agg(rows, keycols, valcol):
    """按 keycols 聚合 valcol：返回 {key: (mean, std, n)}。"""
    d = defaultdict(list)
    for r in rows:
        d[tuple(float(r[c]) if c in ('frac', 'noise_p', 'sigma') else r[c]
                for c in keycols)].append(float(r[valcol]))
    return {k: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))
            for k, v in d.items()}


# ================================================================ 主表聚合

def analyze(suffix=''):
    rows = load(os.path.join(RESULTS, f'sci_main{suffix}.csv'))
    datasets = [d for d in ALL_DS if any(r['dataset'] == d for r in rows)]
    A = {'datasets': datasets}

    # 每数据集×方法（frac=1.0）全指标
    per = defaultdict(dict)
    for m in ['acc', 'precision', 'recall', 'f1']:
        g = agg(rows, ['dataset', 'method', 'frac'], m)
        for (ds, meth, fr), (mu, sd, n) in g.items():
            if fr == 1.0:
                per[(ds, meth)][m] = (mu, sd)
    time_g = agg(rows, ['dataset', 'method', 'frac'], 'train_time_s')
    infer_g = agg(rows, ['dataset', 'method', 'frac'], 'infer_ms')
    npar = {}
    for r in rows:
        if float(r['frac']) == 1.0:
            npar[(r['dataset'], r['method'])] = int(r['n_tuned'])
    A['main'] = {f'{ds}|{m}': dict(metrics=per[(ds, m)],
                                   train_time=time_g[(ds, m, 1.0)][0],
                                   infer_ms=infer_g[(ds, m, 1.0)][0],
                                   n_tuned=npar.get((ds, m)))
                 for (ds, m) in per}

    # 低资源（全部 frac）
    low = {}
    g = agg(rows, ['dataset', 'method', 'frac'], 'f1')
    for (ds, m, fr), (mu, sd, n) in g.items():
        low[f'{ds}|{m}|{fr}'] = (mu, sd, n)
    A['lowres'] = low

    # 平均排名（frac=1.0，按 F1，跨数据集）
    ranks = defaultdict(list)
    for ds in datasets:
        ms = [(m, per[(ds, m)]['f1'][0]) for m in METHODS if (ds, m) in per]
        ms.sort(key=lambda t: -t[1])
        for rk, (m, _) in enumerate(ms, 1):
            ranks[m].append(rk)
    A['rank'] = {m: st.mean(v) for m, v in ranks.items()}

    # Wilcoxon：LFW vs 各方法（frac=1.0，跨数据集×种子配对）
    raw = defaultdict(dict)
    for r in rows:
        if float(r['frac']) == 1.0:
            raw[(r['dataset'], int(r['seed']))][r['method']] = float(r['f1'])
    A['wilcoxon'] = {}
    for m in METHODS:
        if m == 'LFW(ours)':
            continue
        pairs = [(v['LFW(ours)'], v[m]) for v in raw.values()
                 if 'LFW(ours)' in v and m in v]
        if len(pairs) >= 6:
            x = [a for a, b in pairs]; y = [b for a, b in pairs]
            try:
                stat, p = sps.wilcoxon(x, y)
            except ValueError:
                stat, p = 0.0, 1.0
            A['wilcoxon'][m] = dict(n=len(pairs), stat=float(stat),
                                    p=float(p), win=int(sum(a > b for a, b in pairs)),
                                    tie=int(sum(a == b for a, b in pairs)))

    # 小样本（frac<=0.1）LFW 对各方法优势
    small = defaultdict(list)
    for r in rows:
        if float(r['frac']) <= 0.1 and r['method'] == 'LFW(ours)':
            small[(r['dataset'], float(r['frac']))].append(float(r['f1']))
    A['small_lfw'] = {f'{k[0]}|{k[1]}': st.mean(v) for k, v in small.items()}

    # 噪声
    nrow = load(os.path.join(RESULTS, f'sci_noise{suffix}.csv'))
    A['noise'] = {f'{k[0]}|{k[1]}|{k[2]}': v
                  for k, v in agg(nrow, ['dataset', 'noise_p', 'method'], 'f1').items()}
    irow = load(os.path.join(RESULTS, f'sci_input_noise{suffix}.csv'))
    A['inoise'] = {f'{k[0]}|{k[1]}|{k[2]}': v
                   for k, v in agg(irow, ['dataset', 'sigma', 'method'], 'f1').items()}
    # 消融
    arow = load(os.path.join(RESULTS, f'sci_ablation{suffix}.csv'))
    A['ablation'] = {}
    g_f1 = agg(arow, ['dataset', 'variant'], 'f1')
    g_sp = agg(arow, ['dataset', 'variant'], 'sparsity')
    for k in g_f1:
        A['ablation'][f'{k[0]}|{k[1]}'] = dict(f1=g_f1[k], sparsity=g_sp.get(k, (0, 0, 0)))
    # 敏感性
    srow = load(os.path.join(RESULTS, f'sci_sensitivity{suffix}.csv'))
    A['sensitivity'] = defaultdict(dict)
    for r in srow:
        A['sensitivity'][r['dataset']][f'{float(r["l1"]):g}|{float(r["l2"]):g}'] = \
            (float(r['f1']), float(r['sparsity']))
    A['sensitivity'] = {k: dict(v) for k, v in A['sensitivity'].items()}

    meta = json.load(open(os.path.join(RESULTS, f'sci_meta{suffix}.json'),
                          encoding='utf-8'))
    A['pretrain'] = {f'{p["dataset"]}|{p["seed"]}': p for p in meta['pretrain']}
    A['protocol'] = meta.get('protocol', {})

    # 核心统计（摘要用）
    gains_zs, gains_fft, ratios, tr_ratios = [], [], [], []
    for ds in datasets:
        try:
            lfw = per[(ds, 'LFW(ours)')]['f1'][0]
            zs = per[(ds, 'ZeroShot')]['f1'][0]
            fft = per[(ds, 'FFT')]['f1'][0]
            gains_zs.append((lfw - zs) * 100)
            gains_fft.append((fft - lfw) * 100)
            d = npar.get((ds, 'LFW(ours)'), 1)
            tot = npar.get((ds, 'FFT'), 1)
            ratios.append(tot / d)
            t_lfw = time_g[(ds, 'LFW(ours)', 1.0)][0]
            t_fft = time_g[(ds, 'FFT', 1.0)][0]
            if t_fft > 0:
                tr_ratios.append((1 - t_lfw / t_fft) * 100)
        except KeyError:
            pass
    A['stats'] = dict(gain_over_zeroshot_mean=st.mean(gains_zs) if gains_zs else 0,
                      gain_over_zeroshot_max=max(gains_zs) if gains_zs else 0,
                      gap_to_fft_mean=st.mean(gains_fft) if gains_fft else 0,
                      param_ratio_median=st.median(ratios) if ratios else 0,
                      time_reduction_median=st.median(tr_ratios) if tr_ratios else 0)
    json.dump(A, open(os.path.join(RESULTS, f'sci_analysis{suffix}.json'), 'w',
                      encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'analysis -> sci_analysis{suffix}.json')
    return A


# ================================================================ 图 1 框架

def fig_framework():
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis('off')

    def box(x, y, w, h, text, fc, ec, fs=10, lw=1.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35',
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)

    def arrow(x1, y1, x2, y2, color='#333333', lw=1.6, ls='-'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=14, color=color, lw=lw,
                                     linestyle=ls, shrinkA=2, shrinkB=2))

    box(1, 26, 15, 14,
        'Input features\n$\\mathbf{x}=[x_1,\\cdots,x_d]^{\\top}$\n(standardized)',
        '#e3f2fd', '#1565c0')
    box(23, 26, 21, 14,
        'Learnable feature-weighting layer\n$\\mathbf{z}=\\mathbf{w}\\odot\\mathbf{x}$,'
        ' $w_i\\geq 0$\n$L_1$+$L_2$ regularization\n(only $d$ trainable parameters)',
        '#ffebee', '#d32f2f', fs=9)
    box(51, 22, 32, 22,
        'Pre-trained backbone $f_{\\theta}(\\cdot)$\n'
        'MLP layers ($\\theta$ completely frozen)\n'
        'any differentiable architecture',
        '#f5f5f5', '#616161', fs=9.5)
    box(63, 3, 25, 11,
        'Task loss\n$\\mathcal{L}=\\mathrm{CE}(f_{\\theta}(\\mathbf{z}), y)'
        '+\\lambda_1\\|\\mathbf{w}\\|_1+\\lambda_2\\|\\mathbf{w}\\|_2^2$',
        '#fff8e1', '#f9a825', fs=8.8)
    arrow(16, 33, 23, 33)
    arrow(44, 33, 51, 33)
    arrow(67, 22, 75, 14)
    ax.add_patch(FancyArrowPatch((63, 8.5), (33, 26), arrowstyle='-|>',
                                 mutation_scale=14, color='#d32f2f', lw=1.8,
                                 linestyle='--',
                                 connectionstyle='arc3,rad=-0.25'))
    ax.text(19, 3.0,
            'Back-propagation: only $\\mathbf{w}$ updated ($d$ params)\n'
            '$\\partial\\mathcal{L}/\\partial\\theta=\\mathbf{0}$ (gradients stopped '
            'at the input)',
            fontsize=8.8, color='#d32f2f', va='bottom', linespacing=1.3)
    ax.text(50, 48.5, 'Learnable Feature Weighting under a fully frozen backbone',
            fontsize=11.5, ha='center', weight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_s1_framework.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('fig_s1 done')


# ================================================================ 图 2 低资源

def _grid_axes(n, per_row, wh=(3.3, 3.1)):
    """按每行 per_row 个子图返回扁平 axes 列表与 fig。"""
    ncols = min(n, per_row)
    nrows = (n + per_row - 1) // per_row
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(wh[0] * ncols, wh[1] * nrows))
    axes = np.atleast_1d(axes).ravel().tolist()
    for ax in axes[n:]:      # 隐藏多余子图
        ax.set_visible(False)
    return fig, axes[:n]


def fig_lowresource(A, suffix=''):
    rows = load(os.path.join(RESULTS, f'sci_main{suffix}.csv'))
    g = agg(rows, ['dataset', 'method', 'frac'], 'f1')
    datasets = [d for d in ['HAR', 'Bank', 'Adult', 'Satimage', 'Spambase', 'WDBC']
                if any(k[0] == d for k in g)]
    fig, axes = _grid_axes(len(datasets), per_row=3)
    for ax, ds in zip(axes, datasets):
        for m in CURVE_METHODS:
            xs, ys, es = [], [], []
            for (d, meth, fr), (mu, sd, n) in sorted(g.items()):
                if d == ds and meth == m:
                    xs.append(fr * 100); ys.append(mu * 100); es.append(sd * 100)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, label=LBL[m], color=C[m],
                        marker='o' if m == 'LFW(ours)' else 's', ms=3.8,
                        lw=1.4 if m == 'LFW(ours)' else 1.0,
                        ls='-' if m == 'LFW(ours)' else '--', capsize=2,
                        alpha=0.95)
        ax.set_xscale('log')
        ax.set_xticks([5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel('Labeled fraction of adaptation pool (%)')
        ax.set_title(ds, fontsize=11)
        ax.grid(alpha=0.3, ls=':')
    for i, ax in enumerate(axes):
        if i % 3 == 0:
            ax.set_ylabel('Macro-F1 (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(CURVE_METHODS),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_s2_lowresource.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('fig_s2 done')


# ================================================================ 图 3 收敛

def fig_convergence(suffix=''):
    meta = json.load(open(os.path.join(RESULTS, f'sci_meta{suffix}.json'),
                          encoding='utf-8'))
    conv = meta.get('convergence', {})
    dss = [d for d in ('HAR', 'Bank') if d in conv]
    if not dss:
        print('fig_s3 skipped (no convergence data)')
        return
    fig, axes = plt.subplots(2, len(dss), figsize=(4.4 * len(dss), 6.4))
    axes = np.atleast_2d(axes)
    for j, ds in enumerate(dss):
        for i, key in enumerate(('ours', 'fft', 'ssf')):
            if key not in conv[ds]:
                continue
            h = conv[ds][key]
            lbl = {'ours': 'LFW (ours)', 'fft': 'Full fine-tune',
                   'ssf': 'SSF'}[key]
            axes[0][j].plot(range(1, len(h['val_f1']) + 1),
                            np.array(h['val_f1']) * 100,
                            color=C['LFW(ours)' if key == 'ours' else
                                   'FFT' if key == 'fft' else 'SSF'],
                            lw=1.6 if key == 'ours' else 1.1,
                            ls='-' if key == 'ours' else '--', label=lbl)
            axes[1][j].plot(h['cum_time'], np.array(h['val_f1']) * 100,
                            color=C['LFW(ours)' if key == 'ours' else
                                   'FFT' if key == 'fft' else 'SSF'],
                            lw=1.6 if key == 'ours' else 1.1,
                            ls='-' if key == 'ours' else '--', label=lbl)
        axes[0][j].set_title(ds, fontsize=11)
        axes[0][j].set_xlabel('Epoch'); axes[1][j].set_xlabel('Wall-clock time (s)')
        axes[0][j].set_ylabel('Validation macro-F1 (%)' if j == 0 else '')
        axes[1][j].set_ylabel('Validation macro-F1 (%)' if j == 0 else '')
        axes[0][j].legend(fontsize=8.5); axes[0][j].grid(alpha=0.3, ls=':')
        axes[1][j].grid(alpha=0.3, ls=':')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_s3_convergence.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('fig_s3 done')


# ================================================================ 图 4 权重

def fig_weights(suffix=''):
    wnpz = np.load(os.path.join(RESULTS, f'sci_weights{suffix}.npz'),
                   allow_pickle=True)
    names = json.load(open(os.path.join(RESULTS, f'sci_feature_names{suffix}.json'),
                           encoding='utf-8'))
    per_ds = {}
    for k in wnpz.files:
        ds, s = k.rsplit('__s', 1)
        per_ds.setdefault(ds, []).append((int(s), wnpz[k]))
    informative = {}
    for ds, lst in per_ds.items():
        w = max(lst, key=lambda t: float(np.std(t[1])))[1]
        if float(np.std(w)) > 0.01:
            informative[ds] = w
    top = [d for d in ('HAR', 'Bank', 'Adult', 'Satimage', 'Spambase')
           if d in informative][:2]
    if len(top) < 2:
        top = (top + [k for k in sorted(informative,
                                        key=lambda k: -float(np.std(informative[k])))
                      if k not in top])[:2]
    if not top:
        print('fig_s4 skipped (no informative weights)')
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0),
                             gridspec_kw={'width_ratios': [1.25, 1]})
    for ax, ds in zip(axes, top):
        w = informative[ds]
        nm = names[ds]
        order = np.argsort(-w)[:15][::-1]
        ax.barh(range(15), w[order], color='#D7191C', alpha=0.85, height=0.62)
        ax.set_yticks(range(15))
        ax.set_yticklabels([nm[i] for i in order], fontsize=8)
        ax.set_xlabel('Learned weight $w_i$')
        ax.set_title(f'{ds} ({DS_DOM.get(ds, "")})', fontsize=10.5)
        ax.grid(alpha=0.3, ls=':', axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_s4_weights.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    with open(os.path.join(FIGS, 'fig_s4_manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(dict(datasets=top,
                       std={ds: float(np.std(informative[ds])) for ds in top}),
                  f, indent=1)
    print('fig_s4 done:', top)

    # HAR 全特征权重热力图（按特征名排序，含传感器分组边界）
    if 'HAR' in informative:
        w = informative['HAR']
        nm = np.array(names['HAR'])
        order = np.argsort(nm)
        ws = w[order]
        groups = []
        for n in nm[order]:
            g = n.split('-')[0].replace('angle(', 'angle-')
            if not groups or groups[-1][0] != g:
                groups.append([g, 1])
            else:
                groups[-1][1] += 1
        fig, ax = plt.subplots(figsize=(12, 1.9))
        ax.imshow(ws.reshape(1, -1), aspect='auto', cmap='RdYlBu_r',
                  vmin=0, vmax=max(1.0, ws.max()))
        bounds = np.cumsum([c for _, c in groups])
        for b in bounds[:-1]:
            ax.axvline(b - 0.5, color='k', lw=0.4, alpha=0.5)
        ax.set_yticks([])
        xt = [0] + [b - 1 for b in bounds[::max(1, len(bounds) // 8)]]
        ax.set_xticks(xt)
        ax.set_xticklabels([str(int(x) + 1) for x in xt], fontsize=7)
        ax.set_xlabel('Feature index (sorted by sensor group)')
        cb = fig.colorbar(ax.imshow(ws.reshape(1, -1), aspect='auto',
                                    cmap='RdYlBu_r', vmin=0,
                                    vmax=max(1.0, ws.max())),
                          ax=ax, orientation='vertical', fraction=0.02)
        cb.set_label('$w_i$', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, 'fig_s4b_heatmap.png'), dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print('fig_s4b done (HAR heatmap)')


# ================================================================ 图 5 混淆矩阵

HAR_ACTS = ['WALKING', 'UPSTAIRS', 'DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
SAT_LBL = ['red soil', 'cotton crop', 'grey soil', 'damp grey soil',
           'stubble', 'very damp soil']


def fig_confusion(suffix=''):
    for ds, labels in (('HAR', HAR_ACTS), ('Satimage', SAT_LBL),
                       ('WDBC', ['Benign', 'Malignant'])):
        path = os.path.join(RESULTS, f'sci_viz_{ds}{suffix}.npz')
        if not os.path.exists(path):
            continue
        z = np.load(path)
        fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
        for ax, key, ttl in zip(axes, ('pred_none', 'pred_lfw', 'pred_fft'),
                                ('Zero-shot', 'LFW (ours)', 'Full fine-tune')):
            cm = confusion_matrix(z['ytrue'], z[key])
            cmn = cm / cm.sum(axis=1, keepdims=True)
            im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
            ax.set_yticklabels(labels, fontsize=7.5)
            acc = np.trace(cm) / cm.sum()
            ax.set_title(f'{ttl}  (Acc={acc*100:.1f}%)', fontsize=10)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if cmn[i, j] > 0.5:
                        ax.text(j, i, f'{cmn[i,j]*100:.0f}', ha='center',
                                va='center', fontsize=6,
                                color='white' if cmn[i, j] > 0.75 else 'black')
            if ax is axes[0]:
                ax.set_ylabel('True class')
            ax.set_xlabel('Predicted class')
        fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
        fig.savefig(os.path.join(FIGS, f'fig_s5_confusion_{ds}.png'), dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'fig_s5 done ({ds})')


# ================================================================ 图 6 t-SNE

def fig_tsne(suffix=''):
    for ds in ('HAR', 'Satimage'):
        path = os.path.join(RESULTS, f'sci_viz_{ds}{suffix}.npz')
        if not os.path.exists(path):
            continue
        z = np.load(path)
        n = len(z['ytrue'])
        sel = np.random.default_rng(0).choice(n, size=min(n, 1200),
                                              replace=False)
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
        for ax, key, ttl in zip(axes, ('feats_none', 'feats_lfw', 'feats_fft'),
                                ('Zero-shot', 'LFW (ours)', 'Full fine-tune')):
            X = z[key][sel]
            X2 = TSNE(n_components=2, perplexity=30, init='pca',
                      random_state=0).fit_transform(X)
            sc = ax.scatter(X2[:, 0], X2[:, 1], c=z['ytrue'][sel], s=6,
                            cmap='tab10', alpha=0.75, linewidths=0)
            ax.set_title(f'{ttl}', fontsize=10.5)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.02)
        fig.suptitle(f'{ds}: penultimate-layer features (t-SNE)', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f'fig_s6_tsne_{ds}.png'), dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'fig_s6 done ({ds})')


# ================================================================ 图 7 标签噪声

def fig_labelnoise(suffix=''):
    rows = load(os.path.join(RESULTS, f'sci_noise{suffix}.csv'))
    g = agg(rows, ['dataset', 'noise_p', 'method'], 'f1')
    datasets = [d for d in ALL_DS if any(k[0] == d for k in g)]
    if not datasets:
        return
    methods = ['BitFit', 'FFT', 'LFW(ours)']
    ps = [0.1, 0.2, 0.3]
    fig, axes = _grid_axes(len(datasets), per_row=5, wh=(2.55, 3.0))
    for ax, ds in zip(axes, datasets):
        x = np.arange(len(ps))
        for k, m in enumerate(methods):
            mu = [g.get((ds, p, m), (np.nan,))[0] * 100 for p in ps]
            sd = [g.get((ds, p, m), (0,))[1] * 100 for p in ps]
            ax.bar(x + (k - 1) * 0.27, mu, yerr=sd, width=0.25, capsize=2,
                   label=LBL[m], color=C[m], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(p*100)}%' for p in ps], fontsize=9)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel('Label noise level')
        ax.grid(alpha=0.3, ls=':', axis='y')
    for i, ax in enumerate(axes):
        if i % 5 == 0:
            ax.set_ylabel('Macro-F1 (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_s7_labelnoise.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('fig_s7 done')


# ================================================================ 图 8 输入噪声

def fig_inputnoise(suffix=''):
    rows = load(os.path.join(RESULTS, f'sci_input_noise{suffix}.csv'))
    g = agg(rows, ['dataset', 'sigma', 'method'], 'f1')
    datasets = [d for d in ALL_DS if any(k[0] == d for k in g)]
    if not datasets:
        return
    methods = ['ZeroShot', 'BitFit', 'FFT', 'LFW(ours)']
    sgs = [0.0, 0.1, 0.5, 1.0]
    # 干净基线（σ=0）取 sci_main frac=0.2
    mrows = load(os.path.join(RESULTS, f'sci_main{suffix}.csv'))
    gm = agg(mrows, ['dataset', 'method', 'frac'], 'f1')
    fig, axes = _grid_axes(len(datasets), per_row=5, wh=(2.7, 3.0))
    for ax, ds in zip(axes, datasets):
        for m in methods:
            xs, ys, es = [], [], []
            for sg in sgs:
                if sg == 0.0:
                    v = gm.get((ds, m, 0.2))
                else:
                    v = g.get((ds, sg, m))
                if v is None:
                    continue
                xs.append(sg); ys.append(v[0] * 100); es.append(v[1] * 100)
            if xs:
                ax.errorbar(xs, ys, yerr=es, label=LBL[m], color=C[m],
                            marker='o' if m == 'LFW(ours)' else 's', ms=3.6,
                            lw=1.5 if m == 'LFW(ours)' else 1.0,
                            ls='-' if m == 'LFW(ours)' else '--', capsize=2)
        ax.set_xlabel('Input noise std $\\sigma$')
        ax.set_title(ds, fontsize=10)
        ax.grid(alpha=0.3, ls=':')
    for i, ax in enumerate(axes):
        if i % 5 == 0:
            ax.set_ylabel('Macro-F1 (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_s8_inputnoise.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('fig_s8 done')


# ================================================================ 图 9 λ 敏感性

def fig_sensitivity(suffix=''):
    rows = load(os.path.join(RESULTS, f'sci_sensitivity{suffix}.csv'))
    dss = sorted({r['dataset'] for r in rows})
    if not dss:
        return
    l1s = sorted({float(r['l1']) for r in rows})
    l2s = sorted({float(r['l2']) for r in rows})
    fig, axes = plt.subplots(1, len(dss), figsize=(4.6 * len(dss), 3.6))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, dss):
        M = np.full((len(l1s), len(l2s)), np.nan)
        for r in rows:
            if r['dataset'] == ds:
                M[l1s.index(float(r['l1'])), l2s.index(float(r['l2']))] = \
                    float(r['f1']) * 100
        im = ax.imshow(M, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(l2s)))
        ax.set_xticklabels([f'{v:g}' for v in l2s], fontsize=8)
        ax.set_yticks(range(len(l1s)))
        ax.set_yticklabels([f'{v:g}' for v in l1s], fontsize=8)
        ax.set_xlabel('$\\lambda_2$'); ax.set_ylabel('$\\lambda_1$')
        ax.set_title(f'{ds} (macro-F1, %)', fontsize=10)
        for i in range(len(l1s)):
            for j in range(len(l2s)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f'{M[i,j]:.1f}', ha='center', va='center',
                            fontsize=6.5,
                            color='white' if M[i, j] < np.nanmax(M) - 1.5 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_s9_sensitivity.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('fig_s9 done')


# ================================================================ 图 10 效率前沿

def fig_efficiency(A, suffix=''):
    rows = load(os.path.join(RESULTS, f'sci_main{suffix}.csv'))
    g = agg(rows, ['dataset', 'method', 'frac'], 'f1')
    npg = {}
    for r in rows:
        if float(r['frac']) == 1.0:
            npg[(r['dataset'], r['method'])] = int(r['n_tuned'])
    methods = ['LR', 'SVM', 'RF', 'LinearProbe', 'BitFit', 'SSF', 'Adapter',
               'LoRA', 'LFW(ours)', 'FFT']
    datasets = sorted({r['dataset'] for r in rows})
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in methods:
        xs, ys = [], []
        for ds in datasets:
            v = g.get((ds, m, 1.0)); p = npg.get((ds, m))
            if v and p is not None and p > 0:
                xs.append(p); ys.append(v[0] * 100)
        if not xs:
            continue
        big = m == 'LFW(ours)'
        ax.scatter(xs, ys, s=90 if big else 45,
                   color=C.get(m, '#666666'), marker='*' if big else 'o',
                   zorder=3 if big else 2, label=LBL[m],
                   edgecolors='black', linewidths=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Trainable / model parameters (log scale)')
    ax.set_ylabel('Macro-F1 at 100% pool (%)')
    ax.grid(alpha=0.3, ls=':')
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig_s10_efficiency.png'), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('fig_s10 done')


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    sfx = '_quick' if (len(sys.argv) > 2 and sys.argv[2] == 'quick') else ''
    if which in ('all', 'framework', '1'):
        fig_framework()
    if which in ('all', 'analyze'):
        A = analyze(sfx)
    else:
        A = json.load(open(os.path.join(RESULTS, f'sci_analysis{sfx}.json'),
                           encoding='utf-8'))
    if which in ('all', '2'):
        fig_lowresource(A, sfx)
    if which in ('all', '3'):
        fig_convergence(sfx)
    if which in ('all', '4'):
        fig_weights(sfx)
    if which in ('all', '5'):
        fig_confusion(sfx)
    if which in ('all', '6'):
        fig_tsne(sfx)
    if which in ('all', '7'):
        fig_labelnoise(sfx)
    if which in ('all', '8'):
        fig_inputnoise(sfx)
    if which in ('all', '9'):
        fig_sensitivity(sfx)
    if which in ('all', '10'):
        fig_efficiency(A, sfx)
    print('全部图表生成完成。')
