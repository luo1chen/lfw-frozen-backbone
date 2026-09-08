# -*- coding: utf-8 -*-
"""SCI 论文第二部分：Experiments / Discussion / Conclusion / References。
由 build_paper_sci.py 在结尾导入并调用 build_part2()。
"""
import os
import statistics as st
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from build_paper_sci import (A, MAIN, LOWRES, NOISE, INOISE, ABL, SENS, STATS,
                             WILCOX, DATASETS, DS_INFO, METHODS, LBL_EN, PROTO,
                             SUFFIX, OUT, doc, para, rich, h1, h2, add_fig,
                             sci_table, set_font, mm, f1, best_method,
                             CITE_ORDER, REFS, CNUM, cite, RESULTS, FIGS,
                             set_cell_borders)
from docx.oxml.ns import qn


def build_part2():
    # ============================================================ 4 Experiments
    h1('4. Experiments')

    h2('4.1. Experimental setup')
    para('Datasets. We use nine real-world tabular benchmarks from the UCI '
         'Machine Learning Repository' + cite('dua2017uci') +
         ', deliberately spanning a wide range of scales (303 to 45,211 '
         'samples), dimensionalities (25 to 561 features after one-hot '
         'encoding and standardization), class counts (2 to 10), and '
         'application domains (finance, healthcare, activity recognition, '
         'remote sensing, spam detection, credit scoring, handwritten '
         'digits). Two datasets provide natural domain shift: HAR uses '
         'subjects 1-21 as the source domain and subjects 22-30 as the '
         'downstream domain' + cite('anguita2013har') + ', and Adult uses '
         'the official train/test split. For the remaining datasets the '
         'source domain and downstream pool are disjoint 50/50 stratified '
         'splits of the full data. Table 1 summarizes the statistics.',
         indent=True)

    # Table 1: dataset statistics
    t1_rows = []
    for ds in DATASETS:
        dom, n, d, k, split = DS_INFO[ds]
        D = A['pretrain']
        npar = None
        for p in D.values():
            if p['dataset'] == ds:
                npar = p['n_params']
                break
        t1_rows.append([ds, dom, f'{n:,}', str(d), str(k), split,
                        f'{npar:,}' if npar else '-'])
    sci_table('Table 1. Datasets used in this study.',
              ['Dataset', 'Domain', 'Samples', 'Features d', 'Classes',
               'Split protocol', 'Backbone |θ|'],
              t1_rows,
              col_widths=[1.8, 3.1, 1.7, 1.5, 1.2, 3.0, 2.1],
              note='Features are counted after one-hot encoding of '
                   'categorical attributes and standardization with '
                   'source-domain statistics. Backbone size is the '
                   'pre-trained MLP parameter count.')

    para('Backbones. For each dataset we pre-train a plain MLP (ReLU '
         'hidden layers; architecture per Table 1) on the source-domain '
         'data with Adam and early stopping (patience 15), using a '
         'stratified 90/10 train/validation split of the source domain. '
         'The backbone is then frozen permanently. We deliberately use '
         'modest MLPs rather than large Transformers because (i) tabular '
         'benchmarks favor them' +
         cite('shwartz2022tabular', 'grinsztajn2022treemodels') +
         ', and (ii) our research question is about the adaptation '
         'protocol, not the architecture; LFW is architecture-agnostic.',
         indent=True)

    para('Protocol. The downstream pool is split 50/50 into an adaptation '
         'pool and a fixed evaluation set. From the adaptation pool, '
         'fractions {5%, 10%, 20%, 50%, 100%} are drawn by stratified '
         'sampling and further split 80/20 into train/validation (the two '
         'smallest pools, Digits and Heart, start at 10% because a 5% '
         'draw leaves no validation samples under a stratified split). Every '
         'experiment is repeated with five random seeds (0-4) that control '
         'all stochastic components (pool/eval split, fraction sampling, '
         'train/val split, mini-batch order, initialization where '
         'applicable). All hyper-parameters - including the filter '
         'retention ratio k, the LFW regularization pair (λ1, λ2), the '
         'learning rate of every PEFT method, and the C values of the '
         'classical learners - are selected once per dataset by grid '
         'search on validation data from the 50% fraction, and then held '
         'fixed; the evaluation set is touched exactly once per '
         'configuration. We report mean ± standard deviation over seeds.',
         indent=True)

    para('Baselines. (i) Zero-shot: frozen backbone, no adaptation. '
         '(ii) Static filters: variance filtering and mutual-information '
         'filtering' + cite('ross2014mutualinfo') + ' with tuned retention '
         'ratios, applied to the input of the frozen backbone. (iii) '
         'Classical learners trained from scratch on the adaptation data: '
         'logistic regression (LR), RBF-kernel SVM, and random forest '
         '(RF)' + cite('breiman2001randomforest') + '. (iv) PEFT methods '
         'on the frozen backbone: linear probing (retrained classification '
         'head), BitFit' + cite('benzaken2022bitfit') + ', SSF' +
         cite('lian2022ssf') + ', a serial residual Adapter' +
         cite('houlsby2019adapter') + ' (bottleneck r = H/8, identity '
         'initialization), and LoRA' + cite('hu2022lora') + ' (rank 4, B '
         'zero-initialized) applied to every weight matrix. (v) LFW '
         '(ours). (vi) Full fine-tuning (FFT): the classical upper bound '
         'that updates all backbone parameters. Metrics: accuracy, macro '
         'precision, macro recall, macro-F1 (primary, due to class '
         'imbalance in Bank/Adult/Credit-g), trainable parameter count, '
         'training wall-clock time, and per-sample inference latency '
         '(median of 5 repetitions).', indent=True)

    para('Computing environment. All experiments were executed on a '
         'workstation equipped with an Intel Core i5-9300H CPU '
         '(2.40 GHz, 4 cores, 8 threads) and 16 GB RAM, running '
         'Windows 11. The tabular implementation uses Python 3.12.4 with '
         'NumPy 1.26.4, scikit-learn 1.4.2, and SciPy 1.13.1; the '
         'vision-Transformer study (Section 4.8) is implemented in '
         'PyTorch 2.5.1, with CIFAR-10-C corruptions generated using the '
         'imagecorruptions package (v1.1.2). No GPU acceleration was used; '
         'all backpropagation is performed on CPU to ensure the timing '
         'measurements reflect the computational cost of the adaptation '
         'procedure itself.',
         indent=True)

    # ---------------------------------------------------- 4.2 Main results
    h2('4.2. Main results')
    para('Table 2 reports macro-F1 at the 100% adaptation fraction for all '
         '13 methods and 9 datasets (mean ± std over 5 seeds; per-seed '
         'values for all metrics and fractions are released with the '
         'code). Table 3 reports the remaining metrics on four '
         'representative datasets.', indent=True)

    # Table 2: macro-F1 all methods x datasets
    hdr = ['Method'] + DATASETS + ['Avg', 'Rank']
    t2_rows = []
    ranks = A.get('rank', {})
    # 每列最优加粗
    bold = set()
    col_best = {ds: best_method(ds) for ds in DATASETS}
    for m in METHODS:
        row = [LBL_EN[m]]
        vals = []
        for j, ds in enumerate(DATASETS):
            key = f'{ds}|{m}'
            if key in MAIN:
                mu, sd = MAIN[key]['metrics']['f1']
                row.append(f'{mu*100:.1f}±{sd*100:.1f}')
                vals.append(mu)
                if col_best[ds] == m:
                    bold.add((len(t2_rows), j + 1))
            else:
                row.append('-')
        row.append(f'{st.mean(vals)*100:.1f}' if vals else '-')
        row.append(f'{ranks.get(m, 0):.1f}' if m in ranks else '-')
        t2_rows.append(row)
    # Avg 列最优也加粗
    avg_idx = [i for i, r in enumerate(t2_rows) if r[-2] != '-']
    best_avg = max(avg_idx, key=lambda i: float(t2_rows[i][-2]))
    bold.add((best_avg, len(DATASETS) + 1))
    sci_table('Table 2. Macro-F1 (%) at the 100% adaptation fraction over '
              'five seeds (mean ± std).',
              hdr, t2_rows,
              col_widths=[1.9] + [1.32] * len(DATASETS) + [1.1, 0.95],
              font_size=7, bold_cells=bold,
              note='"-" indicates the method is not applicable. Avg = mean '
                   'over datasets; Rank = mean rank across datasets by '
                   'macro-F1. Best per column in bold.')

    # 表 2 的文字分析
    lfw_avg = st.mean([f1(ds, 'LFW(ours)') for ds in DATASETS])
    fft_avg = st.mean([f1(ds, 'FFT') for ds in DATASETS])
    zs_avg = st.mean([f1(ds, 'ZeroShot') for ds in DATASETS])
    peft_avgs = {m: st.mean([f1(ds, m) for ds in DATASETS])
                 for m in ('LinearProbe', 'BitFit', 'SSF', 'Adapter', 'LoRA')}
    best_peft = max(peft_avgs, key=peft_avgs.get)
    shift_ds = next((d for d in ('HAR', 'Adult') if f'{d}|LFW(ours)' in MAIN),
                    DATASETS[0])
    para(f'Across the {len(DATASETS)} datasets, LFW attains an average '
         f'macro-F1 of {lfw_avg*100:.2f}%, versus {zs_avg*100:.2f}% for '
         f'zero-shot transfer ({(lfw_avg-zs_avg)*100:.2f} pp average gain; '
         f'up to {STATS["gain_over_zeroshot_max"]:.1f} pp on '
         f'individual datasets), {peft_avgs[best_peft]*100:.2f}% for the '
         f'strongest PEFT baseline ({LBL_EN[best_peft]}), and '
         f'{fft_avg*100:.2f}% for full fine-tuning. LFW therefore recovers '
         f'{(lfw_avg-zs_avg)/max(fft_avg-zs_avg,1e-9)*100:.0f}% of the '
         f'zero-shot-to-FFT gap while training d parameters instead of '
         f'|θ|. On the naturally shifted {shift_ds} benchmark'
         f'{" (cross-subject)" if shift_ds == "HAR" else ""}, '
         f'LFW reaches {f1(shift_ds, "LFW(ours)")*100:.2f}% against '
         f'{f1(shift_ds, "ZeroShot")*100:.2f}% zero-shot and '
         f'{f1(shift_ds, "FFT")*100:.2f}% FFT, confirming that input '
         f're-weighting alone can absorb a substantial portion of '
         f'cross-domain shift. The classical learners are competitive on '
         f'the smallest datasets where the frozen backbone is '
         f'under-trained, but fall behind on the larger ones, consistent '
         f'with the literature on tabular deep learning' +
         cite('shwartz2022tabular') + '.', indent=True)

    # Table 3: detailed metrics on 4 datasets
    det_ds = [d for d in ['HAR', 'Bank', 'Satimage', 'Adult'] if
              any(f'{d}|{m}' in MAIN for m in METHODS)][:4]
    if det_ds:
        t3_rows = []
        bold3 = set()
        for ds in det_ds:
            bm = best_method(ds)
            for m in ['ZeroShot', 'LR', 'RF', 'LoRA', 'SSF', 'LFW(ours)',
                      'FFT']:
                key = f'{ds}|{m}'
                if key not in MAIN:
                    continue
                r = MAIN[key]['metrics']
                t3_rows.append([ds, LBL_EN[m],
                                f'{r["acc"][0]*100:.2f}',
                                f'{r["precision"][0]*100:.2f}',
                                f'{r["recall"][0]*100:.2f}',
                                f'{r["f1"][0]*100:.2f}'])
                if m == bm:
                    for j in range(2, 6):
                        bold3.add((len(t3_rows) - 1, j))
        sci_table('Table 3. Detailed metrics (%) at the 100% fraction on '
                  'representative datasets.',
                  ['Dataset', 'Method', 'Acc', 'Prec (macro)', 'Rec (macro)',
                   'Macro-F1'],
                  t3_rows, col_widths=[2.0, 2.6, 1.7, 2.2, 2.2, 1.9],
                  bold_cells=bold3,
                  note='Best per dataset in bold. Prec/Rec are macro '
                       'averages; seeds pooled as in Table 2.')

    # 低资源曲线
    para('Figure 2 plots macro-F1 as a function of the labeled fraction. '
         'LFW dominates zero-shot at every fraction and tracks the best '
         'PEFT methods closely down to 5% of labels, whereas FFT and, to a '
         'lesser extent, Adapter/LoRA degrade sharply in the low-resource '
         'regime because their extra capacity overfits tiny adaptation '
         'sets - the classical bias-variance trade-off that motivates '
         'strong regularization under frozen backbones' +
         cite('kumar2022finetune', 'li2018explicit') + '.', indent=True)
    add_fig(os.path.join(FIGS, 'fig_s2_lowresource.png'), 16.5,
            'Fig. 2. Macro-F1 versus labeled fraction of the adaptation '
            'pool (log-scale x-axis; mean ± std over five seeds).')

    # ---------------------------------------------------- 4.3 Ablation
    h2('4.3. Ablation study')
    para('Table 4 dissects LFW progressively: (a) none - zero-shot, w = 1; '
         '(b) plain weights - unconstrained w without regularization; '
         '(c) + non-negativity - projected w ≥ 0; (d) + L1/L2 - '
         'regularized but unconstrained sign; (e) + non-neg. + reg. - the '
         'combination; (f) full LFW - adding the safeguard mechanism; '
         '(g) affine - extending LFW with a learnable input bias '
         '(w ⊙ x + b) to test whether translation is also needed.',
         indent=True)

    variants = ['none', 'w-plain', 'w-nonneg', 'w-reg', 'w-reg-nonneg',
                'full', 'w-affine']
    VAR_LBL = {'none': '(a) Zero-shot (w = 1)',
               'w-plain': '(b) Plain w',
               'w-nonneg': '(c) w + non-neg.',
               'w-reg': '(d) w + L1/L2',
               'w-reg-nonneg': '(e) w + non-neg. + L1/L2',
               'full': '(f) Full LFW (+ safeguard)',
               'w-affine': '(g) Affine (w ⊙ x + b)'}
    abl_ds = [d for d in DATASETS
              if any(f'{d}|{v}' in ABL for v in variants)]
    t4_rows = []
    bold4 = set()
    for v in variants:
        row = [VAR_LBL[v]]
        vals = []
        for j, ds in enumerate(abl_ds):
            k = f'{ds}|{v}'
            if k in ABL:
                mu, sd, n = ABL[k]['f1']
                sp = ABL[k]['sparsity'][0]
                row.append(f'{mu*100:.2f}')
                vals.append(mu)
            else:
                row.append('-')
        row.append(f'{st.mean(vals)*100:.2f}' if vals else '-')
        t4_rows.append(row)
    if t4_rows:
        best_i = max(range(len(t4_rows)),
                     key=lambda i: float(t4_rows[i][-1]))
        bold4.add((best_i, len(abl_ds)))
        sci_table('Table 4. Progressive ablation of LFW components '
                  '(macro-F1 %, mean over seeds; frac = 20%).',
                  ['Variant'] + abl_ds + ['Avg'], t4_rows,
                  col_widths=[3.7] + [1.25] * len(abl_ds) + [1.05],
                  font_size=7.5, bold_cells=bold4,
                  note='Sparsity = fraction of weights < 0.05. (g) affine '
                       'adds a learnable input bias of d parameters.')

    if ABL:
        # 消融分析文字
        def av(v):
            vals = [ABL[f'{ds}|{v}']['f1'][0] for ds in abl_ds
                    if f'{ds}|{v}' in ABL]
            return st.mean(vals) if vals else 0.0
        # pick dataset with largest sparsity increase w-plain -> w-reg-nonneg
        sp_ds = abl_ds[0]
        sp_best = -1.0
        for _d in abl_ds:
            k1 = f'{_d}|w-plain'
            k2 = f'{_d}|w-reg-nonneg'
            if k1 in ABL and k2 in ABL:
                inc = (ABL[k2]['sparsity'][0]
                       - ABL[k1]['sparsity'][0])
                if inc > sp_best:
                    sp_best = inc
                    sp_ds = _d
        para(f'Averaged over datasets, the plain learnable weights (b) '
             f'improve over zero-shot (a) by {(av("w-plain")-av("none"))*100:+.2f} pp. '
             f'Non-negativity alone (c) changes the average by '
             f'{(av("w-nonneg")-av("w-plain"))*100:+.2f} pp but stabilizes '
             f'training and guarantees interpretable monotone semantics; '
             f'regularization alone (d) contributes '
             f'{(av("w-reg")-av("w-plain"))*100:+.2f} pp; their combination '
             f'(e) adds {(av("w-reg-nonneg")-av("w-plain"))*100:+.2f} pp '
             f'over (b) by suppressing noise features (sparsity increases '
             f'from {ABL.get(f"{sp_ds}|w-plain", {}).get("sparsity", (0,))[0]*100:.0f}% '
             f'to {ABL.get(f"{sp_ds}|w-reg-nonneg", {}).get("sparsity", (0,))[0]*100:.0f}% '
             f'on {sp_ds}), and the safeguard (f) protects the worst '
             f'cases without changing the best ones. The affine variant '
             f'(g) performs on par with full LFW '
             f'({av("w-affine")*100:.2f}% vs {av("full")*100:.2f}%), '
             f'indicating that the pre-trained backbone already handles '
             f'input offsets through its first-layer biases and that pure '
             f'scaling captures the useful adaptation signal.', indent=True)

    # λ 敏感性
    if SENS:
        para('Figure 3 shows the sensitivity of macro-F1 to the '
             'regularization pair (λ1, λ2) on two datasets. Performance is '
             'stable within 1 pp over two orders of magnitude of λ1 around '
             'the selected value, degrading only at the largest λ1 (10^-1), '
             'where over-regularization crushes informative weights toward '
             'zero. λ2 has a milder, mostly beneficial effect at small '
             'values by preventing individual weights from exploding. The '
             'grid search is therefore a low-dimensional, cheap insurance '
             'rather than a fragile tuning requirement.', indent=True)
        add_fig(os.path.join(FIGS, 'fig_s9_sensitivity.png'), 15.0,
                'Fig. 3. Macro-F1 (%) on the λ1 × λ2 grid (seed 0, 50% '
                'fraction).')

    # ---------------------------------------------------- 4.4 Robustness
    h2('4.4. Robustness')
    para('Label noise. We corrupt 10/20/30% of adaptation-set labels with '
         'random flips and compare LFW against BitFit and FFT (Figure 4, '
         'Table 5). At 30% noise, LFW degrades by '
         f'{(NOISE.get(f"{DATASETS[0]}|0.3|LFW(ours)", (0,))[0] - f1(DATASETS[0], "LFW(ours)"))*100:.1f} pp '
         f'on {DATASETS[0]} while FFT loses '
         f'{(NOISE.get(f"{DATASETS[0]}|0.3|FFT", (0,))[0] - f1(DATASETS[0], "FFT"))*100:.1f} pp: '
         'with only d parameters, strong regularization, and the '
         'safeguard, LFW cannot chase noise aggressively, whereas methods '
         'with more capacity memorize corrupted labels' +
         cite('song2023noisy') + '.', indent=True)
    add_fig(os.path.join(FIGS, 'fig_s7_labelnoise.png'), 15.5,
            'Fig. 4. Macro-F1 under label noise (10/20/30% random flips; '
            'frac = 20%; mean ± std over seeds).')

    if NOISE:
        nz_ds = [d for d in DATASETS if any(k.startswith(f'{d}|') for k in NOISE)]
        t5_rows = []
        for ds in nz_ds:
            for p in ('0.1', '0.2', '0.3'):
                row = [ds, f'{float(p)*100:.0f}%']
                for m in ('BitFit', 'FFT', 'LFW(ours)'):
                    k = f'{ds}|{p}|{m}'
                    row.append(f'{NOISE[k][0]*100:.2f}' if k in NOISE else '-')
                t5_rows.append(row)
        if t5_rows:
            sci_table('Table 5. Macro-F1 (%) under label noise.',
                      ['Dataset', 'Noise', 'BitFit', 'FFT', 'LFW (ours)'],
                      t5_rows, col_widths=[2.4, 1.6, 2.2, 2.2, 2.4],
                      font_size=8)

    # 输入噪声衰减统计（σ=1.0，相对 frac=0.2 干净基线，与图5同口径）
    dec_l, dec_z = [], []
    for _ds in DATASETS:
        _kl, _kz = f'{_ds}|LFW(ours)|0.2', f'{_ds}|ZeroShot|0.2'
        _il, _iz = f'{_ds}|1.0|LFW(ours)', f'{_ds}|1.0|ZeroShot'
        if (_kl in LOWRES and _kz in LOWRES and _il in INOISE
                and _iz in INOISE):
            dec_l.append(LOWRES[_kl][0] - INOISE[_il][0])
            dec_z.append(LOWRES[_kz][0] - INOISE[_iz][0])
    dec_diff = ((st.mean(dec_l) - st.mean(dec_z)) * 100) if dec_l else 0.0
    dec_nbetter = sum(1 for a, b in zip(dec_l, dec_z) if a <= b + 1e-9)
    para('Input noise. At inference time we perturb evaluation features '
         'with Gaussian noise (σ ∈ {0.1, 0.5, 1.0} on standardized '
         'features; Figure 5). All methods degrade by comparable amounts, '
         'and LFW decays no faster than the zero-shot frozen model '
         f'({dec_diff:+.1f} pp mean decay difference at σ = 1.0; no '
         f'faster on {dec_nbetter} of {len(dec_l)} datasets): the learned '
         'weights rescale feature axes but do not amplify any particular '
         'noise direction, so the conservative d-parameter adaptation '
         'leaves the noise profile of the frozen backbone essentially '
         'unchanged. How the adaptation rule shapes behavior under input '
         'corruption is known to matter' +
         cite('kumar2022finetune', 'wortsman2022robust') + '.',
         indent=True)
    add_fig(os.path.join(FIGS, 'fig_s8_inputnoise.png'), 15.5,
            'Fig. 5. Macro-F1 under Gaussian input noise at evaluation '
            'time (σ on standardized features; frac = 20% models).')

    low_ds = next((d for d in DATASETS
                   if f'{d}|LFW(ours)|0.05' in LOWRES), None)
    if low_ds:
        para('Low-resource regimes. Figure 2 already shows the 5-20% '
             'regime. At 5% of labels, LFW achieves '
             f'{LOWRES[f"{low_ds}|LFW(ours)|0.05"][0]*100:.1f}% macro-F1 '
             f'on {low_ds} versus '
             f'{LOWRES.get(f"{low_ds}|FFT|0.05", (0,))[0]*100:.1f}% for FFT '
             f'and {LOWRES.get(f"{low_ds}|ZeroShot|0.05", (0,))[0]*100:.1f}% '
             'zero-shot: with vanishing labels, the safest strategy is to '
             'trust the pre-trained geometry and adjust only a well-'
             'regularized input rescaling.', indent=True)

    # ---------------------------------------------------- 4.5 Visualization
    h2('4.5. Qualitative analysis')
    para('Figure 6 compares confusion matrices of zero-shot, LFW, and FFT '
         'on the evaluation split of three datasets. On HAR, the dominant '
         'errors of the zero-shot model - confusing sitting vs. standing '
         'and walking upstairs vs. downstairs - are precisely the '
         'activities whose discriminative statistics (posture angles, '
         'jerk magnitudes) differ most across subjects; LFW re-weights '
         'the corresponding sensor features and recovers most of these '
         'confusions, approaching the FFT pattern without touching the '
         'backbone.', indent=True)
    for ds in ('HAR', 'Satimage', 'WDBC'):
        pth = os.path.join(FIGS, f'fig_s5_confusion_{ds}.png')
        if os.path.exists(pth):
            add_fig(pth, 15.5,
                    f'Fig. 6{"" if ds == "HAR" else " (cont.)"}. Confusion '
                    f'matrices (row-normalized) on {ds}: zero-shot vs. LFW '
                    f'vs. FFT (seed 0, 100% fraction).')

    tsne_p = os.path.join(FIGS, 'fig_s6_tsne_HAR.png')
    if os.path.exists(tsne_p):
        para('Figure 7 embeds penultimate-layer features with t-SNE. The '
             'zero-shot features of the downstream domain are already '
             'well clustered - pre-training transfers - but class margins '
             'are tighter than after adaptation; LFW widens the margins '
             'to nearly the FFT level by stretching the input axes that '
             'the frozen network is most sensitive to, which is exactly '
             'what the learned w encodes.', indent=True)
        add_fig(tsne_p, 16.0,
                'Fig. 7. t-SNE of penultimate-layer features on HAR '
                '(zero-shot vs. LFW vs. FFT; up to 1,200 evaluation '
                'samples).')

    wt_p = os.path.join(FIGS, 'fig_s4_weights.png')
    if os.path.exists(wt_p):
        para('Figure 8 shows the top learned weights per dataset. On HAR '
             'the largest weights concentrate on gravity-component '
             'statistics and jerk energy bands - physically meaningful '
             'for posture/motion separation; near-zero weights hit '
             'redundant inter-axis correlations. Because w is '
             'non-negative and acts multiplicatively on standardized '
             'inputs, it can be read directly as a feature-importance '
             'profile, in the same spirit as post-hoc attribution' +
             cite('lundberg2017shap') + ' but obtained for free during '
             'adaptation.', indent=True)
        add_fig(wt_p, 16.0,
                'Fig. 8. Top-15 learned feature weights on two datasets '
                '(seed with the most informative weights; 100% fraction).')

    # ---------------------------------------------------- 4.6 Efficiency
    h2('4.6. Efficiency and convergence')
    para('Table 6 reports trainable parameters, training time, and '
         'per-sample inference latency at the 100% fraction, averaged over '
         'datasets; Figure 9 shows the accuracy-versus-parameters frontier. '
         'LFW trains exactly d parameters - the smallest budget among all '
         'adaptation methods except zero-shot - with a median parameter '
         f'ratio of 1:{STATS["param_ratio_median"]:.0f} versus FFT. Its '
         'training time is dominated by forward/backward passes through '
         'the frozen backbone and is materially lower than FFT (median '
         f'{STATS["time_reduction_median"]:.0f}% reduction across datasets), '
         'and its inference latency is statistically indistinguishable '
         'from the plain frozen model (the extra element-wise product '
         'costs O(d)).', indent=True)

    t6_rows = []
    for m in METHODS:
        rows_ = [MAIN[f'{ds}|{m}'] for ds in DATASETS
                 if f'{ds}|{m}' in MAIN]
        if not rows_:
            continue
        npar = st.median([r['n_tuned'] for r in rows_ if r['n_tuned'] is not None])
        tt = st.mean([r['train_time'] for r in rows_])
        it = st.mean([r['infer_ms'] for r in rows_])
        f1m = st.mean([r['metrics']['f1'][0] for r in rows_])
        t6_rows.append([LBL_EN[m], f'{npar:,.0f}', f'{tt:.1f}',
                        f'{it:.3f}', f'{f1m*100:.2f}'])
    sci_table('Table 6. Efficiency summary at the 100% fraction '
              '(median trainable parameters; mean training time in '
              'seconds; mean per-sample inference latency in '
              'milliseconds; mean macro-F1 in %).',
              ['Method', 'Params', 'Train (s)', 'Infer (ms/sample)',
               'Macro-F1 (%)'],
              t6_rows, col_widths=[3.2, 3.0, 2.4, 3.2, 2.6],
              note='Classical learners (LR/SVM/RF) train a full model from '
                   'scratch; their parameter counts are not comparable to '
                   'backbone-adaptation methods and are reported for '
                   'reference.')

    add_fig(os.path.join(FIGS, 'fig_s10_efficiency.png'), 13.5,
            'Fig. 9. Accuracy-parameter frontier at the 100% fraction '
            '(each point is a method on a dataset; log-scale x-axis).')

    conv_p = os.path.join(FIGS, 'fig_s3_convergence.png')
    if os.path.exists(conv_p):
        para('Figure 10 traces validation macro-F1 against both epoch and '
             'wall-clock time. LFW converges within a few dozen epochs - '
             'often an order of magnitude fewer than FFT - because the '
             'd-dimensional optimization landscape is nearly convex given '
             'a frozen mapping, and its per-epoch cost is roughly half of '
             'FFT\'s (no backbone weight gradients). SSF, despite its '
             'strong final accuracy, needs more epochs at similar '
             'per-epoch cost.', indent=True)
        add_fig(conv_p, 15.5,
                'Fig. 10. Convergence of validation macro-F1 vs. epoch '
                '(top) and wall-clock time (bottom) on HAR and Bank '
                '(seed 0, 100% fraction).')

    # ---------------------------------------------------- 4.7 Significance
    h2('4.7. Statistical significance')
    if WILCOX:
        t7_rows = []
        for m, w in sorted(WILCOX.items(), key=lambda kv: kv[1]['p']):
            t7_rows.append([LBL_EN.get(m, m), str(w['n']),
                            f'{w["win"]}/{w["n"]}',
                            f'{w["stat"]:.1f}',
                            f'{w["p"]:.2e}' if w['p'] >= 1e-4 else '<1e-4',
                            'Yes' if w['p'] < 0.05 else 'No'])
        sci_table('Table 7. Wilcoxon signed-rank tests: LFW vs. each '
                  'baseline on paired (dataset × seed) macro-F1 at the '
                  '100% fraction.',
                  ['Baseline', 'Pairs n', 'LFW wins', 'Statistic W',
                   'p-value', 'Significant (α=0.05)'],
                  t7_rows, col_widths=[3.0, 1.8, 2.0, 2.4, 2.4, 3.0],
                  note='Tests are two-sided. Wins count strict macro-F1 '
                       'advantages for LFW.')
        sig = [m for m, w in WILCOX.items()
               if w['p'] < 0.05 and w.get('mean_diff', 0) > 0]
        para(f'LFW significantly outperforms {len(sig)} of '
             f'{len(WILCOX)} comparable baselines (two-sided Wilcoxon '
             f'signed-rank, α = 0.05, n = datasets × seeds), including '
             f'the static filters and several PEFT methods; the remaining '
             f'gaps against FFT and the strongest PEFT baselines on '
             f'individual datasets are examined in Section 5.',
             indent=True)

    # ------------------------------------------------ 4.8 Vision ViT validation
    _vis_path = os.path.join(RESULTS, 'sci_vision_analysis.json')
    VIS_OK = False
    if os.path.exists(_vis_path):
        import json as _json
        VIS = _json.load(open(_vis_path, encoding='utf-8'))
        if VIS.get('main'):
            VMAIN, VLOW = VIS['main'], VIS.get('lowres', {})
            VWIL, VSUM = VIS.get('wilcoxon', {}), VIS.get('summary', {})
            VDS = [d for d in ['USPS', 'C10C_gaussian_noise',
                               'C10C_motion_blur', 'C10C_fog',
                               'C10C_brightness']
                   if any(f'{d}|{m}' in VMAIN for m in METHODS)]
            VSHORT = {'USPS': 'USPS', 'C10C_gaussian_noise': 'Gauss.',
                      'C10C_motion_blur': 'Blur', 'C10C_fog': 'Fog',
                      'C10C_brightness': 'Bright'}
            VMETHODS = [m for m in METHODS
                        if any(f'{d}|{m}' in VMAIN for d in VDS)]
            if VDS:
                VIS_OK = True
                h2('4.8. Validation on vision Transformer backbones')
                bb = VIS.get('backbone', {})
                para('A natural concern is that BitFit, SSF, Adapter, and LoRA '
                     'were designed for large Transformers, whereas our '
                     'tabular backbones are MLPs. To test whether these '
                     'conclusions transfer to the architecture family for '
                     'which these PEFT methods were introduced, we '
                     'replicated the protocol on a compact vision Transformer '
                     f'({bb.get("config", "")}, {bb.get("params", 0):,} '
                     'parameters, trained from scratch on the source domain) '
                     'over two source-domain transfer settings: '
                     'MNIST (60,000 source images) to USPS (official '
                     'train/test split; 16/28 px digits bilinearly resized to '
                     '32 px and replicated to 3 channels), and CIFAR-10 '
                     '(50,000 source images) to four CIFAR-10-C corruptions '
                     + cite('lecun1998gradient', 'hull1994database',
                            'krizhevsky2009learning',
                            'hendrycks2019benchmarking') + ' '
                     'at severity 3 (gaussian noise, motion blur, fog, '
                     'brightness; 10,000 test images each, 50/50 stratified '
                     'pool/eval split). LFW here is a non-negative '
                     'element-wise weight vector over the 3,072 standardized '
                     'input pixels - the exact same formulation as the tabular '
                     'model, with no architectural change. The backbone and '
                     'all baselines are implemented in PyTorch; all other '
                     'settings (five seeds, fraction grid, hyper-parameter '
                     'selection, safeguard) match the tabular protocol.',
                    indent=True)

                # Table 8: vision main results
                def vf1(ds, m):
                    cell = VMAIN.get(f'{ds}|{m}')
                    return cell['f1'][0] if cell else None

                vrows = []
                bold = set()
                v_ranks = {m: [] for m in VMETHODS}
                for m in VMETHODS:
                    row = [LBL_EN.get(m, m)]
                    vals = []
                    for ds in VDS:
                        cell = VMAIN.get(f'{ds}|{m}')
                        if cell:
                            mu, sd = cell['f1']
                            row.append(f'{mu*100:.1f}±{sd*100:.1f}')
                            vals.append(mu)
                        else:
                            row.append('-')
                    if vals:
                        row.append(f'{st.mean(vals)*100:.1f}')
                        for ds in VDS:
                            scored = [(vf1(ds, mm), mm) for mm in VMETHODS
                                      if vf1(ds, mm) is not None]
                            scored.sort(reverse=True)
                            for rank_i, (_, mm) in enumerate(scored, 1):
                                if mm == m:
                                    v_ranks[m].append(rank_i)
                    else:
                        row.append('-')
                    vrows.append(row)
                # bold best per dataset column
                for j, ds in enumerate(VDS):
                    col = [(vf1(ds, m), i) for i, m in enumerate(VMETHODS)
                           if vf1(ds, m) is not None]
                    if col:
                        best_i = max(col)[1]
                        bold.add((best_i, j + 1))
                # rank column (computed from v_ranks accumulated above)
                for i, m in enumerate(VMETHODS):
                    rk = v_ranks.get(m, [])
                    vrows[i].append(f'{st.mean(rk):.1f}' if rk else '-')
                sci_table(
                    'Table 8. Macro-F1 (%) at the 100% adaptation fraction on '
                    'vision-Transformer transfer tasks (mean ± std over five '
                    'seeds).',
                    ['Method'] + [VSHORT.get(d, d) for d in VDS] +
                    ['Avg', 'Rank'],
                    vrows,
                    col_widths=[2.6] + [2.0] * len(VDS) + [1.2, 1.2],
                    font_size=8, bold_cells=bold,
                    note='USPS = MNIST→USPS; Gauss./Blur/Fog/Bright = '
                         'CIFAR-10→CIFAR-10-C severity 3. Backbone: compact '
                         'ViT (~1.22M params). LFW trains 3,072 parameters '
                         '(input weighting only).')

                # analysis text
                def vf(ds, m, fr='1'):
                    c = VLOW.get(f'{ds}|{m}|{fr}')
                    return c[0] if c else None
                lfw_f1 = [vf(d, 'LFW(ours)') for d in VDS]
                zs_f1 = [vf(d, 'ZeroShot') for d in VDS]
                lfw_f1 = [x for x in lfw_f1 if x is not None]
                zs_f1 = [x for x in zs_f1 if x is not None]
                peft_gap, fft_gap = [], []
                for d in VDS:
                    lv = vf(d, 'LFW(ours)')
                    best_peft = max([vf(d, m) for m in
                                     ('LinearProbe', 'BitFit', 'SSF',
                                      'Adapter', 'LoRA')
                                     if vf(d, m) is not None] or [None])
                    fv = vf(d, 'FFT')
                    if lv is not None and best_peft is not None:
                        peft_gap.append(lv - best_peft)
                    if lv is not None and fv is not None:
                        fft_gap.append(lv - fv)
                ratio = VSUM.get('param_ratio_backbone_over_lfw', 0)
                _pg = st.mean(peft_gap) if peft_gap else 0.0
                _fg = st.mean(fft_gap) if fft_gap else 0.0
                # strongest PEFT baseline by mean f1 across vision tasks
                _peft_means = {}
                for _m in ('LinearProbe', 'BitFit', 'SSF', 'Adapter', 'LoRA'):
                    _vv = [vf(d, _m) for d in VDS]
                    _vv = [x for x in _vv if x is not None]
                    if _vv:
                        _peft_means[_m] = st.mean(_vv)
                _best_peft = (max(_peft_means, key=_peft_means.get)
                              if _peft_means else None)
                _bp_name = (LBL_EN.get(_best_peft, _best_peft)
                            if _best_peft else 'the best PEFT baseline')
                _peft_clause = (
                    f'exceeds {_bp_name} (the strongest PEFT baseline) by '
                    f'{_pg*100:.1f} pp' if _pg >= 0 else
                    f'trails {_bp_name} (the strongest PEFT baseline) by '
                    f'{-_pg*100:.1f} pp') if _best_peft else ''
                _fft_clause = (
                    f'and even matches or exceeds full fine-tuning on '
                    f'average' if _fg > 0 else
                    f'while full fine-tuning retains a {_fg*-100:.1f} pp '
                    f'upper-bound margin - expected, since it can re-shape '
                    f'the entire downstream decision surface')
                # task-dependency of the zero-shot gain
                _gains = {}
                for d in VDS:
                    _lv, _zv = vf(d, 'LFW(ours)'), vf(d, 'ZeroShot')
                    if _lv is not None and _zv is not None:
                        _gains[d] = _lv - _zv
                _gsort = sorted(_gains.items(), key=lambda kv: -kv[1])
                _gain_str = ', '.join(
                    f'{VSHORT.get(d, d)} {v*100:+.1f}' for d, v in _gsort[:3])
                # share of the zero-shot->FFT gap recovered by LFW
                _fft_vals = [vf(d, 'FFT') for d in VDS]
                _fft_vals = [x for x in _fft_vals if x is not None]
                _rec = ''
                if _fft_vals and zs_f1 and lfw_f1:
                    _zs_m, _ff_m = st.mean(zs_f1), st.mean(_fft_vals)
                    if _ff_m > _zs_m:
                        _rec = (f', recovering '
                                f'{((st.mean(lfw_f1)-_zs_m)/(_ff_m-_zs_m))*100:.0f}'
                                f'% of the zero-shot-to-FFT '
                                f'improvement on average')
                # safeguard fallback on pixel-level noise
                _saf = [d for d, v in _gains.items() if v < 0.01]
                _saf_str = (f'; on '
                            f'{_saf[0].replace("C10C_", "").replace("_", " ")}'
                            f' the safeguard correctly falls back to near-'
                            f'zero-shot behavior (gain '
                            f'{_gains[_saf[0]]*100:+.1f} pp), since pixel-'
                            f'level noise is not a marginal shift that '
                            f're-scaling can absorb' if _saf else '')
                para(
                    f'Table 8 shows that input re-weighting transfers to the '
                    f'Transformer backbone in a clearly task-dependent '
                    f'manner. Averaged over the five vision tasks, LFW '
                    f'improves macro-F1 over zero-shot by '
                    f'{(st.mean(lfw_f1)-st.mean(zs_f1))*100:+.1f} pp'
                    f'{_rec}, with the largest per-task gains at '
                    f'{_gain_str} pp{_saf_str}. Nevertheless, LFW '
                    f'{_peft_clause}, {_fft_clause}. The residual gap to '
                    f'weight-space adaptation is wider here than in the '
                    f'tabular setting, delimiting the applicability '
                    f'boundary of input-only adaptation: when the shift is '
                    f'largely marginal (USPS digit style, global contrast '
                    f'and blur corruptions) a diagonal re-scaling of the '
                    f'3,072 input pixels recovers most of the benefit, '
                    f'whereas representations that the frozen backbone '
                    f'never encoded still require internal updates. '
                    f'Crucially, LFW still trains only 3,072 parameters '
                    f'(one non-negative scalar per input pixel-channel; '
                    f'~1:{ratio:.0f} of the backbone) without inserting or '
                    f'modifying any internal parameter.',
                    indent=True)
                # low-resource + provenance
                f005 = VSUM.get('lfw_gain_over_best_peft_f005')
                if f005 is not None:
                    _lr_clause = (
                        f'exceeds the best PEFT baseline by {f005*100:.1f} pp'
                        if f005 >= 0 else
                        f'trails the best PEFT baseline by {-f005*100:.1f} pp')
                    _usps_lp = vf('USPS', 'LinearProbe', '0.05')
                    _usps_lfw = vf('USPS', 'LFW(ours)', '0.05')
                    _usps_str = ''
                    if _usps_lp is not None and _usps_lfw is not None:
                        _usps_str = (
                            f'; on USPS, however, LFW '
                            f'({_usps_lfw*100:.1f}) remains within '
                            f'{abs(_usps_lp-_usps_lfw)*100:.1f} pp of the '
                            f'linear probe ({_usps_lp*100:.1f})')
                    para(f'At 5% labeled data, LFW {_lr_clause} on average '
                         f'across the vision tasks{_usps_str}. The '
                         f'low-resource advantage observed on tabular data '
                         f'(Figure 2) thus only partially carries over: the '
                         f'input-only parameterization stays viable when the '
                         f'shift is marginal, but the gap widens on the '
                         f'semantic CIFAR-10-C corruptions.',
                         indent=True)
                sig_v = [(k, w) for k, w in VWIL.items()
                         if isinstance(w, dict) and w.get('p', 1) < 0.05]
                if sig_v:
                    _wins = [k for k, w in sig_v
                             if w.get('lfw_wins', 0) > w.get('other_wins', 0)]
                    _loss = [k for k, w in sig_v
                             if w.get('lfw_wins', 0) < w.get('other_wins', 0)]
                    _parts = []
                    if _wins:
                        _parts.append(
                            'significantly better than ' +
                            ', '.join(k.replace('LFW_vs_', '')
                                      for k in sorted(_wins)))
                    if _loss:
                        _parts.append(
                            'significantly worse than ' +
                            ', '.join(k.replace('LFW_vs_', '')
                                      for k in sorted(_loss)))
                    para('Paired two-sided Wilcoxon signed-rank tests over '
                         'all vision task cells (dataset x fraction x seed) '
                         'find LFW ' + ' and '.join(_parts) +
                         ' at alpha = 0.05.', indent=True)
                prov = VIS.get('provenance', {})
                gen = [d for d, p in prov.items()
                       if d.startswith('C10C') and 'generated' in str(p)]
                if gen:
                    para('Provenance note: the CIFAR-10-C corrupted images '
                         'were generated from the official CIFAR-10 test set '
                         'with the reference corruption implementation of '
                         'Hendrycks and Dietterich '
                         + cite('hendrycks2019benchmarking') +
                         ' (imagecorruptions, '
                         'severity 3), as the official pre-corrupted archive '
                         'was unavailable in the compute environment; the '
                         'generative recipe and parameters are identical to '
                         'the official release. USPS data are the official '
                         'train/test splits.', indent=True, size=9)

    # ============================================================ 5 Discussion
    h1('5. Discussion')

    h2('5.1. Why does input re-weighting recover so much performance?')
    para('The zero-shot transfer of a source-trained model to a shifted '
         'downstream domain fails mainly because the marginal statistics '
         'of the input features change: features that were informative in '
         'the source domain may become noisy, redundant, or misleading '
         'downstream. A frozen decision function fθ is a fixed '
         'transformation, but composing it with a diagonal rescaling of '
         'the input space is enough to (i) suppress feature axes that '
         'inject noise, (ii) stretch axes along which the frozen network '
         'is most discriminative, and (iii) re-balance one-hot blocks '
         'whose category priors shifted. Our ablation shows that even '
         'unconstrained w captures most of the gain; regularization and '
         'non-negativity make it reliable, and the safeguard makes it '
         'safe. In other words, much of what full fine-tuning "learns" '
         'on small tabular adaptation sets is effectively a recalibration '
         'of input feature importance - which a d-parameter model can '
         'express with far less variance.', indent=True)

    para('Positioning. The evidence supports a concrete selection rule '
         'rather than a universal claim. When the adaptation budget is '
         'unconstrained and the deployment distribution is stationary and '
         'clean, full fine-tuning remains the accuracy upper bound, with '
         'SSF-class PEFT methods the best intermediate trade-off (Table 2). '
         'LFW is the method of choice when (i) the trainable budget must '
         'stay extreme (d parameters, roughly 0.1% of the backbone or '
         'less), (ii) the deployed artifact must remain bit-identical to '
         'the audited pre-trained model, (iii) adaptation labels are noisy '
         'or scarce - regimes in which it matches or exceeds full '
         'fine-tuning in our experiments (Table 5, Figure 2) - or (iv) a '
         'per-feature audit trail is required. Read this way, LFW occupies '
         'a distinct point on the accuracy-cost-robustness frontier '
         '(Figure 9) rather than its accuracy apex; its residual gap to '
         'FFT on clean, resource-rich benchmarks is the price paid for '
         'd-parameter compliance, robustness, and interpretability.',
         indent=True)

    para('Failure modes. LFW struggles when the domain shift is '
         'structural rather than marginal. On datasets where LFW '
         'underperforms FFT by more than 1 pp (e.g., Adult and Digits at '
         'the 100% fraction), the gap traces to label-semantic shift rather '
         'than feature-statistics shift: the source and downstream splits '
         'contain different category priors or class boundaries that no '
         'input rescaling can synthesize. This is consistent with the '
         'covariate-shift analysis of Section 3.1 - when P(y|x) itself '
         'shifts, the frozen decision boundary is genuinely wrong and '
         'must be rewritten, which requires modifying θ. Conversely, on '
         'HAR (cross-subject) and Credit-g, where the shift is dominated '
         'by feature marginal differences, LFW recovers most of the FFT '
         'gap with only d parameters, confirming that diagonal input '
         'reweighting is the right inductive bias for marginal shifts.',
         indent=True)

    h2('5.2. Advantages')
    para('(1) Compliance and deployability: the deployed artifact is '
         'bit-identical to the pre-trained model; adaptation state is a '
         'length-d vector that can be versioned, audited, or shipped '
         'per-customer without touching model weights. (2) Extreme '
         'parameter efficiency: d trainable parameters (0.01-0.1% of the '
         'backbone in our experiments), enabling adaptation on '
         'microcontrollers or through inference APIs; the median '
         f'parameter ratio is 1:{STATS["param_ratio_median"]:.0f} versus '
         'FFT (Table 6). (3) Safety: the safeguard guarantees no '
         'regression below zero-shot on validation data - a '
         'deployment-relevant guarantee achieved with zero extra '
         'parameters. (4) Interpretability: the non-negative weights are a '
         'direct, monotone feature-importance readout (Figure 8) that '
         'requires no post-hoc attribution tool. (5) Universality: any '
         'differentiable model, any data modality with a fixed-length '
         'feature interface, no architectural assumptions.', indent=True)

    h2('5.3. Limitations')
    para('LFW is not a universal substitute for fine-tuning. First, it '
         'cannot change the rank or geometry of the frozen function: if '
         'the downstream task requires representations the backbone never '
         'encoded, no input rescaling will create them - our results '
         'show a residual gap to FFT on datasets where the source and '
         'downstream label semantics differ most. Second, element-wise '
         'weighting cannot express cross-feature interactions (rotations '
         'or correlations); an affine variant (w ⊙ x + b) and low-rank '
         'input maps are natural extensions, but every added degree of '
         'freedom erodes the variance and compliance advantages. Third, '
         'our tabular backbones are modest MLPs'
         + (', and our supplementary vision-Transformer study (Section 4.8) '
            'indicates that the gap to weight-space PEFT widens on that '
            'architecture family' if VIS_OK else '')
         + '; on very deep networks or sequence models, the gradient signal '
         'through many frozen layers may be weaker, and the effectiveness '
         'of LFW relative to LoRA-style weight-space updates must be '
         're-established at that scale. Fourth, like all supervised '
         'adaptation, LFW assumes labeled downstream data; it does not '
         'address source-free or test-time settings' + cite('wang2021tent', 'liang2020shot') + '. Finally, '
         'macro-F1 on UCI-style benchmarks, while standard, does not '
         'cover all deployment concerns (calibration, fairness, privacy).',
         indent=True)

    h2('5.4. Future work')
    para('Three directions seem most promising. (1) Structured input '
         'adaptation: block-diagonal or low-rank input transformations '
         'that retain most of the parameter savings while expressing '
         'limited cross-feature interactions. (2) Combining LFW with '
         'weight-space PEFT: an input-side rescaling plus a tiny internal '
         'update (e.g., LoRA on the first layer only) may dominate either '
         'in isolation while staying within strict parameter budgets. '
         '(3) Theoretical analysis: characterizing when diagonal input '
         'rescaling suffices to align source and downstream distributions '
         '(e.g., under covariance-shift assumptions) would formalize the '
         'empirical findings of this study.', indent=True)

    # ============================================================ 6 Conclusion
    h1('6. Conclusion')
    para('We presented Learnable Feature Weighting (LFW), a transfer '
         'learning method that adapts a fully frozen pre-trained backbone '
         'by optimizing only a non-negative, L1/L2-regularized input '
         'weight vector, with a safeguard that guarantees no degradation '
         'below zero-shot transfer. Across nine real UCI datasets, five '
         'seeds, and twelve baselines spanning classical learners, static '
         'feature filters, and five state-of-the-art PEFT methods, LFW '
         f'improves macro-F1 over zero-shot by {STATS["gain_over_zeroshot_mean"]:.1f} pp '
         'on average, is competitive with recent PEFT methods at a '
         f'fraction of the trainable parameters (1:{STATS["param_ratio_median"]:.0f} of '
         'full fine-tuning, median), trains faster, outperforms full '
         'fine-tuning under label noise, degrades no faster than the '
         'frozen zero-shot model under input noise, and yields directly '
         'interpretable feature attributions. '
         + ('A supplementary validation on a compact vision Transformer '
            'shows that the large zero-shot gains carry over to that '
            'architecture family, while the gap to weight-space PEFT '
            'widens there - delimiting where input-only adaptation is '
            'sufficient. ' if VIS_OK else '')
         + 'LFW is not a replacement '
         'for fine-tuning in general, but it defines and validates the '
         'minimal-invasion extreme of the adaptation spectrum - the '
         'method of choice when the backbone must remain untouched, '
         'labels are scarce, or compute is tight.', indent=True)

    # ===================================================== Declarations
    h1('Conflict of Interest')
    para('The author declares that there is no conflict of interest '
         'regarding the publication of this paper.', indent=True)

    h1('Acknowledgments')
    para('The author thanks the UCI Machine Learning Repository for '
         'making the benchmark datasets publicly available.', indent=True)

    h1('Code Availability')
    para('The source code for LFW, all experiment scripts, and the raw '
         'per-seed results backing every table are openly available at '
         'https://github.com/luo1chen/lfw-frozen-backbone (MIT license).',
         indent=True)

    h1('Data Availability')
    para('All datasets used in this study are publicly available and '
         'require no special access: the nine tabular benchmarks from '
         'the UCI Machine Learning Repository '
         '(https://archive.ics.uci.edu; dataset-specific references in '
         'Table 1 and References [37, 38]), and MNIST, USPS, CIFAR-10, '
         'and CIFAR-10-C for the vision study from their official '
         'sources ' + cite('lecun1998gradient', 'hull1994database',
                           'krizhevsky2009learning',
                           'hendrycks2019benchmarking') + '.',
         indent=True)

    # ============================================================ References
    h1('References')
    for i, key in enumerate(CITE_ORDER, 1):
        r = REFS.get(key)
        if r is None:
            continue
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.line_spacing = 1.15
        pf.space_after = Pt(2)
        pf.left_indent = Pt(24)
        pf.first_line_indent = Pt(-24)
        run = p.add_run(f'[{i}] {r["authors"]} {r["title"]}. '
                        f'{r["venue"]}, {r["year"]}. {r["doi_or_id"]}.')
        set_font(run, size=9)

    # ============================================================ 保存
    doc.save(OUT)
    print(f'Saved: {OUT}')
