# -*- coding: utf-8 -*-
"""SCI 版全部实验（真实数据、真实运行、可复现）。

数据集（9，全部 UCI 官方真实数据）：
  大规模：Adult(32.5K) / Bank(22.6K)；跨被试域偏移：HAR(7.4K,6类)；
  中小规模：Spambase(2.3K) / Satimage(2.2K,6类) / Digits(0.9K,10类) /
            Credit-g(0.5K) / WDBC(0.28K) / Heart(0.15K)。

方法（13）：
  传统(4)：VarFilter / MIFilter（过滤式特征筛选）；LR / SVM / RF（下游从头训练）
  PEFT SOTA(5)：LinearProbe / BitFit / SSF / Adapter / LoRA
  本文(1)：LFW（全冻结 + 可学习输入特征加权）
  参照(3)：ZeroShot（冻结直接推理）/ FFT（全参数微调上界）

协议（可复现）：
  - 5 个随机种子(0-4)；源域按 90/10 训练/验证预训练（分层，按种子）；
  - 下游池 5%-100% 分层采样（Heart/Digits 等小池数据集从 10% 起，
    sample_fraction 已修正为真实小样本），80/20 训练/验证（分层，
    seed+7）；评估集固定为下游域 50%；
  - 全部超参在验证集网格搜索确定，评估集每配置仅评估一次；
  - 指标：Acc / Precision(macro) / Recall(macro) / Macro-F1 /
          可训练参数量 / 训练耗时 / 推理耗时(ms/样本，中位数×5)。

附加：递进式消融(none→w→+nonneg→+reg→+safeguard→full→affine)、
      标签噪声(10/20/30%)、输入高斯噪声(σ=0.1/0.5/1.0)、
      λ 敏感性(5×5 网格)、收敛曲线、t-SNE/混淆矩阵数据导出。
用法：
  python run_sci.py --quick          # 冒烟（2 小数据集 1 种子，全流程）
  python run_sci.py                  # 完整（默认 9 数据集 5 种子）
  python run_sci.py --resume         # 跳过已完成行，断点续跑
  python run_sci.py --datasets HAR Bank --seeds 0 1
"""
import argparse
import csv
import json
import os
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.svm import SVC

from mlp import MLP, evaluate, finetune, infer_time, pretrain, train_weights
from peft import PEFTModel, evaluate_peft, infer_time_peft, train_peft
from data_utils import get_dataset, sample_fraction, stratified_split
from data_sci import get_sci_dataset

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
PRE = os.path.join(RESULTS, 'sci_pretrained')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PRE, exist_ok=True)

# 数据集顺序：核心多类与大规模优先（viz 用 HAR/Satimage，大样本 Bank/Adult）
ALL_DS = ['HAR', 'Bank', 'Adult', 'Satimage', 'Digits', 'Spambase',
          'Credit-g', 'WDBC', 'Heart']
OLD_DS = ['Adult', 'Heart', 'HAR']
FRACTIONS = {ds: ([0.1, 0.2, 0.5, 1.0] if ds in ('Heart', 'Digits')
                  else [0.05, 0.1, 0.2, 0.5, 1.0]) for ds in ALL_DS}
# 说明：Digits 池仅 450 样本，frac=0.05 时 10 类×2 样本/类，80/20 分层
# 划分后验证集为空，无法早停；故与 Heart 一致从 10% 起步。
SEEDS = [0, 1, 2, 3, 4]

PEFT_KIND = {'LinearProbe': 'probe', 'BitFit': 'bitfit', 'SSF': 'ssf',
             'Adapter': 'adapter', 'LoRA': 'lora'}
LR_GRID = (1e-3, 3e-3, 1e-2)
LFW_EPOCHS, LFW_PAT = 300, 20      # 本文方法/FFT 预算（与原协议一致）
PEFT_EPOCHS, PEFT_PAT = 100, 10    # PEFT 基线预算
BATCH = 256
VIZ_DS = ('HAR', 'Satimage', 'WDBC')   # 混淆矩阵/t-SNE 导出数据集
CONV_DS = ('HAR', 'Bank')              # 收敛曲线记录数据集


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def get_D(name, seed):
    return get_dataset(name, seed) if name in OLD_DS else get_sci_dataset(name, seed)


# ---------------------------------------------------------------- 预训练缓存

def get_pretrained(name, seed):
    """预训练主干（磁盘缓存，支持断点续跑）。"""
    path = os.path.join(PRE, f'{name}_s{seed}.npz')
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        model = MLP(z['sizes'].tolist(), seed=seed)
        model.W = [w for w in z['W']]
        model.b = [b for b in z['b']]
        return model, z['meta'].item()
    D = get_D(name, seed)
    i_tr, i_va = stratified_split(D['ys'], 0.9, seed)
    model = MLP(D['sizes'], seed=seed)
    t0 = time.perf_counter()
    val_f1, ep = pretrain(model, D['Xs'][i_tr], D['ys'][i_tr],
                          D['Xs'][i_va], D['ys'][i_va],
                          epochs=200, batch=BATCH, lr=1e-3, patience=15, seed=seed)
    meta = dict(dataset=name, seed=seed, val_f1=float(val_f1), stop_epoch=int(ep),
                n_params=int(model.n_params()), pretrain_time=time.perf_counter() - t0)
    np.savez(path, sizes=np.array(D['sizes']), W=np.array(model.W, dtype=object),
             b=np.array(model.b, dtype=object), meta=np.array(meta))
    log(f'预训练 {name} s{seed}: valF1={val_f1:.4f} ep={ep} '
        f'params={model.n_params()} {meta["pretrain_time"]:.0f}s')
    return model, meta


# ---------------------------------------------------------------- 超参选择

def select_hyperparams(name, D, model):
    """每数据集一次：frac=0.5 池数据 80/20 训练/验证，网格搜索全部超参。"""
    idx = sample_fraction(D['Xpool'], D['ypool'], 0.5, seed=0)
    Xa, ya = D['Xpool'][idx], D['ypool'][idx]
    i_tr, i_va = stratified_split(ya, 0.8, seed=0)
    Xtr, ytr, Xva, yva = Xa[i_tr], ya[i_tr], Xa[i_va], ya[i_va]
    h = {}
    # 静态筛选比例 k
    var = Xtr.var(axis=0)
    mi = mutual_info_classif(Xtr, ytr, random_state=0)
    for tag, score in (('k_var', var), ('k_mi', mi)):
        best = (-1, 0.5)
        for k in (0.25, 0.5, 0.75):
            kk = max(1, int(round(Xtr.shape[1] * k)))
            m = np.zeros(Xtr.shape[1]); m[np.argsort(-score)[:kk]] = 1.0
            v = evaluate(model, Xva, yva, m)['f1_macro']
            if v > best[0]:
                best = (v, k)
        h[tag] = best[1]
    # 本文方法 l1/l2
    best = (-1, (0.0, 0.0))
    for l1 in (1e-4, 1e-3, 1e-2):
        for l2 in (0.0, 1e-5, 1e-4, 1e-3):
            _, v, _, _, _ = train_weights(model, Xtr, ytr, Xva, yva, l1=l1, l2=l2,
                                          epochs=LFW_EPOCHS, batch=BATCH, lr=0.05,
                                          patience=LFW_PAT, seed=0)
            if v > best[0]:
                best = (v, (l1, l2))
    h['l1'], h['l2'] = best[1]
    # PEFT 学习率
    for meth, kind in PEFT_KIND.items():
        best = (-1, LR_GRID[0])
        for lr in LR_GRID:
            pm = PEFTModel(model, kind)
            v, _, _, _ = train_peft(pm, Xtr, ytr, Xva, yva, lr=lr,
                                    epochs=PEFT_EPOCHS, batch=BATCH,
                                    patience=PEFT_PAT, seed=0)
            if v > best[0]:
                best = (v, lr)
        h[f'lr_{meth}'] = best[1]
    # FFT 学习率
    best = (-1, 1e-3)
    for lr in (1e-3, 1e-4):
        _, v, _, _, _ = finetune(model, Xtr, ytr, Xva, yva, epochs=LFW_EPOCHS,
                                 batch=BATCH, lr=lr, patience=LFW_PAT, seed=0)
        if v > best[0]:
            best = (v, lr)
    h['lr_FFT'] = best[1]
    # 传统 ML 基线超参
    best = (-1, 1.0)
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
        v = f1_score(yva, clf.predict(Xva), average='macro')
        if v > best[0]:
            best = (v, C)
    h['C_LR'] = best[1]
    best = (-1, 1.0)
    for C in (0.1, 1.0, 10.0):
        clf = SVC(C=C, gamma='scale').fit(Xtr, ytr)
        v = f1_score(yva, clf.predict(Xva), average='macro')
        if v > best[0]:
            best = (v, C)
    h['C_SVM'] = best[1]
    best = (-1, 1)
    for leaf in (1, 4):
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=leaf,
                                     n_jobs=-1, random_state=0).fit(Xtr, ytr)
        v = f1_score(yva, clf.predict(Xva), average='macro')
        if v > best[0]:
            best = (v, leaf)
    h['rf_leaf'] = best[1]
    log(f'{name} 超参: {h}')
    return h


# ---------------------------------------------------------------- 工具

def pred_metrics(ytrue, ypred):
    return {'acc': float(accuracy_score(ytrue, ypred)),
            'precision': float(precision_score(ytrue, ypred, average='macro',
                                               zero_division=0)),
            'recall': float(recall_score(ytrue, ypred, average='macro',
                                         zero_division=0)),
            'f1': float(f1_score(ytrue, ypred, average='macro'))}


def em(r):
    """evaluate()/evaluate_peft() 返回值 → 主表行指标（统一键名）。"""
    return {'acc': r['acc'], 'precision': r['precision_macro'],
            'recall': r['recall_macro'], 'f1': r['f1_macro']}


def done_keys(path, keycols):
    keys = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path, encoding='utf-8-sig')):
            keys.add(tuple(str(r[c]) for c in keycols))
    return keys


class Appender:
    """追加式 CSV writer（存在则续写，不存在则写表头）。"""

    def __init__(self, path, header):
        self.path = path
        new = not os.path.exists(path)
        self.f = open(path, 'a', newline='', encoding='utf-8-sig')
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(header)
            self.f.flush()

    def row(self, *vals):
        flat = []
        for v in vals:
            if isinstance(v, (list, tuple)):
                flat.extend(v)
            else:
                flat.append(v)
        self.w.writerow(flat)
        self.f.flush()

    def close(self):
        self.f.close()


def run_ml(kind, Xtr, ytr, Xva, yva, Xeval, yeval, h, seed):
    """LR / SVM / RF：下游数据从头训练（真实运行，含耗时与参数量统计）。"""
    t0 = time.perf_counter()
    try:
        if kind == 'LR':
            clf = LogisticRegression(C=h['C_LR'], max_iter=2000).fit(Xtr, ytr)
            n_par = int(clf.coef_.size + clf.intercept_.size)
        elif kind == 'SVM':
            clf = SVC(C=h['C_SVM'], gamma='scale').fit(Xtr, ytr)
            n_par = int(clf.dual_coef_.size + clf.support_vectors_.size)
        else:
            clf = RandomForestClassifier(n_estimators=200,
                                         min_samples_leaf=h['rf_leaf'],
                                         n_jobs=-1, random_state=seed).fit(Xtr, ytr)
            n_par = int(sum(t.tree_.node_count for t in clf.estimators_))
        tt = time.perf_counter() - t0
        m = pred_metrics(yeval, clf.predict(Xeval))
        ts = []
        for _ in range(5):
            t1 = time.perf_counter()
            clf.predict(Xeval)
            ts.append(time.perf_counter() - t1)
        infer_ms = float(np.median(ts)) / len(Xeval) * 1000.0
        return m, n_par, tt, infer_ms, clf
    except Exception as e:  # 极小样本下偶发单类训练集等退化情形
        log(f'  ML基线 {kind} 失败: {e}')
        return None, 0, time.perf_counter() - t0, 0.0, None


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--seeds', nargs='*', type=int, default=None)
    args = ap.parse_args()

    sfx = '_quick' if args.quick else ''
    datasets = (['Heart', 'WDBC'] if args.quick
                else (args.datasets or ALL_DS))
    seeds = [0] if args.quick else (args.seeds or SEEDS)

    p_main = os.path.join(RESULTS, f'sci_main{sfx}.csv')
    p_noise = os.path.join(RESULTS, f'sci_noise{sfx}.csv')
    p_inoise = os.path.join(RESULTS, f'sci_input_noise{sfx}.csv')
    p_abl = os.path.join(RESULTS, f'sci_ablation{sfx}.csv')
    p_sens = os.path.join(RESULTS, f'sci_sensitivity{sfx}.csv')
    p_h = os.path.join(RESULTS, f'sci_hyperparams{sfx}.json')

    wm = Appender(p_main, ['dataset', 'seed', 'frac', 'method', 'acc', 'precision',
                           'recall', 'f1', 'n_tuned', 'train_time_s', 'infer_ms'])
    wn = Appender(p_noise, ['dataset', 'seed', 'noise_p', 'method', 'f1', 'acc'])
    wi = Appender(p_inoise, ['dataset', 'seed', 'sigma', 'method', 'f1', 'acc'])
    wa = Appender(p_abl, ['dataset', 'seed', 'variant', 'f1', 'acc', 'sparsity',
                          'w_min', 'w_max'])
    done_m = done_keys(p_main, ['dataset', 'seed', 'frac', 'method']) if args.resume else set()
    done_n = done_keys(p_noise, ['dataset', 'seed', 'noise_p', 'method']) if args.resume else set()
    done_i = done_keys(p_inoise, ['dataset', 'seed', 'sigma', 'method']) if args.resume else set()
    done_a = done_keys(p_abl, ['dataset', 'seed', 'variant']) if args.resume else set()

    hparams = (json.load(open(p_h, encoding='utf-8')) if os.path.exists(p_h) else {})
    meta_all = {'pretrain': [], 'convergence': {}, 'weights': {},
                'protocol': dict(seeds=seeds, fractions=FRACTIONS, batch=BATCH,
                                 lfw_epochs=LFW_EPOCHS, peft_epochs=PEFT_EPOCHS)}

    for name in datasets:
        log(f'===== {name} =====')
        if name not in hparams:
            m0, _ = get_pretrained(name, 0)
            hparams[name] = select_hyperparams(name, get_D(name, 0), m0)
            json.dump(hparams, open(p_h, 'w'), indent=1)
        h = hparams[name]
        for seed in seeds:
            model, pinfo = get_pretrained(name, seed)
            meta_all['pretrain'].append(pinfo)
            D = get_D(name, seed)
            d = D['Xpool'].shape[1]
            n_total = model.n_params()
            for frac in FRACTIONS[name]:
                idx = sample_fraction(D['Xpool'], D['ypool'], frac, seed=seed)
                Xa, ya = D['Xpool'][idx], D['ypool'][idx]
                i_tr, i_va = stratified_split(ya, 0.8, seed + 7)
                Xtr, ytr, Xva, yva = Xa[i_tr], ya[i_tr], Xa[i_va], ya[i_va]
                rec_conv = (name in CONV_DS and seed == 0 and frac == 1.0)
                do_viz = (name in VIZ_DS and seed == 0 and frac == 1.0)

                def row(method, m, n_tuned, tt, ims):
                    if (name, str(seed), str(frac), method) in done_m:
                        return
                    wm.row([name, seed, frac, method, f'{m["acc"]:.6f}',
                            f'{m["precision"]:.6f}', f'{m["recall"]:.6f}',
                            f'{m["f1"]:.6f}', n_tuned, f'{tt:.2f}', f'{ims:.4f}'])

                # 1) ZeroShot：冻结直接推理
                row('ZeroShot', em(evaluate(model, D['Xeval'], D['yeval'])),
                    0, 0.0, infer_time(model, D['Xeval']))
                # 2/3) 静态筛选（传统过滤式）
                var = Xtr.var(axis=0)
                mi = mutual_info_classif(Xtr, ytr, random_state=seed)
                for meth, score, k in (('VarFilter', var, h['k_var']),
                                       ('MIFilter', mi, h['k_mi'])):
                    kk = max(1, int(round(d * k)))
                    mask = np.zeros(d); mask[np.argsort(-score)[:kk]] = 1.0
                    r = evaluate(model, D['Xeval'], D['yeval'], mask)
                    row(meth, em(r), 0, 0.0, infer_time(model, D['Xeval'], mask))
                # 4/5/6) LR / SVM / RF（传统 ML）
                for meth in ('LR', 'SVM', 'RF'):
                    m, n_par, tt, ims, _ = run_ml(meth, Xtr, ytr, Xva, yva,
                                                  D['Xeval'], D['yeval'], h, seed)
                    if m is not None:
                        row(meth, m, n_par, tt, ims)
                # 7-11) PEFT 基线（近年 SOTA）
                pm_bitfit = None
                for meth, kind in PEFT_KIND.items():
                    if (name, str(seed), str(frac), meth) in done_m and meth != 'BitFit':
                        continue
                    pm = PEFTModel(model, kind)
                    _, _, tt_peft, hist_peft = train_peft(
                        pm, Xtr, ytr, Xva, yva, lr=h[f'lr_{meth}'],
                        epochs=PEFT_EPOCHS, batch=BATCH, patience=PEFT_PAT,
                        seed=seed, record=rec_conv and meth == 'SSF')
                    r = evaluate_peft(pm, D['Xeval'], D['yeval'])
                    row(meth, em(r), pm.n_tuned(), tt_peft,
                        infer_time_peft(pm, D['Xeval']))
                    if meth == 'BitFit':
                        pm_bitfit = pm
                    if rec_conv and meth == 'SSF':
                        meta_all['convergence'].setdefault(name, {})['ssf'] = hist_peft
                # 12) 本文 LFW
                w, _, _, tt_lfw, hist_lfw = train_weights(
                    model, Xtr, ytr, Xva, yva, l1=h['l1'], l2=h['l2'],
                    epochs=LFW_EPOCHS, batch=BATCH, lr=0.05, patience=LFW_PAT,
                    seed=seed, record=rec_conv)
                r = evaluate(model, D['Xeval'], D['yeval'], w)
                row('LFW(ours)', em(r), d, tt_lfw,
                    infer_time(model, D['Xeval'], w))
                if frac == 1.0:
                    meta_all['weights'].setdefault(name, {})[seed] = w.tolist()
                if rec_conv:
                    meta_all['convergence'].setdefault(name, {})['ours'] = hist_lfw
                # 13) FFT 全参数微调
                ft, _, _, tt_ft, hist_ft = finetune(
                    model, Xtr, ytr, Xva, yva, epochs=LFW_EPOCHS, batch=BATCH,
                    lr=h['lr_FFT'], patience=LFW_PAT, seed=seed, record=rec_conv)
                r = evaluate(ft, D['Xeval'], D['yeval'])
                row('FFT', em(r), ft.n_params(), tt_ft, infer_time(ft, D['Xeval']))
                if rec_conv:
                    meta_all['convergence'][name]['fft'] = hist_ft
                log(f'{name} s{seed} f{frac} 主实验完成')

                # ---------- 可视化数据导出（混淆矩阵 / t-SNE）
                if do_viz:
                    _, A0 = model.forward(D['Xeval'])
                    _, A1 = model.forward(D['Xeval'], w)
                    _, A2 = ft.forward(D['Xeval'])
                    np.savez(os.path.join(RESULTS, f'sci_viz_{name}{sfx}.npz'),
                             feats_none=A0[-2], feats_lfw=A1[-2], feats_fft=A2[-2],
                             ytrue=D['yeval'],
                             pred_none=A0[-1].argmax(1), pred_lfw=A1[-1].argmax(1),
                             pred_fft=A2[-1].argmax(1), w_lfw=w)
                    log(f'{name} 可视化数据已导出')

                # ---------- 标签噪声（frac=0.2，全种子）
                if frac == 0.2:
                    for p in (0.1, 0.2, 0.3):
                        rng = np.random.default_rng(seed * 10 + int(p * 100))
                        yn = ytr.copy()
                        flip = rng.random(len(yn)) < p
                        alt = (yn + rng.integers(1, D['n_class'], size=len(yn))) % D['n_class']
                        yn[flip] = alt[flip]
                        # LFW
                        if (name, str(seed), str(p), 'LFW(ours)') not in done_n:
                            w_, _, _, _, _ = train_weights(
                                model, Xtr, yn, Xva, yva, l1=h['l1'], l2=h['l2'],
                                epochs=LFW_EPOCHS, batch=BATCH, lr=0.05,
                                patience=LFW_PAT, seed=seed)
                            r = evaluate(model, D['Xeval'], D['yeval'], w_)
                            wn.row([name, seed, p, 'LFW(ours)',
                                    f'{r["f1_macro"]:.6f}', f'{r["acc"]:.6f}'])
                        # FFT
                        if (name, str(seed), str(p), 'FFT') not in done_n:
                            ftm, _, _, _, _ = finetune(
                                model, Xtr, yn, Xva, yva, epochs=LFW_EPOCHS,
                                batch=BATCH, lr=h['lr_FFT'], patience=LFW_PAT,
                                seed=seed)
                            r = evaluate(ftm, D['Xeval'], D['yeval'])
                            wn.row([name, seed, p, 'FFT',
                                    f'{r["f1_macro"]:.6f}', f'{r["acc"]:.6f}'])
                        # BitFit
                        if (name, str(seed), str(p), 'BitFit') not in done_n:
                            pmb = PEFTModel(model, 'bitfit')
                            train_peft(pmb, Xtr, yn, Xva, yva, lr=h['lr_BitFit'],
                                       epochs=PEFT_EPOCHS, batch=BATCH,
                                       patience=PEFT_PAT, seed=seed)
                            r = evaluate_peft(pmb, D['Xeval'], D['yeval'])
                            wn.row([name, seed, p, 'BitFit',
                                    f'{r["f1_macro"]:.6f}', f'{r["acc"]:.6f}'])
                    log(f'{name} s{seed} 标签噪声完成')

                    # ---------- 输入高斯噪声（评估期特征扰动）
                    for sg in (0.1, 0.5, 1.0):
                        rng = np.random.default_rng(seed * 1000 + int(sg * 100))
                        Xn = D['Xeval'] + rng.normal(0, sg, D['Xeval'].shape)
                        for meth in ('ZeroShot', 'LFW(ours)', 'FFT', 'BitFit'):
                            if (name, str(seed), str(sg), meth) in done_i:
                                continue
                            if meth == 'ZeroShot':
                                r = evaluate(model, Xn, D['yeval'])
                            elif meth == 'LFW(ours)':
                                r = evaluate(model, Xn, D['yeval'], w)
                            elif meth == 'FFT':
                                r = evaluate(ft, Xn, D['yeval'])
                            else:
                                r = evaluate_peft(pm_bitfit, Xn, D['yeval'])
                            wi.row([name, seed, sg, meth,
                                    f'{r["f1_macro"]:.6f}', f'{r["acc"]:.6f}'])
                    log(f'{name} s{seed} 输入噪声完成')

                    # ---------- 递进式消融
                    for vname, l1, l2, nn, sg2 in (
                            ('none', 0.0, 0.0, False, False),
                            ('w-plain', 0.0, 0.0, False, False),
                            ('w-nonneg', 0.0, 0.0, True, False),
                            ('w-reg', None, None, False, False),
                            ('w-reg-nonneg', None, None, True, False),
                            ('full', None, None, True, True),
                            ('w-affine', None, None, True, True)):
                        if (name, str(seed), vname) in done_a:
                            continue
                        l1v = h['l1'] if l1 is None else l1
                        l2v = h['l2'] if l2 is None else l2
                        if vname == 'none':
                            r = evaluate(model, D['Xeval'], D['yeval'])
                            wv = np.ones(d)
                            tt = 0.0
                        elif vname == 'w-affine':
                            pm = PEFTModel(model, 'input-affine')
                            train_peft(pm, Xtr, ytr, Xva, yva, lr=0.05,
                                       epochs=LFW_EPOCHS, batch=BATCH,
                                       patience=LFW_PAT, seed=seed,
                                       l1=l1v, l2=l2v, nonneg=True)
                            r = evaluate_peft(pm, D['Xeval'], D['yeval'])
                            wv = pm.p[0]
                        else:
                            wv, _, _, _, _ = train_weights(
                                model, Xtr, ytr, Xva, yva, l1=l1v, l2=l2v,
                                nonneg=nn, safeguard=sg2,
                                epochs=LFW_EPOCHS, batch=BATCH, lr=0.05,
                                patience=LFW_PAT, seed=seed)
                            r = evaluate(model, D['Xeval'], D['yeval'], wv)
                        wa.row([name, seed, vname, f'{r["f1_macro"]:.6f}',
                                f'{r["acc"]:.6f}',
                                f'{float((wv < 0.05).mean()):.4f}',
                                f'{float(wv.min()):.4f}', f'{float(wv.max()):.4f}'])
                    log(f'{name} s{seed} 消融完成')

    # ---------- λ 敏感性（HAR + Bank，seed 0，frac 0.5）
    sens_ds = ['Heart', 'WDBC'] if args.quick else ['HAR', 'Bank']
    fs = open(p_sens, 'w', newline='', encoding='utf-8-sig')
    ws_ = csv.writer(fs)
    ws_.writerow(['dataset', 'l1', 'l2', 'f1', 'sparsity'])
    for name in sens_ds:
        model, _ = get_pretrained(name, 0)
        D = get_D(name, 0)
        idx = sample_fraction(D['Xpool'], D['ypool'], 0.5, seed=0)
        Xa, ya = D['Xpool'][idx], D['ypool'][idx]
        i_tr, i_va = stratified_split(ya, 0.8, seed=0)
        for l1 in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
            for l2 in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
                w, _, _, _, _ = train_weights(
                    model, Xa[i_tr], ya[i_tr], Xa[i_va], ya[i_va],
                    l1=l1, l2=l2, epochs=LFW_EPOCHS, batch=BATCH, lr=0.05,
                    patience=LFW_PAT, seed=0)
                r = evaluate(model, D['Xeval'], D['yeval'], w)
                ws_.writerow([name, l1, l2, f'{r["f1_macro"]:.6f}',
                              f'{float((w < 0.05).mean()):.4f}'])
                fs.flush()
        log(f'λ 敏感性 {name} 完成')
    fs.close()

    for a in (wm, wn, wi, wa):
        a.close()
    # 权重与特征名
    np.savez(os.path.join(RESULTS, f'sci_weights{sfx}.npz'),
             **{f'{k}__s{s}': np.array(v)
                for k, per in meta_all['weights'].items()
                for s, v in per.items()})
    json.dump({k: get_D(k, 0)['names'] for k in meta_all['weights']},
              open(os.path.join(RESULTS, f'sci_feature_names{sfx}.json'), 'w',
                   encoding='utf-8'), ensure_ascii=False)
    json.dump(meta_all, open(os.path.join(RESULTS, f'sci_meta{sfx}.json'), 'w',
                             encoding='utf-8'), ensure_ascii=False, indent=1)
    log('全部 SCI 实验完成。')


if __name__ == '__main__':
    main()
