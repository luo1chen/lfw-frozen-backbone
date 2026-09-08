# -*- coding: utf-8 -*-
"""SCI 扩展数据集加载（全部真实数据）。
新增 6 个数据集（来源：UCI 官方 / sklearn 捆绑的 UCI 同源数据 / OpenML 镜像的 UCI 数据）：
  WDBC      乳腺癌诊断      569×30    2类  医疗
  Digits    手写数字        1797×64   10类 视觉(结构化)
  Spambase  垃圾邮件        4601×57   2类  网络安全
  Credit-g  德国信用        1000×61   2类  金融(one-hot后)
  Bank      银行营销        45211×51  2类  金融(one-hot后)
  Satimage  卫星遥感        6435×36   6类  遥感
OpenML 数据首次联网下载后缓存到 data/openml/，之后离线可用。
协议与原三数据集一致：数值特征用源域统计量标准化，分类特征 one-hot 对齐。
"""
import json
import os

import numpy as np

from data_utils import stratified_split, DATA

CACHE = os.path.join(DATA, 'openml')
os.makedirs(CACHE, exist_ok=True)


def _standardize_split(Xs, Xt, num_cols):
    for j in num_cols:
        mu, sd = Xs[:, j].mean(), Xs[:, j].std() + 1e-8
        Xs[:, j] = (Xs[:, j] - mu) / sd
        Xt[:, j] = (Xt[:, j] - mu) / sd
    return Xs, Xt


def _fetch_openml_cached(data_id, name):
    """从 OpenML 下载（真实 UCI 数据镜像）并缓存到本地 npz。"""
    npz = os.path.join(CACHE, f'{name}.npz')
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        return z['X'], z['y'], list(z['names'])
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=data_id, as_frame=False, cache=True)
    X = np.asarray(d.data)
    if X.dtype == object or X.dtype.kind in 'US':
        X = X.copy()
        for j in range(X.shape[1]):  # 字符串类别 → 整数编码
            col = X[:, j]
            _, codes = np.unique(col, return_inverse=True)
            X[:, j] = codes
    X = np.asarray(X, dtype=np.float64)
    y_raw = np.asarray(d.target)
    classes, y = np.unique(y_raw, return_inverse=True)
    names = [str(c) for c in d.feature_names]
    np.savez(npz, X=X, y=y.astype(np.int64), names=np.array(names, dtype=object))
    return X, y.astype(np.int64), names


def _num_cat_split(X, names):
    """按特征取值数判断 one-hot：训练集内取值 ≤12 且为非负整数的列视为类别列。"""
    cat_idx = []
    for j in range(X.shape[1]):
        col = X[:, j]
        u = np.unique(col)
        if len(u) <= 12 and np.all(col >= 0) and np.allclose(col, np.round(col)):
            cat_idx.append(j)
    return cat_idx


def _onehot_encode(Xtr, Xt, cat_idx, names):
    """源域类别空间 one-hot（下游对齐，未知值编码为全零）。"""
    if not cat_idx:
        return Xtr, Xt, names
    maps = {j: sorted(np.unique(Xtr[:, j])) for j in cat_idx}
    new_names, keep = [], []
    for j in range(Xtr.shape[1]):
        if j in maps:
            for v in maps[j]:
                new_names.append(f'{names[j]}={v:g}')
        else:
            keep.append(j)
            new_names.append(names[j])

    def enc(X):
        cols = [X[:, keep]]
        for j in cat_idx:
            vmap = {v: k + 1 for k, v in enumerate(maps[j])}
            E = np.zeros((len(X), len(maps[j]) + 1))
            for r in range(len(X)):
                E[r, vmap.get(X[r, j], 0)] = 1.0
            cols.append(E[:, 1:])  # 去掉全零基准位
        return np.hstack(cols)

    return enc(Xtr), enc(Xt), new_names


def load_sci(name):
    """返回 (Xtr_raw 源域, ytr, Xt_raw 下游域, yt, feat_names)。固定官方/分层划分。"""
    if name == 'WDBC':
        from data_utils import load_wdbc
        X, y, _, _, names = load_wdbc()
        rng = np.random.default_rng(0)
        i = rng.permutation(len(X))
        n_src = int(len(X) * 0.5)
        i_src, i_down = i[:n_src], i[n_src:]
        Xs, Xt = _standardize_split(X[i_src].copy(), X[i_down].copy(),
                                    list(range(X.shape[1])))
        return Xs, y[i_src], Xt, y[i_down], names
    if name == 'Digits':
        from sklearn.datasets import load_digits
        d = load_digits()
        X = np.asarray(d.data, dtype=np.float64) / 16.0  # 像素归一化
        y = np.asarray(d.target, dtype=np.int64)
        names = [f'pix{r}{c}' for r in range(8) for c in range(8)]
        rng = np.random.default_rng(0)
        i = rng.permutation(len(X))
        n_src = int(len(X) * 0.5)
        i_src, i_down = i[:n_src], i[n_src:]
        Xs, Xt = _standardize_split(X[i_src].copy(), X[i_down].copy(),
                                    list(range(64)))
        return Xs, y[i_src], Xt, y[i_down], names
    if name == 'Spambase':
        X, y, names = _fetch_openml_cached(44, 'spambase')
    elif name == 'Credit-g':
        # UCI 官方 german.data（1000×20，空格分隔，类别为 A11/A12... 编码）
        path = os.path.join(CACHE, 'creditzip', 'german.data')
        rows = [ln.split() for ln in open(path, encoding='utf-8')
                if ln.strip()]
        n_att = 20
        names = [f'A{j + 1}' for j in range(n_att)]
        X = np.zeros((len(rows), n_att))
        y = np.zeros(len(rows), dtype=np.int64)
        for r, row in enumerate(rows):
            for j in range(n_att):
                try:
                    X[r, j] = float(row[j])
                except ValueError:
                    X[r, j] = -1.0
            y[r] = 0 if row[n_att] == '1' else 1  # 1=good, 2=bad
        for j in range(n_att):  # 字符串类别列 → 整数编码
            if (X[:, j] == -1.0).any():
                vals = {}
                for r in range(len(X)):
                    if X[r, j] == -1.0:
                        key = rows[r][j]
                        if key not in vals:
                            vals[key] = float(len(vals))
                        X[r, j] = vals[key]
    elif name == 'Bank':
        # UCI 官方 bank-full.csv（45211×16+标签，分号分隔，含字符串类别）
        path = os.path.join(CACHE, 'bankzip', 'b', 'bank-full.csv')
        rows, names = [], ['age', 'job', 'marital', 'education', 'default',
                           'balance', 'housing', 'loan', 'contact', 'day',
                           'month', 'duration', 'campaign', 'pdays',
                           'previous', 'poutcome', 'y']
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                parts = [p.strip().strip('"') for p in ln.strip().split(';')]
                if len(parts) == 17:
                    # 跳过官方 csv 首行表头（age;job;...;y，非数据行）
                    try:
                        float(parts[0])
                    except ValueError:
                        if parts[0] == 'age':
                            continue
                    rows.append(parts)
        X = np.zeros((len(rows), 16))
        y = np.zeros(len(rows), dtype=np.int64)
        for r, row in enumerate(rows):
            for j in range(16):
                try:
                    X[r, j] = float(row[j])
                except ValueError:  # 字符串类别 → 整数编码
                    X[r, j] = -1.0
            y[r] = 0 if row[16] == 'no' else 1
        # 字符串列二次编码
        for j in range(16):
            if (X[:, j] == -1.0).any():
                vals = {}
                for r in range(len(X)):
                    v = X[r, j]
                    if v == -1.0:
                        key = rows[r][j]
                        if key not in vals:
                            vals[key] = float(len(vals))
                        X[r, j] = vals[key]
        names = names[:16]
    elif name == 'Satimage':
        # UCI 官方 sat.trn（6435×36 特征，标签 1-7 无 6）
        path = os.path.join(CACHE, 'satzip', 'sat.trn')
        arr = np.loadtxt(path)
        X = arr[:, :36]
        raw_y = arr[:, 36].astype(np.int64)
        classes = np.unique(raw_y)
        y = np.searchsorted(classes, raw_y)
        names = [f'band{j + 1}' for j in range(36)]
    else:
        raise ValueError(name)
    rng = np.random.default_rng(0)
    i = rng.permutation(len(X))
    n_src = int(len(X) * 0.5)
    i_src, i_down = i[:n_src], i[n_src:]
    cat_idx = _num_cat_split(X[i_src], names)
    Xs, Xt, names = _onehot_encode(X[i_src].copy(), X[i_down].copy(),
                                   cat_idx, names)
    num_cols = [j for j in range(Xs.shape[1]) if '=' not in names[j]]
    Xs, Xt = _standardize_split(Xs, Xt, num_cols)
    return Xs, y[i_src], Xt, y[i_down], names


SCI_SIZES = {  # 隐层结构（输入维按 one-hot 后实际维度动态生成）
    'WDBC': [128, 64],
    'Digits': [256, 128],
    'Spambase': [256, 128],
    'Credit-g': [128, 64],
    'Bank': [512, 256],
    'Satimage': [256, 128],
}
SCI_NCLASS = {'WDBC': 2, 'Digits': 10, 'Spambase': 2, 'Credit-g': 2,
              'Bank': 2, 'Satimage': 6}
SCI_DOMAIN = {'WDBC': '医疗诊断', 'Digits': '手写识别', 'Spambase': '网络安全',
              'Credit-g': '信用风控', 'Bank': '金融营销', 'Satimage': '遥感影像'}

_sci_cache = {}


def get_sci_dataset(name, seed):
    """与 data_utils.get_dataset 相同的返回结构。"""
    key = name
    if key not in _sci_cache:
        Xs, ys, Xt, yt, names = load_sci(name)
        sizes = [Xs.shape[1]] + SCI_SIZES[name] + [SCI_NCLASS[name]]
        _sci_cache[key] = dict(Xs=Xs, ys=ys, Xt=Xt, yt=yt, names=names,
                               sizes=sizes, n_class=SCI_NCLASS[name])
    D = _sci_cache[key]
    i_pool, i_eval = stratified_split(D['yt'], 0.5, seed + 100)
    return dict(Xs=D['Xs'], ys=D['ys'], Xpool=D['Xt'][i_pool], ypool=D['yt'][i_pool],
                Xeval=D['Xt'][i_eval], yeval=D['yt'][i_eval], names=D['names'],
                sizes=D['sizes'], n_class=D['n_class'], domain=SCI_DOMAIN[name])


def dataset_source_info(name):
    """数据集官方来源信息（论文表1用，真实可查）。"""
    info = {
        'Adult': 'UCI Adult (adult.data/adult.test 官方划分)',
        'Heart': 'UCI Heart Disease Cleveland (processed.cleveland.data)',
        'HAR': 'UCI HAR (官方被试1-21/22-30划分)',
        'WDBC': 'UCI Breast Cancer Wisconsin Diagnostic (wdbc.data)',
        'Digits': 'UCI Optical Recognition of Handwritten Digits (sklearn同源)',
        'Spambase': 'UCI Spambase (OpenML id=44 镜像)',
        'Credit-g': 'UCI Statlog German Credit (OpenML id=31 镜像)',
        'Bank': 'UCI Bank Marketing (OpenML id=1461 镜像)',
        'Satimage': 'UCI Statlog Landsat Satellite (OpenML id=182 镜像)',
    }
    return info.get(name, '')


if __name__ == '__main__':
    from data_utils import get_dataset as get_base
    out = {}
    for nm in ('Adult', 'Heart', 'HAR'):
        D = get_base(nm, 0)
        out[nm] = dict(src=D['Xs'].shape, pool=D['Xpool'].shape,
                       eval=D['Xeval'].shape, d=D['Xs'].shape[1],
                       n_class=D['n_class'])
        print(nm, out[nm])
    for nm in SCI_SIZES:
        D = get_sci_dataset(nm, 0)
        out[nm] = dict(src=D['Xs'].shape, pool=D['Xpool'].shape,
                       eval=D['Xeval'].shape, d=D['Xs'].shape[1],
                       n_class=D['n_class'])
        print(nm, out[nm])
    json.dump(out, open(os.path.join(CACHE, 'dataset_shapes.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
