# -*- coding: utf-8 -*-
"""真实数据集加载与预处理（全部来自 UCI 机器学习数据库官方发布数据）。
数据集：
  1) Adult（金融/人口普查收入预测，二分类）
  2) Breast Cancer Wisconsin Diagnostic（医疗诊断，二分类）
  3) HAR 智能手机人体行为识别（6 分类，跨被试域偏移）
协议：
  - Adult:  源域 = adult.data，下游 = adult.test（池/评估 5:5 分层划分）
  - WDBC:   全量按种子分层 5:5 划分为源域 / 下游（下游再 5:5 分为池/评估）
  - HAR:    源域 = 官方 train（被试 1-21），下游 = 官方 test（被试 22-30，天然域偏移），池/评估 5:5
  - 数值特征用源域统计量标准化；分类特征 one-hot 编码（源/下游类别空间对齐）
"""
import os
import re
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

ADULT_NUM = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
ADULT_CAT = ['workclass', 'education', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'native-country']
ADULT_ALL = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

WDBC_METRICS = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dim']
WDBC_NAMES = [f'{m}({s})' for s in ('mean', 'se', 'worst') for m in WDBC_METRICS]

HEART_COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
              'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
HEART_NUM = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']          # 连续
HEART_BIN = ['sex', 'fbs', 'exang']                                     # 0/1
HEART_CAT = ['cp', 'restecg', 'slope', 'ca', 'thal']                    # 离散(含缺失'?')


def load_heart():
    """UCI Heart Disease (Cleveland): 303 样本、13 特征、二分类(0=无病,1=患病)。
    连续特征标准化（源域统计量）；离散特征 one-hot（'?' 缺失编码为全零向量）。"""
    path = os.path.join(DATA, 'heart', 'processed.cleveland.data')
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(',')]
            if len(parts) != 14:
                continue
            rows.append(parts)
    cat_values = {c: sorted({r[HEART_COLS.index(c)] for r in rows
                             if r[HEART_COLS.index(c)] != '?'}) for c in HEART_CAT}
    feat_names, encoders = [], []
    for c in HEART_COLS[:-1]:
        i = HEART_COLS.index(c)
        if c in HEART_NUM or c in HEART_BIN:
            feat_names.append(c)
            encoders.append(('num', i))
        else:
            for v in cat_values[c]:
                feat_names.append(f'{c}={v}')
                encoders.append(('cat', i, v))

    def encode(rs):
        X = np.zeros((len(rs), len(encoders)), dtype=np.float64)
        y = np.zeros(len(rs), dtype=np.int64)
        for r, row in enumerate(rs):
            for j, e in enumerate(encoders):
                if e[0] == 'num':
                    if row[e[1]] != '?':
                        X[r, j] = float(row[e[1]])
                else:
                    if row[e[1]] == e[2]:
                        X[r, j] = 1.0
            y[r] = 0 if row[13] == '0' else 1
        return X, y

    X, y = encode(rows)
    return X, y, None, None, feat_names


def _parse_adult(path, skip_first=False):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if skip_first:
        lines = lines[1:]
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith('|'):
            continue
        parts = [p.strip().rstrip('.') for p in ln.split(',')]
        if len(parts) != 15:
            continue
        rows.append(parts)
    return rows


def load_adult():
    tr = _parse_adult(os.path.join(DATA, 'adult', 'adult.data'))
    te = _parse_adult(os.path.join(DATA, 'adult', 'adult.test'), skip_first=True)
    allrows = tr + te
    # 类别空间（源+下游对齐）
    cat_values = {c: sorted({r[ADULT_ALL.index(c)] for r in allrows}) for c in ADULT_CAT}
    feat_names, encoders = [], []
    for i, c in enumerate(ADULT_ALL):
        if c in ADULT_CAT:
            for v in cat_values[c]:
                feat_names.append(f'{c}={v}')
                encoders.append(('cat', i, v))
        else:
            feat_names.append(c)
            encoders.append(('num', i, None))

    def encode(rows):
        X = np.zeros((len(rows), len(encoders)), dtype=np.float64)
        y = np.zeros(len(rows), dtype=np.int64)
        for r, row in enumerate(rows):
            for j, (kind, idx, v) in enumerate(encoders):
                if kind == 'num':
                    X[r, j] = float(row[idx])
                else:
                    X[r, j] = 1.0 if row[idx] == v else 0.0
            y[r] = 0 if '<=50' in row[14] else 1
        return X, y

    Xs, ys = encode(tr)
    Xt, yt = encode(te)
    # 数值列索引：用源域统计量标准化
    num_cols = [j for j, e in enumerate(encoders) if e[0] == 'num']
    for j in num_cols:
        mu, sd = Xs[:, j].mean(), Xs[:, j].std() + 1e-8
        Xs[:, j] = (Xs[:, j] - mu) / sd
        Xt[:, j] = (Xt[:, j] - mu) / sd
    return Xs, ys, Xt, yt, feat_names


def load_wdbc():
    """优先读取 UCI 原始 wdbc.data；文件不完整时回退到 sklearn 捆绑的同源数据
    （sklearn 的 breast_cancer 数据集即 UCI WDBC 官方数据，特征顺序一致）。"""
    path = os.path.join(DATA, 'wdbc', 'wdbc.data')
    if os.path.exists(path) and os.path.getsize(path) > 100000:
        X, y = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for ln in f:
                parts = ln.strip().split(',')
                if len(parts) != 32:
                    continue
                X.append([float(v) for v in parts[2:]])
                y.append(0 if parts[1] == 'B' else 1)
        return np.array(X), np.array(y, dtype=np.int64), None, None, WDBC_NAMES
    from sklearn.datasets import load_breast_cancer
    b = load_breast_cancer()
    X = np.asarray(b.data, dtype=np.float64)
    y = (1 - np.asarray(b.target)).astype(np.int64)  # 统一为 0=良性B, 1=恶性M
    names = [n.replace(' ', '_') for n in b.feature_names]
    return X, y, None, None, names


def _load_har_matrix(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append([float(v) for v in ln.split()])
    return np.array(rows)


def load_har():
    root = None
    for cand in (os.path.join(DATA, 'har'), os.path.join(DATA, 'har', 'UCI HAR Dataset')):
        if os.path.exists(os.path.join(cand, 'train', 'X_train.txt')):
            root = cand
            break
    Xs = _load_har_matrix(os.path.join(root, 'train', 'X_train.txt'))
    Xt = _load_har_matrix(os.path.join(root, 'test', 'X_test.txt'))
    ys = np.loadtxt(os.path.join(root, 'train', 'y_train.txt'), dtype=np.int64) - 1
    yt = np.loadtxt(os.path.join(root, 'test', 'y_test.txt'), dtype=np.int64) - 1
    feat_names, act_names = [], {}
    with open(os.path.join(root, 'features.txt'), 'r', encoding='utf-8') as f:
        for ln in f:
            parts = ln.strip().split(' ', 1)
            if len(parts) == 2:
                feat_names.append(parts[1])
    with open(os.path.join(root, 'activity_labels.txt'), 'r', encoding='utf-8') as f:
        for ln in f:
            parts = ln.strip().split(' ', 1)
            if len(parts) == 2:
                act_names[int(parts[0]) - 1] = parts[1]
    # 源域统计量标准化
    mu = Xs.mean(axis=0)
    sd = Xs.std(axis=0) + 1e-8
    Xs = (Xs - mu) / sd
    Xt = (Xt - mu) / sd
    return Xs, ys, Xt, yt, feat_names, act_names


def stratified_split(y, ratio, seed):
    """按类别分层随机划分，返回 (idx1, idx2)，idx1 占比约 ratio。"""
    rng = np.random.default_rng(seed)
    idx1, idx2 = [], []
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        rng.shuffle(ic)
        k = max(1, int(round(len(ic) * ratio)))
        idx1.extend(ic[:k]); idx2.extend(ic[k:])
    # dtype=int：空列表时 np.array 会退化为 float64，导致无法用作索引
    idx1 = np.array(sorted(idx1), dtype=int)
    idx2 = np.array(sorted(idx2), dtype=int)
    return idx1, idx2


def sample_fraction(X, y, frac, seed):
    """从池中按比例分层抽样（作为下游有标注训练集），返回约 frac 比例的索引。
    修正说明：原实现误返回 1-frac 部分（idx2），导致 frac=0.05 实际保留 95% 样本；
    现改为返回 idx1，保证低资源实验使用真实的小样本量。"""
    if frac >= 1.0:
        return np.arange(len(X))
    keep, _ = stratified_split(y, frac, seed)  # idx1 比例约 frac
    return keep


def train_val_split(X, y, val_ratio=0.2, seed=0):
    idx_tr, idx_va = stratified_split(y, 1.0 - val_ratio, seed)
    return idx_tr, idx_va


DATASETS = {}


def get_dataset(name, seed):
    """返回 dict: Xs,ys(源域), Xpool,ypool(下游池), Xeval,yeval(评估集), names, sizes, n_class"""
    key = name
    if name not in DATASETS:
        if name == 'Adult':
            Xs, ys, Xt, yt, names = load_adult()
            sizes = [Xs.shape[1], 512, 256, 128, 2]
            DATASETS[key] = dict(Xs=Xs, ys=ys, Xt=Xt, yt=yt, names=names, sizes=sizes, n_class=2)
        elif name == 'Heart':
            X, y, _, _, names = load_heart()
            DATASETS[key] = dict(rawX=X, rawy=y, names=names,
                                 sizes=[X.shape[1], 128, 64, 2], n_class=2)
        elif name == 'HAR':
            Xs, ys, Xt, yt, names, acts = load_har()
            DATASETS[key] = dict(Xs=Xs, ys=ys, Xt=Xt, yt=yt, names=names, acts=acts,
                                 sizes=[Xs.shape[1], 1024, 512, 6], n_class=6)
        else:
            raise ValueError(name)
    D = DATASETS[key]
    if name == 'Heart':
        i_src, i_down = stratified_split(D['rawy'], 0.5, seed)
        X, y = D['rawX'], D['rawy']
        names = D['names']
        cont_idx = [j for j, n in enumerate(names) if n in HEART_NUM]
        X = X.copy()
        for j in cont_idx:  # 仅连续特征标准化（源域统计量）
            mu, sd = X[i_src, j].mean(), X[i_src, j].std() + 1e-8
            X[:, j] = (X[:, j] - mu) / sd
        Xs, Xt = X[i_src], X[i_down]
        ys, yt = y[i_src], y[i_down]
        i_pool, i_eval = stratified_split(yt, 0.5, seed + 100)
        return dict(Xs=Xs, ys=ys, Xpool=Xt[i_pool], ypool=yt[i_pool],
                    Xeval=Xt[i_eval], yeval=yt[i_eval], names=D['names'],
                    sizes=D['sizes'], n_class=2)
    i_pool, i_eval = stratified_split(D['yt'], 0.5, seed + 100)
    return dict(Xs=D['Xs'], ys=D['ys'], Xpool=D['Xt'][i_pool], ypool=D['yt'][i_pool],
                Xeval=D['Xt'][i_eval], yeval=D['yt'][i_eval], names=D['names'],
                sizes=D['sizes'], n_class=D['n_class'],
                acts=D.get('acts'))


if __name__ == '__main__':
    for nm in ('Adult', 'Heart', 'HAR'):
        D = get_dataset(nm, seed=0)
        print(nm, 'src', D['Xs'].shape, 'pool', D['Xpool'].shape,
              'eval', D['Xeval'].shape, 'classes', np.bincount(D['ypool']))
