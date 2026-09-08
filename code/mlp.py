# -*- coding: utf-8 -*-
"""numpy 实现的 MLP 主干网络、Adam 优化器与训练流程。
包含：预训练（全参数）、冻结主干下的特征权重训练（本文方法）、全参数微调（基线）。
"""
import time
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def relu(x):
    return np.maximum(0.0, x)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.lr, self.t, self.b1, self.b2, self.eps = lr, 0, betas[0], betas[1], eps

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * (g * g)
            mhat = m / (1 - self.b1 ** self.t)
            vhat = v / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class MLP:
    """多层全连接网络：ReLU 中间层，输出 logits。"""

    def __init__(self, sizes, seed=0, l2=0.0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.l2 = l2

    def copy(self):
        m = MLP(self.sizes, seed=0, l2=self.l2)
        m.W = [w.copy() for w in self.W]
        m.b = [b.copy() for b in self.b]
        return m

    def params(self):
        return self.W + self.b

    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

    def forward(self, X, w=None):
        Xin = X if w is None else X * w
        A = [Xin]
        a = Xin
        L = len(self.W)
        for i in range(L):
            z = a @ self.W[i] + self.b[i]
            a = relu(z) if i < L - 1 else z
            A.append(a)
        return a, A

    def backward(self, A, X, dlogits, w=None, input_only=False):
        """input_only=True 时仅计算输入层梯度 dw（本文方法训练用），
        跳过主干参数梯度 gW/gb 的计算——主干冻结无需这些梯度，可节省约半次反向传播开销。"""
        L = len(self.W)
        gW = [None] * L if not input_only else None
        gb = [None] * L if not input_only else None
        delta = dlogits
        for i in range(L - 1, -1, -1):
            if not input_only:
                gW[i] = A[i].T @ delta
                if self.l2:
                    gW[i] = gW[i] + self.l2 * self.W[i]
                gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (A[i] > 0)
        dXin = delta @ self.W[0].T
        dw = (dXin * X).sum(axis=0) if w is not None else None
        return gW, gb, dw


# ---------- 损失与指标 ----------

def ce_loss(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(y)), y].mean()


def dce(logits, y):
    p = softmax(logits)
    p[np.arange(len(y)), y] -= 1.0
    return p / len(y)


def evaluate(model, X, y, w=None):
    logits, _ = model.forward(X, w)
    pred = logits.argmax(axis=1)
    return {
        'acc': float(accuracy_score(y, pred)),
        'f1_macro': float(f1_score(y, pred, average='macro')),
        'f1_weighted': float(f1_score(y, pred, average='weighted')),
        'precision_macro': float(precision_score(y, pred, average='macro',
                                                 zero_division=0)),
        'recall_macro': float(recall_score(y, pred, average='macro',
                                           zero_division=0)),
    }


def infer_time(model, X, w=None, repeats=5):
    """批量推理耗时：对整个 X 前向 repeats 次取中位数，返回每样本毫秒。"""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.forward(X, w)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / len(X) * 1000.0


def iterate_minibatches(n, batch, rng):
    idx = rng.permutation(n)
    for s in range(0, n, batch):
        yield idx[s:s + batch]


# ---------- 训练流程 ----------

def pretrain(model, Xtr, ytr, Xval, yval, epochs=300, batch=128, lr=1e-3,
             patience=20, seed=0):
    """源域全参数预训练（早停）。返回最佳验证 F1 与停止轮次。"""
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr)
    best_f1, best_W, best_b, best_ep, bad = -1.0, None, None, 0, 0
    for ep in range(epochs):
        for bidx in iterate_minibatches(len(Xtr), batch, rng):
            logits, A = model.forward(Xtr[bidx])
            gW, gb, _ = model.backward(A, None, dce(logits, ytr[bidx]))
            opt.step(gW + gb)
        v = evaluate(model, Xval, yval)['f1_macro']
        if v > best_f1:
            best_f1, best_ep, bad = v, ep, 0
            best_W = [x.copy() for x in model.W]
            best_b = [x.copy() for x in model.b]
        else:
            bad += 1
            if bad >= patience:
                break
    model.W, model.b = best_W, best_b
    return best_f1, best_ep


def train_weights(model, Xtr, ytr, Xval, yval, l1=0.0, l2=0.0, nonneg=True,
                  epochs=300, batch=128, lr=0.05, patience=20, seed=0,
                  record=False, safeguard=True):
    """冻结主干，仅训练输入特征权重 w（本文方法）。
    w 经投影保证非负（nonneg=True）；L1/L2 正则作用于 w。
    safeguard=True 时启用单位权重保底机制：以 w=1（等价零样本推理）为第 0 轮候选，
    训练仅在验证集 F1 超过保底值时才采纳学习到的权重，确保适配不劣化；
    safeguard=False 时（消融用）无条件返回学习到的权重。
    返回 (w, best_f1, epochs_run, train_time, history)。
    """
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    w = np.ones(d)
    opt = Adam([w], lr)
    v0 = evaluate(model, Xval, yval, w)['f1_macro'] if safeguard else -1.0
    best_f1, best_w, best_ep, bad = v0, w.copy(), 0, 0
    t0 = time.perf_counter()
    hist = {'val_f1': [v0], 'train_loss': [None], 'cum_time': [0.0]}
    for ep in range(epochs):
        losses = []
        for bidx in iterate_minibatches(len(Xtr), batch, rng):
            Xb, yb = Xtr[bidx], ytr[bidx]
            logits, A = model.forward(Xb, w)
            losses.append(ce_loss(logits, yb))
            _, _, dw = model.backward(A, Xb, dce(logits, yb), w, input_only=True)
            if nonneg:
                g = dw + 2.0 * l2 * w + l1 * np.where(w > 0, 1.0, 0.0)
            else:
                g = dw + 2.0 * l2 * w + l1 * np.sign(w)
            opt.step([g])
            if nonneg:
                np.maximum(w, 0.0, out=w)
        v = evaluate(model, Xval, yval, w)['f1_macro']
        if record:
            hist['val_f1'].append(v)
            hist['train_loss'].append(float(np.mean(losses)))
            hist['cum_time'].append(time.perf_counter() - t0)
        if v > best_f1:
            best_f1, best_ep, bad = v, ep, 0
            best_w = w.copy()
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.perf_counter() - t0
    return best_w, best_f1, best_ep, train_time, hist


def finetune(model0, Xtr, ytr, Xval, yval, epochs=300, batch=128, lr=1e-3,
             patience=20, seed=0, record=False):
    """全参数微调基线（复制主干后更新全部参数）。"""
    model = model0.copy()
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr)
    best_f1, best_W, best_b, best_ep, bad = -1.0, None, None, 0, 0
    t0 = time.perf_counter()
    hist = {'val_f1': [], 'train_loss': [], 'cum_time': []}
    for ep in range(epochs):
        losses = []
        for bidx in iterate_minibatches(len(Xtr), batch, rng):
            Xb, yb = Xtr[bidx], ytr[bidx]
            logits, A = model.forward(Xb)
            losses.append(ce_loss(logits, yb))
            gW, gb, _ = model.backward(A, None, dce(logits, yb))
            opt.step(gW + gb)
        v = evaluate(model, Xval, yval)['f1_macro']
        if record:
            hist['val_f1'].append(v)
            hist['train_loss'].append(float(np.mean(losses)))
            hist['cum_time'].append(time.perf_counter() - t0)
        if v > best_f1:
            best_f1, best_ep, bad = v, ep, 0
            best_W = [x.copy() for x in model.W]
            best_b = [x.copy() for x in model.b]
        else:
            bad += 1
            if bad >= patience:
                break
    model.W, model.b = best_W, best_b
    train_time = time.perf_counter() - t0
    return model, best_f1, best_ep, train_time, hist


# ---------- 梯度自检 ----------

def gradcheck():
    rng = np.random.default_rng(0)
    model = MLP([6, 9, 4], seed=1)
    X = rng.normal(size=(8, 6))
    y = rng.integers(0, 4, size=8)
    w = rng.uniform(0.5, 1.5, size=6)

    def loss_fn():
        logits, _ = model.forward(X, w)
        return ce_loss(logits, y)

    logits, A = model.forward(X, w)
    gW, gb, dw = model.backward(A, X, dce(logits, y), w)
    eps = 1e-6
    max_rel = 0.0
    for grads, params in ((gW, model.W), (gb, model.b)):
        for g, p in zip(grads, params):
            it = np.nditer(p, flags=['multi_index'])
            for _ in range(min(p.size, 30)):
                ix = it.multi_index
                old = p[ix]
                p[ix] = old + eps; lp = loss_fn()
                p[ix] = old - eps; lm = loss_fn()
                p[ix] = old
                num = (lp - lm) / (2 * eps)
                den = max(1e-8, abs(num) + abs(g[ix]))
                max_rel = max(max_rel, abs(num - g[ix]) / den)
                it.iternext()
    old = w[2]
    w[2] = old + eps; lp = loss_fn()
    w[2] = old - eps; lm = loss_fn()
    w[2] = old
    num = (lp - lm) / (2 * eps)
    max_rel_w = abs(num - dw[2]) / max(1e-8, abs(num) + abs(dw[2]))
    print(f'gradcheck: max_rel_Wb={max_rel:.2e}, rel_w={max_rel_w:.2e}')
    assert max_rel < 1e-5 and max_rel_w < 1e-5, '梯度检验未通过!'
    print('梯度检验通过。')


if __name__ == '__main__':
    gradcheck()
