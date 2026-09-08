# -*- coding: utf-8 -*-
"""PEFT 基线实现（numpy，与 mlp.MLP 兼容）：Linear Probe / BitFit / SSF / Adapter / LoRA。
全部仅训练各自的可学习参数，主干权重冻结。含数值梯度检验。
"""
import time

import numpy as np

from mlp import Adam, MLP, ce_loss, dce, evaluate, iterate_minibatches, relu, softmax


class PEFTModel:
    """在冻结 MLP 主干之上按 kind 构造可训练参数并执行前向/反向。

    kind:
      probe    Linear Probing：仅重训最后一层分类头
      bitfit   BitFit：仅训练全部偏置 b
      ssf      SSF：每层输出后加可学习逐通道仿射 γ⊙x+β（含输入位置）
      adapter  串行残差 Adapter：倒数第二层特征后接瓶颈残差模块（r=H/8, 初始化恒等）
      lora     LoRA：每层 W' = W + A·B（rank=4, B 零初始化）
      input-affine 本文消融：输入侧可学习仿射 w⊙x+b（w 承接 L1/L2 与非负约束）
    """

    def __init__(self, model: MLP, kind, r=4, adapter_r=None):
        self.m = model
        self.kind = kind
        self.r = r
        rng = np.random.default_rng(0)
        L = len(model.W)
        H = model.sizes[-2]
        self.adapter_r = adapter_r or max(4, H // 8)
        p = []
        if kind == 'probe':
            p = [np.zeros_like(model.W[-1]), np.zeros_like(model.b[-1])]
            p[0][...] = model.W[-1]  # 从预训练头热启动（与主干一致的初始化）
        elif kind == 'bitfit':
            p = [b.copy() for b in model.b]
        elif kind == 'ssf':
            self.ssf_pos = [model.sizes[0]] + [model.sizes[i] for i in range(1, L)]
            p = [np.ones(d) for d in self.ssf_pos] + [np.zeros(d) for d in self.ssf_pos]
        elif kind == 'adapter':
            rr = self.adapter_r
            p = [rng.normal(0, 0.01, (H, rr)) / np.sqrt(H), np.zeros(rr),
                 np.zeros((rr, H)), np.zeros(H)]  # Wu 零初始化 → 恒等启动
        elif kind == 'lora':
            for i in range(L):
                din, dout = model.sizes[i], model.sizes[i + 1]
                p.append(rng.normal(0, 1.0 / r, (din, r)))
                p.append(np.zeros((r, dout)))
        elif kind == 'input-affine':
            d = model.sizes[0]
            p = [np.ones(d), np.zeros(d)]
        else:
            raise ValueError(kind)
        self.p = p

    # ------------------------------------------------ 前向
    def forward(self, X):
        m, p, kind = self.m, self.p, self.kind
        cache = {}
        L = len(m.W)
        if kind in ('probe', 'bitfit', 'input-affine'):
            w = p[0] if kind == 'input-affine' else None
            b0 = p[1] if kind == 'input-affine' else None
            Xin = X * w + b0 if kind == 'input-affine' else X
            a = Xin
            A = [Xin]
            for i in range(L):
                Wi = m.W[i]
                bi = p[i] if kind == 'bitfit' else m.b[i]
                z = a @ Wi + bi
                a = relu(z) if i < L - 1 else z
                A.append(a)
            if kind == 'probe':
                A[-1] = None
                logits = A[-2] @ p[0] + p[1]
            else:
                logits = a
            cache['A'] = A
            cache['X'] = X
            return logits, cache
        if kind == 'ssf':
            gammas, betas = p[:len(self.ssf_pos)], p[len(self.ssf_pos):]
            t = gammas[0] * X + betas[0]
            cache['t'] = [t]
            cache['pre'] = [X]
            logits = None
            for i in range(L):
                z = t @ m.W[i] + m.b[i]
                if i < L - 1:
                    a = relu(z)
                    cache['pre'].append(a)
                    t = gammas[i + 1] * a + betas[i + 1]
                    cache['t'].append(t)
                else:
                    logits = z
            return logits, cache
        if kind == 'adapter':
            A = [X]
            a = X
            for i in range(L - 1):
                a = relu(a @ m.W[i] + m.b[i])
                A.append(a)
            h = A[-1]
            Wd, bd, Wu, bu = p
            inner = relu(h @ Wd + bd)
            t = h + inner @ Wu + bu
            cache['h'], cache['inner'], cache['t'] = h, inner, t
            logits = t @ m.W[-1] + m.b[-1]
            return logits, cache
        if kind == 'lora':
            a = X
            A = [X]
            for i in range(L):
                Wp = m.W[i] + p[2 * i] @ p[2 * i + 1]
                a = relu(a @ Wp + m.b[i]) if i < L - 1 else a @ Wp + m.b[i]
                A.append(a)
            cache['A'] = A
            return a, cache
        raise ValueError(kind)

    # ------------------------------------------------ 反向（仅计算可训练参数梯度）
    def backward(self, cache, dlogits):
        m, p, kind = self.m, self.p, self.kind
        L = len(m.W)
        if kind == 'probe':
            h = cache['A'][-2]
            return [h.T @ dlogits, dlogits.sum(axis=0)]
        if kind == 'input-affine':
            A, X = cache['A'], cache['X']
            delta = dlogits
            for i in range(L - 1, 0, -1):
                delta = (delta @ m.W[i].T) * (A[i] > 0)
            dXin = delta @ m.W[0].T  # dL/dt0, t0 = w⊙X + b
            return [(dXin * X).sum(axis=0), dXin.sum(axis=0)]
        if kind == 'bitfit':
            A = cache['A']
            delta = dlogits
            g = [None] * L
            for i in range(L - 1, -1, -1):
                g[i] = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ m.W[i].T) * (A[i] > 0)
            return g
        if kind == 'ssf':
            t, pre = cache['t'], cache['pre']
            n = len(self.ssf_pos)
            gammas, betas = p[:n], p[n:]
            grads_g, grads_b = [None] * n, [None] * n
            delta_z = dlogits                      # dL/dz_{L-1}
            for i in range(L - 1, 0, -1):
                delta_t = delta_z @ m.W[i].T        # dL/dt_i
                a_prev = pre[i]                     # t_i = γ_i·a_{i-1}+β_i, a_{i-1}=pre[i]
                grads_g[i] = (delta_t * a_prev).sum(axis=0)
                grads_b[i] = delta_t.sum(axis=0)
                delta_a = delta_t * gammas[i]       # dL/da_{i-1}
                delta_z = delta_a * (a_prev > 0)    # dL/dz_{i-1}
            delta_t0 = delta_z @ m.W[0].T           # dL/dt_0
            grads_g[0] = (delta_t0 * pre[0]).sum(axis=0)
            grads_b[0] = delta_t0.sum(axis=0)
            return grads_g + grads_b
        if kind == 'adapter':
            Wd, bd, Wu, bu = p
            h, inner, t = cache['h'], cache['inner'], cache['t']
            dt = dlogits @ m.W[-1].T          # dL/dt
            d_inner = dt @ Wu.T               # dL/d(inner)
            gWu = inner.T @ dt
            gbu = dt.sum(axis=0)
            d_pre = d_inner * (inner > 0)     # dL/d(relu前)
            gWd = h.T @ d_pre
            gbd = d_pre.sum(axis=0)
            return [gWd, gbd, gWu, gbu]
        if kind == 'lora':
            A = cache['A']
            delta = dlogits
            grads = [None] * (2 * L)
            for i in range(L - 1, -1, -1):
                gWp = A[i].T @ delta          # dL/dW'
                grads[2 * i + 1] = p[2 * i].T @ gWp        # gB = A_i^T? → B 梯度
                grads[2 * i] = gWp @ p[2 * i + 1].T        # gA
                if i > 0:
                    delta = (delta @ (m.W[i] + p[2 * i] @ p[2 * i + 1]).T) * (A[i] > 0)
            return grads
        raise ValueError(kind)

    def params(self):
        return self.p

    def n_tuned(self):
        return sum(q.size for q in self.p)


def evaluate_peft(pm: PEFTModel, X, y):
    logits, _ = pm.forward(X)
    pred = logits.argmax(axis=1)
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    return {'acc': float(accuracy_score(y, pred)),
            'f1_macro': float(f1_score(y, pred, average='macro')),
            'f1_weighted': float(f1_score(y, pred, average='weighted')),
            'precision_macro': float(precision_score(y, pred, average='macro',
                                                     zero_division=0)),
            'recall_macro': float(recall_score(y, pred, average='macro',
                                               zero_division=0))}


def infer_time_peft(pm: PEFTModel, X, repeats=5):
    """PEFT 模型批量推理耗时（每样本毫秒，中位数）。"""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pm.forward(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / len(X) * 1000.0


def train_peft(pm: PEFTModel, Xtr, ytr, Xval, yval, lr=1e-3, epochs=100,
               batch=256, patience=10, seed=0, record=False,
               l1=0.0, l2=0.0, nonneg=False):
    """统一训练流程（Adam + 早停）。input-affine 消融用 l1/l2/nonneg。"""
    rng = np.random.default_rng(seed)
    opt = Adam(pm.params(), lr)
    d = pm.m.sizes[0] if pm.kind == 'input-affine' else 0
    v0 = evaluate_peft(pm, Xval, yval)['f1_macro']
    best_f1, best_p, best_ep, bad = v0, [q.copy() for q in pm.p], 0, 0
    t0 = time.perf_counter()
    hist = {'val_f1': [v0], 'train_loss': [None], 'cum_time': [0.0]}
    for ep in range(epochs):
        losses = []
        for bidx in iterate_minibatches(len(Xtr), batch, rng):
            Xb, yb = Xtr[bidx], ytr[bidx]
            logits, cache = pm.forward(Xb)
            losses.append(ce_loss(logits, yb))
            grads = pm.backward(cache, dce(logits, yb))
            if (l1 or l2) and pm.kind == 'input-affine':
                q = pm.p[0]
                sub = np.sign(q) if not nonneg else np.where(q > 0, 1.0, 0.0)
                grads[0] = grads[0] + 2 * l2 * q + l1 * sub
            opt.step(grads)
            if nonneg and pm.kind == 'input-affine':
                np.maximum(pm.p[0], 0.0, out=pm.p[0])
        v = evaluate_peft(pm, Xval, yval)['f1_macro']
        if record:
            hist['val_f1'].append(v)
            hist['train_loss'].append(float(np.mean(losses)))
            hist['cum_time'].append(time.perf_counter() - t0)
        if v > best_f1:
            best_f1, best_ep, bad = v, ep, 0
            best_p = [q.copy() for q in pm.p]
        else:
            bad += 1
            if bad >= patience:
                break
    pm.p = best_p
    return best_f1, best_ep, time.perf_counter() - t0, hist


# ------------------------------------------------ 数值梯度检验（每种 kind）
def gradcheck_peft():
    rng = np.random.default_rng(0)
    model = MLP([6, 9, 8, 4], seed=1)
    X = rng.normal(size=(16, 6))
    y = rng.integers(0, 4, size=16)
    eps = 1e-5  # 1e-6 时差分噪声可达 1e-5 量级（ReLU 拐点），故用 1e-5
    for kind in ('probe', 'bitfit', 'ssf', 'adapter', 'lora', 'input-affine'):
        pm = PEFTModel(model, kind)
        logits, cache = pm.forward(X)
        grads = pm.backward(cache, dce(logits, y))

        def loss_fn():
            lg, _ = pm.forward(X)
            return ce_loss(lg, y)

        max_rel = 0.0
        for q, g in zip(pm.p, grads):
            if g is None:
                continue
            flat = q.reshape(-1)
            gflat = g.reshape(-1)
            for k in rng.choice(flat.size, size=min(12, flat.size), replace=False):
                old = flat[k]
                flat[k] = old + eps
                lp = loss_fn()
                flat[k] = old - eps
                lm = loss_fn()
                flat[k] = old
                num = (lp - lm) / (2 * eps)
                den = max(1e-8, abs(num) + abs(gflat[k]))
                max_rel = max(max_rel, abs(num - gflat[k]) / den)
        status = 'OK' if max_rel < 1e-5 else 'FAIL'
        print(f'{kind:13s} gradcheck max_rel={max_rel:.2e}  {status}')
        assert max_rel < 1e-5, f'{kind} 梯度检验未通过!'


if __name__ == '__main__':
    gradcheck_peft()
