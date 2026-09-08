# -*- coding: utf-8 -*-
"""P1-5 视觉骨干实验：compact ViT 上的 PEFT 基线 vs LFW（真实数据、真实运行、可复现）。

目的：回应可预期审稿意见“BitFit/SSF/LoRA 等为 Transformer 设计，仅在 MLP 上验证不公平”
——在真实视觉 Transformer 骨干上检验核心结论是否成立。

迁移对（标签空间对齐，zero-shot 有意义）：
  a) MNIST(源, 60,000) → USPS(下游)：USPS 16→32 双线性 resize、1→3 通道复制对齐；
     池 = USPS 官方 train 7,291，评估 = USPS 官方 test 2,007。
  b) CIFAR-10(源, 50,000) → CIFAR-10-C(下游, severity 3)：
     4 种 corruption（gaussian_noise / motion_blur / fog / brightness）× 各 10,000 张，
     每种 → 池/评估 = 5:5 分层（固定 seed 123）。
     官方 tar（zenodo 2535967）可得则用官方数据；否则用官方配方
     （Hendrycks & Dietterich 2019 的 imagecorruptions 实现）作用于真实
     CIFAR-10 test 图像生成，并在论文中如实注明。

骨干：compact ViT（patch 8, dim 128, depth 6, heads 4, MLP ratio 4, cls token，
      学习式位置编码，约 1.22M 参数），CPU 训练。

方法（8）：
  ZeroShot（冻结直接推理）/ LinearProbe（重训头，预训练头热启动）/ BitFit（仅偏置）/
  SSF（patch embedding 后与各 block 输出后逐维 scale-shift）/ Adapter（各 block 后
  串行残差瓶颈 r=dim/8，上投影零初始化）/ LoRA（qkv/proj/fc1/fc2/head 全部权重矩阵
  rank-4 旁路，B 零初始化）/ LFW(ours)（w∈R^3072 非负逐元素加权标准化输入，
  L1/L2 + 单位权重保底，与表格版同构）/ FFT（全参数微调上界）。

协议（与表格版 run_sci.py 对齐）：
  - 5 种子(0-4)；源域 90/10 分层训练/验证预训练（按种子）；
  - 下游池 frac{5,10,20,50,100}% 分层抽样（seed），80/20 训练/验证（分层, seed+7）；
  - 固定评估集；全部超参在 frac=0.5、seed=0 的池数据上网格搜索确定；
  - macro-F1 主指标；Adam + 梯度裁剪(1.0) + 早停；
  - 下游输入统一用源域统计量标准化（冻结骨干迁移的分布对齐要求）；
  - LFW 的 safeguard：以 w=1（等价零样本）为第 0 轮候选，学习权重仅在验证集
    更优时采纳；PEFT 基线同样以初始参数为第 0 轮候选（与表格版一致）。

输出：results/sci_vision_main[_quick].csv（与表格版同 schema）
      results/sci_vision_analysis[_quick].json（论文 4.8 节 / Table 8 数据源）
用法：
  python run_sci_vision.py --quick        # 冒烟（MNIST 对、1 种子、小预算全流程）
  python run_sci_vision.py                # 完整（2 对 5 下游数据集 × 5 种子）
  python run_sci_vision.py --analysis-only
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from data_utils import stratified_split, sample_fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VDATA = os.path.join(ROOT, 'data', 'vision')
PRE = os.path.join(ROOT, 'data', 'vit_pre')
RESULTS = os.path.join(ROOT, 'results')
C10C_TAR = os.path.join(ROOT, 'data', 'CIFAR-10-C.tar')
C10C_OFF = os.path.join(VDATA, 'CIFAR-10-C')
C10C_GEN = os.path.join(VDATA, 'CIFAR-10-C-gen')
PROV_PATH = os.path.join(RESULTS, 'sci_vision_provenance.json')
HP_PATH = os.path.join(RESULTS, 'sci_vision_hyperparams.json')
os.makedirs(VDATA, exist_ok=True)
os.makedirs(PRE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------- 协议常量
SEEDS = [0, 1, 2, 3, 4]
FRACTIONS = [0.05, 0.1, 0.2, 0.5, 1.0]
METHODS = ['ZeroShot', 'LinearProbe', 'BitFit', 'SSF', 'Adapter', 'LoRA',
           'LFW(ours)', 'FFT']
TRAIN_METHODS = METHODS[1:]
CORRUPTIONS = ['gaussian_noise', 'motion_blur', 'fog', 'brightness']
SEVERITY = 3
LR_GRID = (1e-3, 3e-3, 1e-2)
FFT_LR_GRID = (1e-3, 1e-4)
LFW_LR = 0.05                       # 与表格版 train_weights 一致
LFW_L1 = (1e-4, 1e-3, 1e-2)
LFW_L2 = (0.0, 1e-4)
PRE_EPOCHS, PRE_PAT, PRE_BATCH = 10, 3, 256
ADAPT_EPOCHS, ADAPT_PAT, ADAPT_BATCH = 25, 5, 128
SEL_EPOCHS, SEL_PAT = 10, 3         # 超参搜索预算（LFW 15/4）
INFER_N, INFER_BS, INFER_REPS = 2048, 256, 3
HEADER = ['dataset', 'seed', 'frac', 'method', 'acc', 'precision', 'recall',
          'f1', 'n_tuned', 'train_time_s', 'infer_ms']

# ViT 配置
IMG, PATCH, DIM, DEPTH, HEADS, N_CLS, MLP_RATIO = 32, 8, 128, 6, 4, 10, 4
LORA_R = 4

QUICK = False   # 由 main() 设置


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def fstr(frac):
    return f'{frac:g}'


# ---------------------------------------------------------------- 模型
def _lora(din, dout, r=LORA_R):
    return nn.ParameterList([nn.Parameter(torch.randn(din, r) / r),
                             nn.Parameter(torch.zeros(r, dout))])


def _lin_lora(x, layer, lora):
    if lora is None:
        return layer(x)
    return F.linear(x, layer.weight, layer.bias) + (x @ lora[0]) @ lora[1]


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.fc2 = nn.Linear(int(mlp_ratio * dim), dim)
        # PEFT 附件（build_model 中按方法挂载）
        self.lora_qkv = None
        self.lora_proj = None
        self.lora_fc1 = None
        self.lora_fc2 = None
        self.ssf = None       # ParameterList [gamma, beta]
        self.adapter = None   # ParameterList [Wd, bd, Wu, bu]

    def forward(self, h):
        B, N, _ = h.shape
        H = self.heads
        x = self.ln1(h)
        qkv = _lin_lora(x, self.qkv, self.lora_qkv)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, H, -1).transpose(1, 2)
        k = k.view(B, N, H, -1).transpose(1, 2)
        v = v.view(B, N, H, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        att = att.softmax(dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, N, -1)
        o = _lin_lora(o, self.proj, self.lora_proj)
        h = h + o
        x = self.ln2(h)
        m = _lin_lora(x, self.fc1, self.lora_fc1)
        m = F.gelu(m)
        m = _lin_lora(m, self.fc2, self.lora_fc2)
        h = h + m
        if self.adapter is not None:
            Wd, bd, Wu, bu = self.adapter
            h = h + F.relu(h @ Wd + bd) @ Wu + bu
        if self.ssf is not None:
            h = h * self.ssf[0] + self.ssf[1]
        return h


class SmallViT(nn.Module):
    """compact ViT：patch8 / dim128 / depth6 / heads4 / mlp4（约 1.22M 参数）。"""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, DIM, PATCH, PATCH)
        n_patches = (IMG // PATCH) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_patches, DIM))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        self.blocks = nn.ModuleList(
            [Block(DIM, HEADS, MLP_RATIO) for _ in range(DEPTH)])
        for blk in self.blocks:
            for lin in (blk.qkv, blk.proj, blk.fc1, blk.fc2):
                nn.init.trunc_normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)
        self.ln_f = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, N_CLS)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        # PEFT 附件
        self.ssf_embed = None
        self.lora_head = None
        self.w_lfw = None

    def forward(self, x, w=None):
        if w is not None:                      # LFW：逐元素输入加权
            x = (x.flatten(1) * w).view(x.shape)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        if self.ssf_embed is not None:
            x = x * self.ssf_embed[0] + self.ssf_embed[1]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x[:, 0])
        return _lin_lora(x, self.head, self.lora_head)


def build_model(state, method):
    """从预训练权重构建指定方法的模型（PEFT 附件在 load 之后挂载）。"""
    model = SmallViT()
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(0)          # PEFT 参数固定 rng(0) 初始化（同表格版 PEFTModel）
    if method == 'FFT':
        for p in model.parameters():
            p.requires_grad_(True)
    elif method == 'LinearProbe':
        model.head.weight.requires_grad_(True)
        model.head.bias.requires_grad_(True)
    elif method == 'BitFit':
        for n, p in model.named_parameters():
            if n.endswith('.bias'):
                p.requires_grad_(True)
    elif method == 'SSF':
        model.ssf_embed = nn.ParameterList(
            [nn.Parameter(torch.ones(DIM)), nn.Parameter(torch.zeros(DIM))])
        for blk in model.blocks:
            blk.ssf = nn.ParameterList(
                [nn.Parameter(torch.ones(DIM)), nn.Parameter(torch.zeros(DIM))])
    elif method == 'Adapter':
        r = DIM // 8
        for blk in self_blocks(model):
            blk.adapter = nn.ParameterList([
                nn.Parameter(torch.randn(DIM, r) * 0.01 / math.sqrt(DIM)),
                nn.Parameter(torch.zeros(r)),
                nn.Parameter(torch.zeros(r, DIM)),
                nn.Parameter(torch.zeros(DIM))])
    elif method == 'LoRA':
        for blk in self_blocks(model):
            blk.lora_qkv = _lora(DIM, 3 * DIM)
            blk.lora_proj = _lora(DIM, DIM)
            blk.lora_fc1 = _lora(DIM, MLP_RATIO * DIM)
            blk.lora_fc2 = _lora(MLP_RATIO * DIM, DIM)
        model.lora_head = _lora(DIM, N_CLS)
    elif method == 'LFW(ours)':
        model.w_lfw = nn.Parameter(torch.ones(3 * IMG * IMG))
    elif method == 'ZeroShot':
        pass
    else:
        raise ValueError(method)
    return model


def self_blocks(model):
    return model.blocks


def n_tuned(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- 数据
def _torch_u8(np_arr):
    return torch.from_numpy(np.ascontiguousarray(np_arr))


def _prep_gray(data_u8):
    """灰度图 (N,H,W) uint8 → (N,3,32,32) uint8：双线性 resize + 通道复制。"""
    x = torch.as_tensor(np.asarray(data_u8), dtype=torch.uint8).unsqueeze(1)
    x = F.interpolate(x.float(), size=(IMG, IMG), mode='bilinear',
                      align_corners=False)
    return x.round_().clamp_(0, 255).to(torch.uint8).repeat(1, 3, 1, 1)


def load_mnist():
    """MNIST train 全量作为源域（28→32 resize + 1→3 通道复制，d=3072）。"""
    from torchvision.datasets import MNIST
    d = MNIST(root=VDATA, train=True, download=True)
    return _prep_gray(d.data), np.asarray(d.targets, dtype=np.int64)


def load_usps_pair():
    """USPS 官方 train(7,291)=池 / test(2,007)=评估（16→32 resize + 通道复制）。"""
    from torchvision.datasets import USPS
    tr = USPS(root=VDATA, train=True, download=True)
    te = USPS(root=VDATA, train=False, download=True)
    return (_prep_gray(tr.data), np.asarray(tr.targets, dtype=np.int64),
            _prep_gray(te.data), np.asarray(te.targets, dtype=np.int64))


def load_cifar10(train=True):
    from torchvision.datasets import CIFAR10
    d = CIFAR10(root=VDATA, train=train, download=True)
    X = _torch_u8(d.data).permute(0, 3, 1, 2).contiguous()   # HWC→CHW
    y = np.asarray(d.targets, dtype=np.int64)
    return X, y


def channel_stats(X_u8):
    """逐通道 mean/std（[0,1] 尺度，分块累计避免大 float 拷贝）。"""
    s = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    n = 0
    for i in range(0, len(X_u8), 8192):
        b = X_u8[i:i + 8192].numpy().astype(np.float64) / 255.0
        s += b.sum(axis=(0, 2, 3))
        s2 += (b ** 2).sum(axis=(0, 2, 3))
        n += b.shape[0] * IMG * IMG
    mean, var = s / n, s2 / n - (s / n) ** 2
    return {'mean': torch.tensor(mean, dtype=torch.float32).view(3, 1, 1),
            'std': torch.tensor(np.sqrt(var), dtype=torch.float32).view(3, 1, 1)}


def _find_c10c(name):
    for root, _, files in os.walk(C10C_OFF):
        if name in files:
            return os.path.join(root, name)
    return None


def _extract_c10c(wanted):
    os.makedirs(C10C_OFF, exist_ok=True)
    with tarfile.open(C10C_TAR) as tf:
        for m in tf.getmembers():
            if os.path.basename(m.name) in wanted:
                tf.extract(m, C10C_OFF)
                log(f'  tar 解压 {m.name}')


def _ensure_imagecorruptions():
    try:
        from imagecorruptions import corrupt
        return corrupt
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'imagecorruptions'], check=True)
        from imagecorruptions import corrupt
        return corrupt


def get_c10c(corruption):
    """severity 3 的 10,000 张 corrupted CIFAR-10 test 图（官方 tar 优先，
    否则官方配方生成）。返回 (X_u8, y, provenance)。"""
    os.makedirs(C10C_OFF, exist_ok=True)
    fname = f'{corruption}.npy'
    f, lf = _find_c10c(fname), _find_c10c('labels.npy')
    if not (f and lf) and os.path.exists(C10C_TAR) \
            and os.path.getsize(C10C_TAR) > 2_500_000_000:
        try:
            _extract_c10c({fname, 'labels.npy'})
            f, lf = _find_c10c(fname), _find_c10c('labels.npy')
        except Exception as e:
            log(f'  官方 tar 解压失败: {e}')
    if f and lf:
        arr = np.load(f, mmap_mode='r')
        X = np.array(arr[(SEVERITY - 1) * 10000: SEVERITY * 10000])
        yfull = np.load(lf)
        sl = (np.s_[:] if len(yfull) == 10000
              else np.s_[(SEVERITY - 1) * 10000: SEVERITY * 10000])
        y = np.asarray(yfull[sl], dtype=np.int64)
        prov = 'official_tar'
    else:
        # 官方配方（imagecorruptions = Hendrycks 官方实现）作用于真实 test 图
        corrupt = _ensure_imagecorruptions()
        os.makedirs(C10C_GEN, exist_ok=True)
        gf = os.path.join(C10C_GEN, fname)
        Xte, yte = load_cifar10(train=False)
        if not os.path.exists(gf):
            log(f'  生成 corruption={corruption} sev={SEVERITY}（官方配方）')
            imgs = Xte.numpy().transpose(0, 2, 3, 1)
            out = np.stack([corrupt(im, severity=SEVERITY,
                                    corruption_name=corruption)
                            for im in imgs], axis=0)
            np.save(gf, out)
        X = np.load(gf)
        y = yte
        prov = 'generated_official_recipe'
    X_t = _torch_u8(X).permute(0, 3, 1, 2).contiguous() if X.shape[1] != 3 \
        else _torch_u8(X)
    return X_t, y, prov


def load_downstream(name):
    """返回 (Xpool, ypool, Xeval, yeval, provenance)。"""
    if name == 'USPS':
        Xp, yp, Xe, ye = load_usps_pair()
        return Xp, yp, Xe, ye, 'official'
    if name.startswith('C10C_'):
        X, y, prov = get_c10c(name[5:])
        i_pool, i_eval = stratified_split(y, 0.5, 123)      # 固定 池/评估 5:5
        return X[i_pool], y[i_pool], X[i_eval], y[i_eval], prov
    raise ValueError(name)


def preload_datasets():
    """预下载全部 torchvision 数据集（供后台并行下载用）。"""
    load_mnist()
    log('MNIST 就绪')
    load_usps_pair()
    log('USPS 就绪')
    load_cifar10(True)
    load_cifar10(False)
    log('CIFAR-10 就绪')


# ---------------------------------------------------------------- 训练工具
def norm_batch(x_u8, st):
    x = x_u8.float().div_(255.0)
    return (x - st['mean']) / st['std']


def augment_u8(x, flip):
    """随机平移裁剪（pad 4）+ 可选水平翻转（uint8 上、训练期源域增广）。"""
    B = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    ii = torch.randint(0, 9, (B,))
    jj = torch.randint(0, 9, (B,))
    out = torch.empty_like(x)
    for b in range(B):
        out[b] = xp[b, :, ii[b]:ii[b] + IMG, jj[b]:jj[b] + IMG]
    if flip:
        m = torch.rand(B) < 0.5
        out[m] = out[m].flip(-1)
    return out


@torch.no_grad()
def evaluate_v(model, X_u8, y, st, w=None, bs=512):
    model.eval()
    preds = []
    for i in range(0, len(X_u8), bs):
        logits = model(norm_batch(X_u8[i:i + bs], st), w)
        preds.append(logits.argmax(1).numpy())
    p = np.concatenate(preds)
    return {'acc': float(accuracy_score(y, p)),
            'precision_macro': float(precision_score(y, p, average='macro',
                                                     zero_division=0)),
            'recall_macro': float(recall_score(y, p, average='macro',
                                               zero_division=0)),
            'f1_macro': float(f1_score(y, p, average='macro'))}


@torch.no_grad()
def infer_ms_v(model, X_u8, st, w=None):
    """固定 2,048 张评估图的批量推理耗时（3 次取中位，每样本毫秒）。"""
    idx = np.random.default_rng(7).choice(len(X_u8),
                                          min(INFER_N, len(X_u8)), replace=False)
    X = X_u8[torch.from_numpy(idx)]
    ts = []
    for _ in range(INFER_REPS):
        t0 = time.perf_counter()
        for i in range(0, len(X), INFER_BS):
            model(norm_batch(X[i:i + INFER_BS], st), w)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / len(X) * 1000.0


def iterate_minibatches(n, batch, generator):
    perm = torch.randperm(n, generator=generator)
    for s in range(0, n, batch):
        yield perm[s:s + batch]


def pretrain_vit(src_name, X_u8, y, st, seed):
    """源域全参数预训练（磁盘缓存，支持断点续跑；缓存名含 epoch 预算，
    避免 --quick 的 2-epoch 缓存被全量运行误用）。"""
    path = os.path.join(PRE, f'{src_name}_s{seed}_e{PRE_EPOCHS}.pt')
    if os.path.exists(path):
        ck = torch.load(path, map_location='cpu', weights_only=True)
        model = SmallViT()
        model.load_state_dict(ck['state'])
        return model, ck['meta']
    i_tr, i_va = stratified_split(y, 0.9, seed)
    tr_t, va_t = torch.from_numpy(i_tr), torch.from_numpy(i_va)
    y_t = torch.from_numpy(y)
    torch.manual_seed(seed)
    model = SmallViT()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    best_f1, best_state, best_ep, bad = -1.0, None, 0, 0
    t0 = time.perf_counter()
    for ep in range(PRE_EPOCHS):
        model.train()
        for bidx in iterate_minibatches(len(tr_t), PRE_BATCH, g):
            xb = augment_u8(X_u8[tr_t[bidx]], flip=(src_name == 'CIFAR-10'))
            logits = model(norm_batch(xb, st))
            loss = F.cross_entropy(logits, y_t[tr_t[bidx]])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate_v(model, X_u8[va_t], y[i_va], st)['f1_macro']
        if v > best_f1:
            best_f1, best_ep, bad = v, ep, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PRE_PAT:
                break
        log(f'  {src_name} s{seed} ep{ep}: valF1={v:.4f}')
    model.load_state_dict(best_state)
    meta = dict(source=src_name, seed=seed, val_f1=float(best_f1),
                stop_epoch=int(best_ep), n_params=int(
                    sum(p.numel() for p in model.parameters())),
                pretrain_time=time.perf_counter() - t0)
    torch.save({'state': best_state, 'meta': meta}, path)
    log(f'预训练 {src_name} s{seed}: valF1={best_f1:.4f} ep={best_ep} '
        f'params={meta["n_params"]} {meta["pretrain_time"]:.0f}s')
    return model, meta


def adapt_model(model, method, Xtr, ytr_np, Xva, yva_np, st, lr, seed,
                l1=0.0, l2=0.0, epochs=None, patience=None):
    """统一适配流程（Adam + 裁剪 + 早停；LFW 含非负投影与单位权重保底；
    PEFT 基线以初始参数为第 0 轮候选；FFT 无初始候选——均与表格版一致）。
    返回 (best_val_f1, stop_epoch, train_time_s)。"""
    epochs = epochs or ADAPT_EPOCHS
    patience = patience or ADAPT_PAT
    g = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    w = model.w_lfw if method == 'LFW(ours)' else None
    ytr_t = torch.from_numpy(ytr_np)
    lfw = method == 'LFW(ours)'
    if lfw or method in TRAIN_METHODS[:5]:        # 初始参数候选（ZeroShot 除外）
        best_f1 = evaluate_v(model, Xva, yva_np, st, w)['f1_macro']
    else:
        best_f1 = -1.0
    if lfw:
        best_state = w.detach().clone()
    else:
        best_state = {n: p.detach().clone()
                      for n, p in model.named_parameters() if p.requires_grad}
    best_ep, bad = 0, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for bidx in iterate_minibatches(len(Xtr), ADAPT_BATCH, g):
            xb = norm_batch(Xtr[bidx], st)
            logits = model(xb, w)
            loss = F.cross_entropy(logits, ytr_t[bidx])
            if lfw and (l1 or l2):
                loss = loss + l1 * w.sum() + l2 * (w ** 2).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if lfw:
                with torch.no_grad():
                    w.clamp_(min=0.0)
        v = evaluate_v(model, Xva, yva_np, st, w)['f1_macro']
        if v > best_f1:
            best_f1, best_ep, bad = v, ep, 0
            if lfw:
                best_state = w.detach().clone()
            else:
                best_state = {n: p.detach().clone() for n, p in
                              model.named_parameters() if p.requires_grad}
        else:
            bad += 1
            if bad >= patience:
                break
    if lfw:
        with torch.no_grad():
            w.copy_(best_state)
    else:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_state:
                    p.copy_(best_state[n])
    return best_f1, best_ep, time.perf_counter() - t0


# ---------------------------------------------------------------- CSV 工具
def done_keys(path, keycols):
    keys = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path, encoding='utf-8-sig')):
            keys.add(tuple(str(r[c]) for c in keycols))
    return keys


class Appender:
    def __init__(self, path, header):
        self.path = path
        new = not os.path.exists(path)
        self.f = open(path, 'a', newline='', encoding='utf-8-sig')
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(header)
            self.f.flush()

    def row(self, *vals):
        self.w.writerow([v for v in vals])
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------- 超参搜索
def select_hyperparams(ds, state, Xpool, ypool, st):
    """每下游数据集一次：frac=0.5、seed=0 池数据 80/20，网格搜索全部超参。"""
    hp_all = {}
    if os.path.exists(HP_PATH):
        hp_all = json.load(open(HP_PATH, encoding='utf-8'))
        if ds in hp_all:
            return hp_all[ds]
    idx = sample_fraction(Xpool, ypool, 0.5, seed=0)
    Xa, ya = Xpool[torch.from_numpy(idx)], ypool[idx]
    i_tr, i_va = stratified_split(ya, 0.8, seed=0)
    Xtr, ytr = Xa[torch.from_numpy(i_tr)], ya[i_tr]
    Xva, yva = Xa[torch.from_numpy(i_va)], ya[i_va]
    h = {}
    for meth in ('LinearProbe', 'BitFit', 'SSF', 'Adapter', 'LoRA'):
        best = (-1.0, LR_GRID[0])
        for lr in LR_GRID:
            model = build_model(state, meth)
            v, _, _ = adapt_model(model, meth, Xtr, ytr, Xva, yva, st, lr, 0,
                                  epochs=SEL_EPOCHS, patience=SEL_PAT)
            if v > best[0]:
                best = (v, lr)
        h[f'lr_{meth}'] = best[1]
    best = (-1.0, FFT_LR_GRID[0])
    for lr in FFT_LR_GRID:
        model = build_model(state, 'FFT')
        v, _, _ = adapt_model(model, 'FFT', Xtr, ytr, Xva, yva, st, lr, 0,
                              epochs=SEL_EPOCHS, patience=SEL_PAT)
        if v > best[0]:
            best = (v, lr)
    h['lr_FFT'] = best[1]
    best = (-1.0, (LFW_L1[0], LFW_L2[0]))
    for l1 in LFW_L1:
        for l2 in LFW_L2:
            model = build_model(state, 'LFW(ours)')
            v, _, _ = adapt_model(model, 'LFW(ours)', Xtr, ytr, Xva, yva, st,
                                  LFW_LR, 0, l1=l1, l2=l2, epochs=15, patience=4)
            if v > best[0]:
                best = (v, (l1, l2))
    h['l1'], h['l2'] = best[1]
    hp_all[ds] = h
    json.dump(hp_all, open(HP_PATH, 'w', encoding='utf-8'), indent=1)
    log(f'{ds} 超参: {h}')
    return h


# ---------------------------------------------------------------- 主流程
PAIR_SPECS = [
    dict(pair='mnist_usps', source='MNIST', src_loader=load_mnist,
         downstreams=['USPS']),
    dict(pair='cifar_c10c', source='CIFAR-10',
         src_loader=lambda: load_cifar10(True),
         downstreams=[f'C10C_{c}' for c in CORRUPTIONS]),
]

PROV = {}


def save_prov():
    json.dump(PROV, open(PROV_PATH, 'w', encoding='utf-8'), indent=1)


def run_pair(spec, seeds, fractions, appender, done):
    src_name = spec['source']
    log(f'===== 迁移对 {spec["pair"]}: {src_name} → {spec["downstreams"]} =====')
    Xs, ys = spec['src_loader']()
    st = channel_stats(Xs)
    states = {}
    for seed in seeds:
        model, meta = pretrain_vit(src_name, Xs, ys, st, seed)
        states[seed] = {k: t.clone() for k, t in model.state_dict().items()}
    del Xs
    for ds in spec['downstreams']:
        log(f'----- 下游 {ds} -----')
        Xp, yp, Xe, ye, prov = load_downstream(ds)
        PROV[ds] = prov
        save_prov()
        hparams = (dict(lr_LinearProbe=LR_GRID[0], lr_BitFit=LR_GRID[0],
                        lr_SSF=LR_GRID[0], lr_Adapter=LR_GRID[0],
                        lr_LoRA=LR_GRID[0], lr_FFT=FFT_LR_GRID[0],
                        l1=LFW_L1[0], l2=LFW_L2[0]) if QUICK
                   else select_hyperparams(ds, states[0], Xp, yp, st))
        for seed in seeds:
            state = states[seed]
            zs_key = (ds, str(seed), '1', 'ZeroShot')
            if zs_key not in done:
                model = build_model(state, 'ZeroShot')
                m = evaluate_v(model, Xe, ye, st)
                im = infer_ms_v(model, Xe, st)
                appender.row(ds, seed, '1', 'ZeroShot', f"{m['acc']:.4f}",
                             f"{m['precision_macro']:.4f}",
                             f"{m['recall_macro']:.4f}", f"{m['f1_macro']:.4f}",
                             0, 0.0, f'{im:.4f}')
                done.add(zs_key)
            for frac in fractions:
                idx = sample_fraction(Xp, yp, frac, seed)
                Xa, ya = Xp[torch.from_numpy(idx)], yp[idx]
                i_tr, i_va = stratified_split(ya, 0.8, seed + 7)
                Xtr, ytr = Xa[torch.from_numpy(i_tr)], ya[i_tr]
                Xva, yva = Xa[torch.from_numpy(i_va)], ya[i_va]
                for meth in TRAIN_METHODS:
                    key = (ds, str(seed), fstr(frac), meth)
                    if key in done:
                        continue
                    model = build_model(state, meth)
                    if meth == 'LFW(ours)':
                        lr, l1, l2 = LFW_LR, hparams['l1'], hparams['l2']
                    else:
                        lr, l1, l2 = hparams[f'lr_{meth}'], 0.0, 0.0
                    v, ep, tt = adapt_model(model, meth, Xtr, ytr, Xva, yva,
                                            st, lr, seed, l1=l1, l2=l2)
                    w = model.w_lfw if meth == 'LFW(ours)' else None
                    m = evaluate_v(model, Xe, ye, st, w)
                    im = infer_ms_v(model, Xe, st, w)
                    appender.row(ds, seed, fstr(frac), meth, f"{m['acc']:.4f}",
                                 f"{m['precision_macro']:.4f}",
                                 f"{m['recall_macro']:.4f}",
                                 f"{m['f1_macro']:.4f}", n_tuned(model),
                                 f'{tt:.2f}', f'{im:.4f}')
                    done.add(key)
                log(f'{ds} s{seed} f{frac} 完成')
        del Xp, Xe


# ---------------------------------------------------------------- 分析
def analyze(csv_path, out_path):
    """聚合：主表(frac=1.0) mean±std、低资源曲线、Wilcoxon、协议与溯源。"""
    from collections import defaultdict
    rows = list(csv.DictReader(open(csv_path, encoding='utf-8-sig')))
    per = defaultdict(list)                      # (ds, frac, meth) -> [f1...]
    met = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r['dataset'], fstr(float(r['frac'])), r['method'])
        per[k].append(float(r['f1']))
        for mm in ('acc', 'precision', 'recall', 'f1'):
            met[k][mm].append(float(r[mm]))
        met[k]['n_tuned'].append(int(float(r['n_tuned'])))
        met[k]['train_time_s'].append(float(r['train_time_s']))
        met[k]['infer_ms'].append(float(r['infer_ms']))

    def ms(k, mm):
        v = met[k][mm]
        return [float(np.mean(v)), float(np.std(v))]

    datasets = sorted({r['dataset'] for r in rows})
    main = {}
    for ds in datasets:
        for meth in METHODS:
            k = (ds, '1', meth)
            if k not in met:
                continue
            main[f'{ds}|{meth}'] = {
                'f1': ms(k, 'f1'), 'acc': ms(k, 'acc'),
                'precision': ms(k, 'precision'), 'recall': ms(k, 'recall'),
                'n': len(met[k]['f1']), 'n_tuned': int(met[k]['n_tuned'][0]),
                'train_time_s': float(np.mean(met[k]['train_time_s'])),
                'infer_ms': float(np.mean(met[k]['infer_ms']))}
    lowres = {}
    for (ds, fr, meth), v in per.items():
        lowres[f'{ds}|{meth}|{fr}'] = [float(np.mean(v)), float(np.std(v)),
                                       len(v)]
    # Wilcoxon：LFW vs 各方法（全部 ds×frac×seed 配对单元）
    wil = {}
    try:
        from scipy.stats import wilcoxon
        cells = defaultdict(dict)
        for r in rows:
            cells[(r['dataset'], fstr(float(r['frac'])), r['seed'])][
                r['method']] = float(r['f1'])
        for meth in METHODS:
            if meth == 'LFW(ours)':
                continue
            xs, ys_ = [], []
            for c in cells.values():
                if 'LFW(ours)' in c and meth in c:
                    xs.append(c['LFW(ours)'])
                    ys_.append(c[meth])
            if len(xs) < 8:
                continue
            stat, p = wilcoxon(xs, ys_)
            wil[f'LFW_vs_{meth}'] = {
                'stat': float(stat), 'p': float(p), 'n': len(xs),
                'lfw_wins': int(sum(a > b for a, b in zip(xs, ys_))),
                'other_wins': int(sum(a < b for a, b in zip(xs, ys_)))}
    except Exception as e:
        wil['error'] = str(e)
    # 汇总量
    bb_params = sum(p.numel() for p in SmallViT().parameters())
    summary = {'backbone_params': int(bb_params),
               'lfw_params': 3 * IMG * IMG,
               'param_ratio_backbone_over_lfw': bb_params / (3 * IMG * IMG)}
    for tag, fr in (('f005', '0.05'), ('f100', '1')):
        gains, gains_b = [], []
        for ds in datasets:
            kf = f'{ds}|LFW(ours)|{fr}'
            if kf not in lowres:
                continue
            gains.append(lowres[kf][0])
            best_peft = -1
            for meth in ('LinearProbe', 'BitFit', 'SSF', 'Adapter', 'LoRA'):
                kk = f'{ds}|{meth}|{fr}'
                if kk in lowres:
                    best_peft = max(best_peft, lowres[kk][0])
            if best_peft >= 0:
                gains_b.append(lowres[kf][0] - best_peft)
        if gains:
            summary[f'lfw_mean_f1_{tag}'] = float(np.mean(gains))
        if gains_b:
            summary[f'lfw_gain_over_best_peft_{tag}'] = float(np.mean(gains_b))
    out = {
        'provenance': dict(PROV) if PROV else (
            json.load(open(PROV_PATH, encoding='utf-8'))
            if os.path.exists(PROV_PATH) else {}),
        'backbone': {'params': int(bb_params),
                     'config': f'patch{PATCH}/dim{DIM}/depth{DEPTH}/'
                               f'heads{HEADS}/mlp{MLP_RATIO}',
                     'lora_rank': LORA_R, 'severity': SEVERITY},
        'pairs': {**{d: 'MNIST->USPS' for d in datasets if d == 'USPS'},
                  **{d: f'CIFAR-10->CIFAR-10-C({d[5:]}, sev{SEVERITY})'
                     for d in datasets if d.startswith('C10C_')}},
        'protocol': {'seeds': SEEDS, 'fractions': FRACTIONS,
                     'pretrain': {'epochs': PRE_EPOCHS, 'patience': PRE_PAT,
                                  'batch': PRE_BATCH, 'lr': 1e-3,
                                  'split': '90/10 stratified per seed'},
                     'adapt': {'epochs': ADAPT_EPOCHS, 'patience': ADAPT_PAT,
                               'batch': ADAPT_BATCH, 'clip': 1.0,
                               'split': '80/20 stratified seed+7'},
                     'lr_grid': list(LR_GRID), 'fft_lr_grid': list(FFT_LR_GRID),
                     'lfw_lr': LFW_LR, 'lfw_l1': list(LFW_L1),
                     'lfw_l2': list(LFW_L2),
                     'infer_subset_n': INFER_N,
                     'usps_pool': 'official train 7291',
                     'c10c_pool_eval': '5000/5000 seed123',
                     'note': '下游输入用源域统计量标准化；灰度图 16/28->32 '
                             'bilinear + 1->3 通道复制'},
        'summary': summary, 'main': main, 'lowres': lowres, 'wilcoxon': wil,
        'n_rows': len(rows)}
    json.dump(out, open(out_path, 'w', encoding='utf-8'), indent=1)
    log(f'分析完成: {out_path}（{len(rows)} 行, {len(main)} 主表单元）')


# ---------------------------------------------------------------- 入口
def main():
    global QUICK, PRE_EPOCHS, PRE_PAT, ADAPT_EPOCHS, ADAPT_PAT, INFER_N
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='冒烟测试')
    ap.add_argument('--analysis-only', action='store_true')
    ap.add_argument('--seeds', nargs='+', type=int, default=None)
    ap.add_argument('--datasets', nargs='+', default=None)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    suffix = '_quick' if args.quick else ''
    csv_path = os.path.join(RESULTS, f'sci_vision_main{suffix}.csv')
    ana_path = os.path.join(RESULTS, f'sci_vision_analysis{suffix}.json')

    seeds, fractions = SEEDS, FRACTIONS
    if args.quick:
        QUICK = True
        seeds, fractions = [0], [0.1]
        PRE_EPOCHS, PRE_PAT = 2, 2
        ADAPT_EPOCHS, ADAPT_PAT = 2, 2
        INFER_N = 1024
    if args.seeds:
        seeds = args.seeds
    if args.analysis_only:
        analyze(csv_path, ana_path)
        return

    done = done_keys(csv_path, ['dataset', 'seed', 'frac', 'method'])
    app = Appender(csv_path, HEADER)
    specs = [s for s in PAIR_SPECS
             if not args.datasets
             or any(d in args.datasets for d in s['downstreams'])]
    try:
        for spec in specs:
            run_pair(spec, seeds, fractions, app, done)
    finally:
        app.close()
    analyze(csv_path, ana_path)
    log('视觉实验全部完成。')


if __name__ == '__main__':
    main()
