"""
H-v2 130M 完整训练套件
===========================
用法：
  # 1. 下载数据
  python train_130m.py --mode download

  # 2. 训练H-v2 130M
  python train_130m.py --mode train --model hv2 --run_name hv2_130m

  # 3. 训练TF 130M（对照）
  python train_130m.py --mode train --model tf --run_name tf_130m

  # 4. 大海捞针评估
  python train_130m.py --mode needle --ckpt checkpoints/hv2_130m_best.pt

  # 5. 生成对比图
  python train_130m.py --mode plot
"""
import sys, os, math, json, time, argparse
import glob as _glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Triton ──
for _f in _glob.glob('/root/**/triton_scan_v2.py', recursive=True) + \
          _glob.glob('/home/**/triton_scan_v2.py', recursive=True):
    with open(_f) as ff:
        if 'def mipt_scan' in ff.read():
            sys.path.insert(0, os.path.dirname(_f))
            print(f"  Triton找到: {_f}"); break
try:
    from triton_scan_v2 import mipt_scan as _triton_scan
    _USE_TRITON = True; print("✓ Triton kernel 已加载")
except Exception as e:
    _USE_TRITON = False; print(f"  Triton未加载: {e}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB\n")


# ══════════════════════════════════════════════
#  模型配置（130M参数）
# ══════════════════════════════════════════════

CONFIGS = {
    # ~125M参数（GPT-2 small对标）
    # d=512, L=12: embedding(100277*512≈51M) + 12层(≈6M/层) ≈ 123M
    '130m': dict(d_model=576, n_layers=12, n_heads=8,
                 ffn_mult=4, n_givens=4, cache_k=16),
    # ~350M参数
    '350m': dict(d_model=1024, n_layers=16, n_heads=16,
                 ffn_mult=4, n_givens=8, cache_k=16),
    # 小规模测试
    'debug': dict(d_model=128, n_layers=2, n_heads=4,
                  ffn_mult=4, n_givens=4, cache_k=8),
    # 无位置编码版本
    '130m_nopos': dict(d_model=576, n_layers=12, n_heads=8,
                       ffn_mult=4, n_givens=4, cache_k=16),
    # nopos + input_conv（Mamba风格，PPL预计降30-50%）
    '130m_conv': dict(d_model=576, n_layers=12, n_heads=8,
                      ffn_mult=4, n_givens=4, cache_k=16),
    # nopos + input_conv + aggressive p初始化
    '130m_full': dict(d_model=576, n_layers=12, n_heads=8,
                      ffn_mult=4, n_givens=4, cache_k=16),
}


# ══════════════════════════════════════════════
#  基础组件
# ══════════════════════════════════════════════

def _fast_scan_final_state(p, theta, inp_re, inp_im):
    """
    Parallel prefill优化：只返回最终隐状态(h_re, h_im)，不返回全序列
    用于prefill阶段快速提取最终隐状态，比逐步forward_step快10-100倍
    复杂度：O(T log T)并行 vs O(T)串行（但串行常数大）
    """
    B, T, D = p.shape
    if _USE_TRITON and p.is_cuda:
        out = _triton_scan(p, theta, torch.cat([inp_re, inp_im], -1))
        # 取最后一步的隐状态
        final = out[:, -1, :]  # (B, 2D)
        return final[:, :D], final[:, D:]
    # CPU fallback：串行但只需要最终状态
    h_re = torch.zeros(B, D, device=p.device, dtype=p.dtype)
    h_im = torch.zeros(B, D, device=p.device, dtype=p.dtype)
    for t in range(T):
        c = torch.cos(theta[:,t]); s = torch.sin(theta[:,t])
        r = c*h_re - s*h_im;      i = s*h_re + c*h_im
        h_re = (1-p[:,t])*r + p[:,t]*inp_re[:,t]
        h_im = (1-p[:,t])*i + p[:,t]*inp_im[:,t]
    return h_re, h_im


def _fast_scan(p, theta, inp_re, inp_im):
    """完整复数扫描，返回实部(B,T,D)"""
    B, T, D = p.shape
    if _USE_TRITON and p.is_cuda:
        out = _triton_scan(p, theta, torch.cat([inp_re, inp_im], -1))
        return out[:,:,:D], p
    h_re = torch.zeros(B, D, device=p.device, dtype=p.dtype)
    h_im = torch.zeros(B, D, device=p.device, dtype=p.dtype)
    hl = []
    for t in range(T):
        c = torch.cos(theta[:,t]); s = torch.sin(theta[:,t])
        r = c*h_re - s*h_im;      i = s*h_re + c*h_im
        h_re = (1-p[:,t])*r + p[:,t]*inp_re[:,t]
        h_im = (1-p[:,t])*i + p[:,t]*inp_im[:,t]
        hl.append(h_re.unsqueeze(1))
    return torch.cat(hl, 1), p


class GivensRotation(nn.Module):
    def __init__(self, d, n=4):
        super().__init__()
        self.r = n
        pairs = [(k*2%d, (k*2+1)%d) for k in range(n)]
        self.register_buffer('pairs', torch.tensor(pairs))
        self.W = nn.Linear(d, n)
    def forward(self, h, e):
        angles = self.W(e)
        parts = list(h.unbind(-1))
        for k in range(self.r):
            i, j = int(self.pairs[k,0]), int(self.pairs[k,1])
            c = torch.cos(angles[...,k]); s = torch.sin(angles[...,k])
            parts[i], parts[j] = c*parts[i]-s*parts[j], s*parts[i]+c*parts[j]
        return torch.stack(parts, -1)


def causal_topk_mask(score, K, T, local_window=64):
    """
    Local-Global混合Cache mask（Longformer思想）：
    - 局部窗口：永远保留最近local_window个token（保证PPL）
    - 全局极值：保留历史中p最大的K个token（保证长程记忆）
    两者取并集，确保短程预测和长程召回都有保障
    """
    B = score.shape[0]
    causal = torch.tril(torch.ones(T,T,device=score.device), diagonal=-1).bool()

    # 局部窗口mask：每个位置看最近local_window个token
    local_mask = torch.zeros(T, T, device=score.device, dtype=torch.bool)
    for i in range(T):
        start = max(0, i - local_window)
        local_mask[i, start:i] = True
    local_mask = local_mask.unsqueeze(0).expand(B,-1,-1)

    # 全局极值mask：历史中p最大的K个token
    if K > 0:
        s_exp = score.unsqueeze(1).expand(B,T,T)
        sc = s_exp.masked_fill(~causal, float('-inf'))
        _, idx = sc.topk(min(K,T), dim=-1)
        thr = sc.gather(-1,idx)[:,:,-1:]
        global_mask = causal & (s_exp >= thr - 1e-6)
    else:
        global_mask = torch.zeros_like(local_mask)

    # 取并集
    return local_mask | global_mask


# ══════════════════════════════════════════════
#  H-v2 Block
# ══════════════════════════════════════════════

class Hv2Block(nn.Module):
    def __init__(self, d, ffn_mult=4, n_givens=4, dropout=0.1,
                 use_input_conv=False, p_bias=-2.0):
        super().__init__()
        self.use_input_conv = use_input_conv
        self.norm1      = nn.LayerNorm(d)
        self.W_theta    = nn.Linear(d, d)
        self.W_r        = nn.Linear(d, d)
        self.W_i        = nn.Linear(d, d)
        self.causal_conv= nn.Conv1d(d, d, kernel_size=4,
                                     padding=3, groups=max(1,d//64))
        # W_beta：更强的非线性，让p对输入响应更敏感
        self.W_beta     = nn.Sequential(
            nn.Linear(d, d//4), nn.SiLU(),
            nn.Linear(d//4, d//8), nn.SiLU(),
            nn.Linear(d//8, 1))
        self.givens_t   = GivensRotation(d, n_givens)
        self.givens_h   = GivensRotation(d, n_givens)
        self.delta_gate = nn.Parameter(torch.zeros(1))
        self.norm2      = nn.LayerNorm(d)
        # SwiGLU：比普通FFN更强的非线性，所有顶级模型标配
        # 两个分支相乘后降维，补充线性扫描的表达能力
        self.ffn_gate = nn.Linear(d, d*ffn_mult, bias=False)
        self.ffn_up   = nn.Linear(d, d*ffn_mult, bias=False)
        self.ffn_down = nn.Linear(d*ffn_mult, d, bias=False)
        self.ffn_drop = nn.Dropout(dropout)
        # Mamba风格的因果input_conv：局部特征提取，解放波态专注长程
        # 关键：groups=d（深度可分离卷积）+ 只在左侧padding（因果性）
        if use_input_conv:
            self.input_conv = nn.Conv1d(d, d, kernel_size=4,
                                         padding=0, groups=d)  # padding=0，手动左侧padding
            self._conv_pad = 3  # kernel_size - 1
        # p偏置初始化：控制初始测量率
        # -2.0: p≈0.12（默认）
        # -5.0: p≈0.006（极度保守，适合长程任务）
        self._p_bias = p_bias

    def forward(self, x, p_wave):
        B, T, D = x.shape
        res = x; xn = self.norm1(x)
        theta = self.givens_t(self.W_theta(xn), xn)
        # 因果input_conv：只看过去，不看未来
        if self.use_input_conv:
            # 左侧padding：只在序列左边加0，确保因果性
            xn_t = xn.transpose(1, 2)  # (B, D, T)
            xn_padded = F.pad(xn_t, (self._conv_pad, 0))  # 只在左侧padding
            xn_conv = self.input_conv(xn_padded).transpose(1, 2)  # (B, T, D)
            xn_conv = F.silu(xn_conv)
            ir = self.W_r(xn_conv)
            ii = self.W_i(xn_conv)
        else:
            ir, ii = self.W_r(xn), self.W_i(xn)
        pd = p_wave.unsqueeze(-1).expand(-1,-1,D)
        h1, _ = _fast_scan(pd, theta, ir, ii)
        delta  = self.givens_h(h1, xn) - h1
        g      = torch.sigmoid(self.delta_gate)
        h2, _  = _fast_scan(pd, theta, ir+g*delta, ii)
        x = res + h2
        # SwiGLU前向
        xn2 = self.norm2(x)
        gate = F.silu(self.ffn_gate(xn2))
        up   = self.ffn_up(xn2)
        out  = self.ffn_drop(self.ffn_down(gate * up))
        return x + out


# ══════════════════════════════════════════════
#  H-v2 130M 语言模型
# ══════════════════════════════════════════════

class Hv2LM(nn.Module):
    """
    H-v2 语言模型
    无S0先验，有位置调制+RG跑动+Givens+两遍扫描+Cache
    O(N)训练内存，O(K·D)推理cache
    """
    def __init__(self, vocab_size, d_model=768, n_layers=12,
                 ffn_mult=4, n_givens=8, cache_k=16,
                 max_len=2048, dropout=0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_model
        self.cache_k = cache_k
        self.n_layers = n_layers

        self.embed    = nn.Embedding(vocab_size, d_model)
        self.use_pos  = (max_len > 0)  # max_len=0时不用pos_emb
        if self.use_pos:
            self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop    = nn.Dropout(dropout)

        # 位置调制（sin/cos）
        self.W_pos = nn.Linear(2, 1, bias=True)
        nn.init.zeros_(self.W_pos.weight)
        nn.init.zeros_(self.W_pos.bias)

        # mu：控制初始测量率，-2.0偏向波态
        self.mu = nn.Parameter(torch.tensor(-2.0))

        # MIPT层
        _use_conv  = kwargs.get('use_input_conv', False)
        _p_bias    = kwargs.get('p_bias', -2.0)
        self.layers = nn.ModuleList([
            Hv2Block(d_model, ffn_mult, n_givens, dropout,
                     use_input_conv=_use_conv, p_bias=_p_bias)
            for _ in range(n_layers)])

        # Cache组件
        if cache_k > 0:
            self.norm_x = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.norm_h = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.Wk = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(n_layers)])
            self.Wv = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(n_layers)])
            self.Wq = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(n_layers)])
            self.Wg = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(n_layers)])
            for wg in self.Wg:
                nn.init.constant_(wg.bias, -3.0)

        self.local_window = kwargs.get('local_window', 64)
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # 权重绑定

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens):
        B, T = tokens.shape
        if self.use_pos:
            pos = torch.arange(T, device=tokens.device).unsqueeze(0)
            # pos取模防止越界
            max_pos = self.pos_emb.weight.shape[0]
            pos = pos % max_pos
            x = self.drop(self.embed(tokens) + self.pos_emb(pos))
        else:
            x = self.drop(self.embed(tokens))

        # 位置调制
        t_idx    = torch.arange(T, device=tokens.device, dtype=x.dtype)
        pos_norm = t_idx / max(T-1, 1)
        pe       = torch.stack([
            torch.sin(math.pi * pos_norm),
            torch.cos(math.pi * pos_norm)], dim=-1)
        pos_corr = self.W_pos(pe).squeeze(-1).unsqueeze(0).expand(B,-1)
        ls       = torch.log(
            torch.arange(2, T+2, device=tokens.device, dtype=x.dtype))

        for i, layer in enumerate(self.layers):
            if self.cache_k > 0:
                xn = self.norm_x[i](x)
            else:
                xn = x

            # RG跑动（每层独立）
            ctx  = layer.causal_conv(xn.transpose(1,2))[:,:,:T].transpose(1,2)
            beta = layer.W_beta(torch.tanh(ctx)).squeeze(-1)
            seff = pos_corr + beta * ls.unsqueeze(0)
            p_wave = torch.sigmoid(seff + self.mu) * 0.5 + 1e-4

            h = layer(x, p_wave)

            if self.cache_k > 0:
                mask  = causal_topk_mask(seff, self.cache_k, T, self.local_window)
                hn    = self.norm_h[i](h)
                K_mat = self.Wk[i](xn)
                V_mat = self.Wv[i](xn)
                Q_mat = self.Wq[i](hn)
                sc    = torch.bmm(Q_mat, K_mat.transpose(1,2)) / math.sqrt(self.d)
                sc    = sc.masked_fill(~mask, float('-inf'))
                has   = mask.any(-1, keepdim=True)
                at    = torch.where(has,
                    F.softmax(sc, -1).nan_to_num(0.),
                    torch.zeros_like(sc))
                cout  = torch.bmm(at, V_mat)
                gate  = torch.sigmoid(self.Wg[i](h))
                x     = h + gate * cout
            else:
                x = h

        return self.lm_head(self.norm_out(x))

    @torch.no_grad()
    def parallel_prefill(self, tokens):
        """
        Parallel prefill：用parallel scan一次性处理整个prompt
        提取每层最终隐状态，同时建立cache
        
        比逐步forward_step快10-100倍（GPU上用Triton kernel）
        
        返回：
          h_re, h_im: (n_layers, B, D) 最终隐状态
          cache: 含local-global历史的cache
          last_logits: (B, vocab_size) 最后一个token的预测
        """
        B, T = tokens.shape
        dev = tokens.device

        if self.use_pos:
            pos = torch.arange(T, device=dev).unsqueeze(0) % self.pos_emb.weight.shape[0]
            x = self.drop(self.embed(tokens) + self.pos_emb(pos))
        else:
            x = self.drop(self.embed(tokens))

        t_idx   = torch.arange(T, device=dev, dtype=x.dtype)
        pos_norm= t_idx / max(T-1, 1)
        pe      = torch.stack([torch.sin(math.pi*pos_norm),
                               torch.cos(math.pi*pos_norm)], dim=-1)
        pos_corr= self.W_pos(pe).squeeze(-1).unsqueeze(0).expand(B,-1)
        ls      = torch.log(torch.arange(2, T+2, device=dev, dtype=x.dtype))

        h_re_all = []
        h_im_all = []
        cache = []

        for i, layer in enumerate(self.layers):
            if self.cache_k > 0:
                xn = self.norm_x[i](x)
            else:
                xn = x

            ctx  = layer.causal_conv(xn.transpose(1,2))[:,:,:T].transpose(1,2)
            beta = layer.W_beta(torch.tanh(ctx)).squeeze(-1)
            seff = pos_corr + beta * ls.unsqueeze(0)
            p_wave = torch.sigmoid(seff + self.mu) * 0.5 + 1e-4

            # 用parallel scan计算完整序列（训练模式）
            theta = layer.givens_t(layer.W_theta(x), x)
            ir = layer.W_r(x); ii = layer.W_i(x)
            pd = p_wave.unsqueeze(-1).expand(-1,-1,self.d)

            h1_seq, _ = _fast_scan(pd, theta, ir, ii)
            delta = layer.givens_h(h1_seq, x) - h1_seq
            g = torch.sigmoid(layer.delta_gate)
            h2_seq, _ = _fast_scan(pd, theta, ir + g*delta, ii)

            # 提取最终隐状态
            h_re_all.append(h2_seq[:, -1, :])  # (B, D)
            h_im_all.append(torch.zeros_like(h2_seq[:, -1, :]))  # 近似

            # 建立固定大小cache（和init_state格式一致）
            max_slots = self.local_window + self.cache_k
            dev2 = x.device
            c = {
                'K':      torch.zeros(max_slots, self.d, device=dev2),
                'V':      torch.zeros(max_slots, self.d, device=dev2),
                'seff':   torch.full((max_slots,), -1e9, device=dev2),
                'ptr':    0,
                'filled': 0,
                't':      T,
            }
            if self.cache_k > 0:
                seff_mean = seff.mean(0)  # (T,)
                # Local window
                local_start = max(0, T - self.local_window)
                local_idx   = list(range(local_start, T))
                # Global topK（排除local）
                remaining = [j for j in range(T) if j not in set(local_idx)]
                if remaining:
                    seff_rem = seff_mean[torch.tensor(remaining, device=dev2)]
                    topk_n   = min(self.cache_k, len(remaining))
                    _, topk_rel = seff_rem.topk(topk_n)
                    global_idx = [remaining[j] for j in topk_rel.tolist()]
                else:
                    global_idx = []
                keep = sorted(set(local_idx + global_idx))
                n_keep = min(len(keep), max_slots)
                for slot, kid in enumerate(keep[:n_keep]):
                    c['K'][slot]    = xn[0, kid, :].detach()
                    c['V'][slot]    = xn[0, kid, :].detach()
                    c['seff'][slot] = seff_mean[kid].item()
                c['filled'] = n_keep
                c['ptr']    = n_keep % max_slots
            cache.append(c)

            # Cache attention
            if self.cache_k > 0:
                mask  = causal_topk_mask(seff, self.cache_k, T, self.local_window)
                hn    = self.norm_h[i](h2_seq)
                K_mat = self.Wk[i](xn)
                V_mat = self.Wv[i](xn)
                Q_mat = self.Wq[i](hn)
                sc    = torch.bmm(Q_mat, K_mat.transpose(1,2)) / math.sqrt(self.d)
                sc    = sc.masked_fill(~mask, float('-inf'))
                has   = mask.any(-1, keepdim=True)
                at    = torch.where(has, F.softmax(sc,-1).nan_to_num(0.),
                                    torch.zeros_like(sc))
                cout  = torch.bmm(at, V_mat)
                gate  = torch.sigmoid(self.Wg[i](h2_seq))
                x     = h2_seq + gate * cout
            else:
                x = h2_seq

        last_logits = self.lm_head(self.norm_out(x[:, -1, :]))  # (B, vocab)

        h_re = torch.stack(h_re_all, dim=0)  # (n_layers, B, D)
        h_im = torch.stack(h_im_all, dim=0)
        return h_re, h_im, cache, last_logits

    def init_state(self, batch_size=1):
        """
        初始化推理状态（含固定大小cache）
        cache用预分配tensor，显存固定为O(W+K)
        """
        D   = self.d
        dev = next(self.parameters()).device
        h_re = torch.zeros(self.n_layers, batch_size, D, device=dev)
        h_im = torch.zeros(self.n_layers, batch_size, D, device=dev)

        # 固定大小cache：max_slots = local_window + cache_k
        max_slots = self.local_window + self.cache_k
        cache = []
        for _ in range(self.n_layers):
            cache.append({
                # 预分配固定大小tensor：(max_slots, D)
                'K':      torch.zeros(max_slots, D, device=dev),
                'V':      torch.zeros(max_slots, D, device=dev),
                'seff':   torch.full((max_slots,), -1e9, device=dev),
                'ptr':    0,      # 当前写入位置（循环覆盖）
                'filled': 0,      # 已填充的slot数
                't':      0,
            })
        return h_re, h_im, cache

    @torch.no_grad()
    def forward_step(self, token, h_re, h_im, t_step, cache=None):
        """
        O(1)流式推理：每次只处理一个token（含Local-Global Cache）

        参数：
          token:  (B,) 当前token
          h_re:   (n_layers, B, D) 实部隐状态
          h_im:   (n_layers, B, D) 虚部隐状态
          t_step: 当前时间步
          cache:  list of dict（来自init_state），None则跳过cache

        返回：
          logits: (B, vocab_size)
          h_re, h_im: 更新后的隐状态
          cache: 更新后的cache（若输入非None）
        """
        B = token.shape[0]
        D = self.d

        # Embedding（无pos_emb）
        e = self.embed(token.unsqueeze(1))  # (B, 1, D)
        e = e.squeeze(1)  # (B, D)

        # 位置调制（单步）
        max_t = self.pos_emb.weight.shape[0] if self.use_pos else 10000
        pos_norm = t_step / max(max_t - 1, 1)
        pe = torch.tensor([[
            math.sin(math.pi * pos_norm),
            math.cos(math.pi * pos_norm)
        ]], device=e.device, dtype=e.dtype)
        pos_corr = self.W_pos(pe).squeeze()  # scalar
        ls_t = math.log(t_step + 2)

        x = e
        new_h_re = h_re.clone()
        new_h_im = h_im.clone()

        for i, layer in enumerate(self.layers):
            # RG跑动（单步，用当前x估计beta）
            # causal_conv需要序列，这里用当前x近似
            x_3d = x.unsqueeze(1)  # (B, 1, D)
            ctx = layer.causal_conv(x_3d.transpose(1,2))[:,:,:1].transpose(1,2)
            beta = layer.W_beta(torch.tanh(ctx)).squeeze()  # scalar or (B,)

            seff = pos_corr + beta * ls_t
            p_wave = torch.sigmoid(seff + self.mu) * 0.5 + 1e-4
            if p_wave.dim() == 0:
                p_wave = p_wave.unsqueeze(0).expand(B)

            # 单步MIPT更新：h_t = (1-p)*rot(h_{t-1}) + p*input
            # 第一遍
            theta = layer.givens_t(layer.W_theta(x), x)  # (B, D)
            ir = layer.W_r(x); ii = layer.W_i(x)

            cos_t = torch.cos(theta); sin_t = torch.sin(theta)
            h_re_i = h_re[i]; h_im_i = h_im[i]
            rot_re = cos_t * h_re_i - sin_t * h_im_i
            rot_im = sin_t * h_re_i + cos_t * h_im_i
            p_d = p_wave.unsqueeze(-1).expand(-1, D)
            new_re = (1 - p_d) * rot_re + p_d * ir
            new_im = (1 - p_d) * rot_im + p_d * ii

            # Givens微扰
            h1 = new_re
            delta = layer.givens_h(h1, x) - h1
            g = torch.sigmoid(layer.delta_gate)
            ir2 = ir + g * delta

            # 第二遍
            rot_re2 = cos_t * h_re_i - sin_t * h_im_i
            rot_im2 = sin_t * h_re_i + cos_t * h_im_i
            new_re2 = (1 - p_d) * rot_re2 + p_d * ir2
            new_im2 = (1 - p_d) * rot_im2 + p_d * ii

            new_h_re[i] = new_re2
            new_h_im[i] = new_im2

            h_out = new_re2  # (B, D)

            # ── Local-Global Cache检索（O(W+K)固定显存） ──
            if self.cache_k > 0 and cache is not None:
                c = cache[i]
                xn_step = self.norm_x[i](x.unsqueeze(1)).squeeze(1)  # (B,D)
                hn_step = self.norm_h[i](h_out.unsqueeze(1)).squeeze(1)

                max_slots = self.local_window + self.cache_k
                seff_val  = seff.mean().item() if seff.dim()>0 else float(seff)

                # 写入当前token到固定slot（循环覆盖最老的global slot）
                # 策略：前local_window个slot用循环覆盖（保持最新local）
                #        后cache_k个slot用seff最低的替换（保持最重要global）
                filled = c['filled']
                ptr    = c['ptr']

                if filled < max_slots:
                    # 还有空slot，直接写入
                    c['K'][ptr]    = xn_step[0].detach()
                    c['V'][ptr]    = xn_step[0].detach()
                    c['seff'][ptr] = seff_val
                    c['ptr']       = (ptr + 1) % max_slots
                    c['filled']    = filled + 1
                else:
                    # cache满了：找seff最低的slot替换（LRU近似）
                    min_idx = int(c['seff'].argmin().item())
                    if seff_val > c['seff'][min_idx]:
                        c['K'][min_idx]    = xn_step[0].detach()
                        c['V'][min_idx]    = xn_step[0].detach()
                        c['seff'][min_idx] = seff_val
                c['t'] += 1

                # 检索：用所有有效slot
                n_valid = c['filled']
                if n_valid > 0:
                    K_hist = c['K'][:n_valid].unsqueeze(0).expand(B,-1,-1)  # (B,M,D)
                    V_hist = c['V'][:n_valid].unsqueeze(0).expand(B,-1,-1)

                    Q_s  = self.Wq[i](hn_step).unsqueeze(1)     # (B,1,D)
                    K_s  = self.Wk[i](K_hist)                    # (B,M,D)
                    V_s  = self.Wv[i](V_hist)
                    sc_s = torch.bmm(Q_s, K_s.transpose(1,2)) / math.sqrt(self.d)
                    at_s = F.softmax(sc_s, dim=-1).nan_to_num(0.)
                    cout = torch.bmm(at_s, V_s).squeeze(1)       # (B,D)

                    gate  = torch.sigmoid(self.Wg[i](h_out))
                    h_out = h_out + gate * cout

            # SwiGLU FFN（与forward保持一致）
            xn2  = layer.norm2(h_out)
            gate = F.silu(layer.ffn_gate(xn2))
            up   = layer.ffn_up(xn2)
            ffn_out = layer.ffn_drop(layer.ffn_down(gate * up))
            h_out = h_out + ffn_out
            x = h_out

        logits = self.lm_head(self.norm_out(x))  # (B, vocab_size)
        return logits, new_h_re, new_h_im, cache

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════
#  Transformer 130M（对照）
# ══════════════════════════════════════════════

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12,
                 n_heads=12, ffn_mult=4, max_len=2048,
                 dropout=0.1, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop    = nn.Dropout(dropout)
        dec = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model*ffn_mult,
            dropout, batch_first=True, norm_first=True,
            activation='gelu')
        self.decoder  = nn.TransformerDecoder(dec, n_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens):
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        x    = self.drop(self.embed(tokens) + self.pos_emb(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tokens.device)
        x = self.decoder(x, x, tgt_mask=mask, memory_mask=mask,
                         tgt_is_causal=True, memory_is_causal=True)
        return self.lm_head(self.norm_out(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════
#  数据集
# ══════════════════════════════════════════════

class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens  = tokens
        self.seq_len = seq_len
        self.n = (len(tokens) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start   : start + self.seq_len]
        y = self.tokens[start+1 : start + self.seq_len + 1]
        return x, y


def download_data(data_dir='./data'):
    """下载训练数据（OpenWebText或WikiText-103）"""
    os.makedirs(data_dir, exist_ok=True)

    # 优先用本地WikiText
    wikitext_path = '/root/nature/data/wikitext/wikitext-103-raw-v1'
    if os.path.exists(wikitext_path):
        print(f"找到本地WikiText: {wikitext_path}")
        return wikitext_path

    # 尝试HF镜像下载OpenWebText
    print("尝试下载OpenWebText...")
    try:
        import subprocess
        env = os.environ.copy()
        env['HF_ENDPOINT'] = 'https://hf-mirror.com'
        result = subprocess.run([
            'python', '-c',
            '''
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
ds = load_dataset("openwebtext", split="train", streaming=False)
print(f"OpenWebText加载成功，样本数: {len(ds)}")
'''
        ], capture_output=True, text=True, env=env, timeout=60)
        if result.returncode == 0:
            print(result.stdout)
            return 'openwebtext'
    except Exception as e:
        print(f"下载失败: {e}")

    print("使用本地WikiText-103")
    return wikitext_path


def load_tokens(data_path, max_tokens=10_000_000_000, cache_file='./data/tokens.pt'):
    """加载并缓存token"""
    import tiktoken
    enc = tiktoken.get_encoding('cl100k_base')
    V   = enc.n_vocab

    if os.path.exists(cache_file):
        print(f"加载缓存token: {cache_file}")
        data = torch.load(cache_file)
        return data['tokens'], data['vocab_size']

    print("编码文本...")
    if os.path.isdir(data_path):
        # parquet格式
        import pandas as pd
        files = sorted(f for f in os.listdir(data_path)
                       if f.endswith('.parquet'))
        texts = []
        for f in files:
            df = pd.read_parquet(os.path.join(data_path, f))
            texts.extend(df['text'].fillna('').tolist())
        text = ' '.join(texts)
    else:
        with open(data_path) as f:
            text = f.read()

    print(f"  文本长度: {len(text):,} 字符")
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens[:max_tokens], dtype=torch.long)
    print(f"  Token数: {len(tokens):,}  vocab={V}")

    os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
    torch.save({'tokens': tokens, 'vocab_size': V}, cache_file)
    print(f"  缓存保存至: {cache_file}")
    return tokens, V


# ══════════════════════════════════════════════
#  训练
# ══════════════════════════════════════════════

def train(model, tokens, vocab_size, args):
    model = model.to(device)
    n_params = model.count_params()
    print(f"参数量: {n_params:,} ({n_params/1e6:.1f}M)")

    # 切分训练/验证
    n_val = min(len(tokens)//20, 500_000)
    val_tokens = tokens[-n_val:].clamp(0, vocab_size-1)
    tr_tokens  = tokens[:-n_val].clamp(0, vocab_size-1)

    tr_ds  = TokenDataset(tr_tokens,  args.seq_len)
    val_ds = TokenDataset(val_tokens, args.seq_len)

    tr_loader  = DataLoader(tr_ds,  batch_size=args.bs,
                            shuffle=True,  num_workers=4,
                            pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs,
                            shuffle=False, num_workers=4,
                            pin_memory=True)

    print(f"训练: {len(tr_ds):,}样本  验证: {len(val_ds):,}样本")

    # 优化器（GPT-3风格）
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.1)

    # 学习率调度：warmup + cosine decay
    def lr_lambda(step):
        warmup = args.warmup_steps
        total  = args.max_steps
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/{args.run_name}.jsonl'

    best_ppl = float('inf')
    step = 0
    start_time = time.time()

    print(f"\n开始训练: {args.run_name}")
    print(f"目标steps: {args.max_steps:,}  warmup: {args.warmup_steps:,}")
    print(f"seq_len={args.seq_len}  bs={args.bs}  lr={args.lr}\n")

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.; n_batches = 0

        for xb, yb in tr_loader:
            if step >= args.max_steps:
                break

            xb = xb.to(device).clamp(0, vocab_size-1)
            yb = yb.to(device).clamp(0, vocab_size-1)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(xb)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B*T, V), yb.reshape(B*T))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()

            epoch_loss += loss.item()
            n_batches  += 1
            step       += 1

            # 打印进度
            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                tokens_seen = step * args.bs * args.seq_len
                tok_per_sec = tokens_seen / elapsed
                lr_now = sch.get_last_lr()[0]  # get_last_lr已经是实际lr，不需要再乘
                print(f"  step={step:6d} loss={loss.item():.4f} "
                      f"lr={lr_now:.2e} "
                      f"tok/s={tok_per_sec:.0f} "
                      f"elapsed={elapsed/3600:.1f}h")

            # 验证
            if step % args.eval_every == 0:
                val_ppl = evaluate(model, val_loader, vocab_size)
                tr_ppl  = math.exp(epoch_loss / max(n_batches, 1))
                print(f"\n  ── 验证 step={step} ──")
                print(f"  tr_ppl={tr_ppl:.1f}  val_ppl={val_ppl:.1f}")

                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    ckpt_path = f'checkpoints/{args.run_name}_best.pt'
                    torch.save({
                        'step': step,
                        'model_state': model.state_dict(),
                        'val_ppl': val_ppl,
                        'args': vars(args),
                    }, ckpt_path)
                    print(f"  ✓ 保存最优checkpoint: {ckpt_path}  PPL={val_ppl:.1f}")

                # 记录日志
                log = {'step': step, 'tr_ppl': tr_ppl,
                       'val_ppl': val_ppl, 'best_ppl': best_ppl,
                       'elapsed': time.time()-start_time}
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log) + '\n')
                print()

        if step >= args.max_steps:
            break

    print(f"\n训练完成！最优PPL: {best_ppl:.1f}")
    return best_ppl


def evaluate(model, loader, vocab_size, max_batches=200):
    model.eval()
    total_loss = 0.; n = 0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            if i >= max_batches: break
            xb = xb.to(device).clamp(0, vocab_size-1)
            yb = yb.to(device).clamp(0, vocab_size-1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(xb)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B*T,V), yb.reshape(B*T))
            total_loss += loss.item(); n += 1
    model.train()
    return math.exp(total_loss / max(n, 1))


# ══════════════════════════════════════════════
#  大海捞针评估
# ══════════════════════════════════════════════

def needle_in_haystack(model, vocab_size, args):
    """
    大海捞针评估：生成热力图
    针的位置 vs 序列长度，颜色=正确率
    """
    import tiktoken
    enc = tiktoken.get_encoding('cl100k_base')
    model.eval()

    seq_lengths  = [512, 1024, 2048, 4096, 8192, 16384]
    needle_depths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_trials = 50

    results = {}
    memory_usage = {}

    for seq_len in seq_lengths:
        results[seq_len] = {}
        torch.cuda.empty_cache()

        # 测显存
        torch.cuda.reset_peak_memory_stats()
        try:
            x_test = torch.randint(0, vocab_size, (1, seq_len),
                                   device=device).clamp(0, vocab_size-1)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = model(x_test)
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            memory_usage[seq_len] = mem_mb
            print(f"  seq={seq_len:6d}: 显存={mem_mb:.0f}MB")
        except torch.cuda.OutOfMemoryError:
            memory_usage[seq_len] = float('inf')
            print(f"  seq={seq_len:6d}: OOM")
            results[seq_len] = {d: 0.0 for d in needle_depths}
            continue

        for depth in needle_depths:
            correct = 0
            for _ in range(n_trials):
                # 生成随机背景序列
                seq = torch.randint(100, vocab_size,
                                    (seq_len,), device=device)
                # 在depth位置放入针（特殊token）
                needle_pos = int(seq_len * depth)
                needle_tok = torch.randint(10, 100, (1,)).item()
                seq[needle_pos] = needle_tok

                # 预测needle_pos之后的位置是否能召回针
                query_pos = min(needle_pos + 10, seq_len - 1)
                seq_input = seq[:query_pos].unsqueeze(0).clamp(0, vocab_size-1)

                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(seq_input)
                    pred = logits[0, -1].argmax().item()

                # 简化判断：模型是否"记住"了针的token
                # 真实评估需要设计具体任务，这里用top-10命中率
                top10 = logits[0, -1].topk(10).indices.tolist()
                if needle_tok in top10:
                    correct += 1

            acc = correct / n_trials
            results[seq_len][depth] = acc

        avg = sum(results[seq_len].values()) / len(needle_depths)
        print(f"    平均召回率: {avg:.3f}")

    # 保存结果
    os.makedirs('results', exist_ok=True)
    save_data = {
        'results': {str(k): v for k,v in results.items()},
        'memory_mb': {str(k): v for k,v in memory_usage.items()},
        'model': args.run_name,
    }
    with open(f'results/needle_{args.run_name}.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n结果保存至 results/needle_{args.run_name}.json")

    # 打印热力图ASCII版
    print("\n大海捞针热力图（行=序列长度，列=针的深度）")
    print(f"{'':8s}", end='')
    for d in needle_depths:
        print(f"  {d:.1f}", end='')
    print()
    for sl in seq_lengths:
        if sl not in results: continue
        print(f"{sl:8d}", end='')
        for d in needle_depths:
            acc = results[sl].get(d, 0)
            bar = '█' if acc > 0.8 else ('▓' if acc > 0.5 else '░')
            print(f"  {bar}{acc:.2f}"[:6], end='')
        print()

    return results, memory_usage


# ══════════════════════════════════════════════
#  生成对比图
# ══════════════════════════════════════════════

def plot_results():
    """生成论文用对比图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib未安装，跳过画图")
        return

    os.makedirs('figures', exist_ok=True)

    # 1. 显存对比图
    hv2_needle_f = 'results/needle_hv2_130m.json'
    tf_needle_f  = 'results/needle_tf_130m.json'

    if os.path.exists(hv2_needle_f) and os.path.exists(tf_needle_f):
        with open(hv2_needle_f) as f: hv2_data = json.load(f)
        with open(tf_needle_f)  as f: tf_data  = json.load(f)

        seq_lens = sorted(int(k) for k in hv2_data['memory_mb'].keys())
        hv2_mem  = [hv2_data['memory_mb'][str(s)] for s in seq_lens]
        tf_mem   = [tf_data['memory_mb'].get(str(s), float('inf'))
                    for s in seq_lens]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 显存曲线
        valid_tf = [(s, m) for s,m in zip(seq_lens, tf_mem)
                    if m != float('inf')]
        ax1.plot(seq_lens, hv2_mem, 'b-o', label='H-v2 (O(N))', linewidth=2)
        if valid_tf:
            ax1.plot([s for s,_ in valid_tf], [m for _,m in valid_tf],
                     'r-s', label='Transformer (O(N²))', linewidth=2)
        ax1.set_xlabel('Sequence Length'); ax1.set_ylabel('GPU Memory (MB)')
        ax1.set_title('Memory Scaling')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        # 大海捞针热力图（H-v2）
        depths   = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        heatmap  = [[hv2_data['results'].get(str(sl),{}).get(d,0)
                     for d in depths] for sl in seq_lens]
        import numpy as np
        im = ax2.imshow(np.array(heatmap), aspect='auto',
                        cmap='RdYlGn', vmin=0, vmax=1)
        ax2.set_xticks(range(len(depths)))
        ax2.set_xticklabels([f'{d:.1f}' for d in depths])
        ax2.set_yticks(range(len(seq_lens)))
        ax2.set_yticklabels([str(s) for s in seq_lens])
        ax2.set_xlabel('Needle Depth'); ax2.set_ylabel('Sequence Length')
        ax2.set_title('H-v2 Needle-in-a-Haystack')
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.savefig('figures/main_comparison.png', dpi=150,
                    bbox_inches='tight')
        print("图片保存至 figures/main_comparison.png")


# ══════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   type=str, default='train',
                        choices=['download','train','needle','plot','debug'])
    parser.add_argument('--model',  type=str, default='hv2',
                        choices=['hv2','tf'])
    parser.add_argument('--size',   type=str, default='130m',
                        choices=['debug','130m','130m_nopos','350m'])
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--ckpt',   type=str, default=None)
    parser.add_argument('--data',   type=str, default=None)

    # 训练超参
    parser.add_argument('--seq_len',      type=int,   default=1024)
    parser.add_argument('--bs',           type=int,   default=8)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--epochs',       type=int,   default=3)
    parser.add_argument('--max_steps',    type=int,   default=100_000)
    parser.add_argument('--warmup_steps', type=int,   default=2_000)
    parser.add_argument('--log_every',    type=int,   default=100)
    parser.add_argument('--eval_every',   type=int,   default=1_000)
    parser.add_argument('--cache_k',      type=int,   default=16)
    parser.add_argument('--local_window', type=int,   default=64,
                        help='局部窗口大小，越大PPL越低，显存越大')
    parser.add_argument('--no_pos',       action='store_true',
                        help='去掉pos_emb，支持任意长度外推')
    parser.add_argument('--input_conv',   action='store_true',
                        help='加input_conv（Mamba风格局部特征）')
    parser.add_argument('--p_bias',       type=float, default=-2.0,
                        help='p初始化偏置，-5.0更保守')
    parser.add_argument('--freeze_wave',  action='store_true',
                        help='冻结波态，只训练cache参数')
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = f'{args.model}_{args.size}'

    # ── 下载数据 ──
    if args.mode == 'download':
        data_path = download_data()
        print(f"\n数据路径: {data_path}")
        print("接下来运行:")
        print(f"  python train_130m.py --mode train --model hv2 --data {data_path}")
        return

    # ── Debug模式 ──
    if args.mode == 'debug':
        args.size = 'debug'
        args.max_steps = 100
        args.eval_every = 50
        args.log_every  = 10
        args.bs = 2
        args.seq_len = 128
        print("Debug模式：快速验证流程")

    # ── 加载数据 ──
    data_path = args.data
    if data_path is None:
        for c in ['/root/nature/data/wikitext/wikitext-103-raw-v1',
                  '/root/nature/data/wikitext']:
            if os.path.exists(c): data_path=c; break
    if data_path is None:
        print("错误：请指定 --data 路径")
        return

    print(f"\n加载数据: {data_path}")
    tokens, vocab_size = load_tokens(
        data_path,
        cache_file=f'./data/tokens_{args.size}.pt')
    print(f"总tokens: {len(tokens):,}  vocab: {vocab_size:,}")

    # ── 创建模型 ──
    size_key = args.size
    if args.no_pos and args.model == 'hv2':
        size_key = '130m_nopos'
        print("  使用无位置编码版本（支持任意长度外推）")
    cfg = CONFIGS[size_key].copy()
    cfg['cache_k'] = args.cache_k
    # no_pos时max_len=0，不创建pos_emb
    cfg['max_len'] = 0 if args.no_pos else (args.seq_len + 16)

    if args.model == 'hv2':
        # 把命令行参数传给模型
        extra = {}
        if hasattr(args, 'input_conv') and args.input_conv:
            extra['use_input_conv'] = True
            print("  启用input_conv（Mamba风格）")
        if hasattr(args, 'p_bias'):
            extra['p_bias'] = args.p_bias
            print(f"  p_bias={args.p_bias} (初始p≈{1/(1+2.718**(-args.p_bias)):.3f})")
        if hasattr(args, 'local_window'):
            extra['local_window'] = args.local_window
            print(f"  local_window={args.local_window}")
        model = Hv2LM(vocab_size, **cfg, **extra)
    else:
        model = TransformerLM(vocab_size, **cfg)

    print(f"\n模型: {args.model.upper()}-{args.size}")
    print(f"配置: {cfg}")
    print(f"参数量: {model.count_params()/1e6:.1f}M")

    # ── 加载checkpoint ──
    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
        print(f"加载checkpoint: {args.ckpt}  PPL={ckpt.get('val_ppl',0):.1f}")

    # ── 执行 ──
    # 冻结波态
    if getattr(args, 'freeze_wave', False):
        print("⚡ 冻结扫描核心，Cache+Embed+LMHead可训练")
        # 只冻结波态扫描的核心参数
        # 逻辑：波态已经收敛(PPL=460)，锁住它
        # Cache+Embed可以联合学习自然语言的检索模式（归纳头机制）
        FREEZE_KEYS = ['W_theta', 'W_r', 'W_i', 'W_beta',
                       'givens_t', 'givens_h', 'delta_gate',
                       'causal_conv']
        frozen, trainable = 0, 0
        for name, param in model.named_parameters():
            should_freeze = any(k in name for k in FREEZE_KEYS)
            if should_freeze:
                param.requires_grad = False
                frozen += param.numel()
            else:
                param.requires_grad = True
                trainable += param.numel()
        print(f"  冻结扫描核心: {frozen/1e6:.1f}M")
        print(f"  可训练(Cache+Embed+LMHead): {trainable/1e6:.1f}M")

    if args.mode in ['train', 'debug']:
        train(model, tokens, vocab_size, args)

    elif args.mode == 'needle':
        model = model.to(device)
        print(f"\n大海捞针评估: {args.run_name}")
        needle_in_haystack(model, vocab_size, args)

    elif args.mode == 'plot':
        plot_results()


if __name__ == '__main__':
    main()
