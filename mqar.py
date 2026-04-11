"""
MQAR: Multi-Query Associative Recall
=====================================
测试模型的关联记忆能力
直接证明cache的必要性：
  纯扫描（K=0）: 低准确率
  H-v2+Cache:   高准确率
  Transformer:  高准确率

任务设计：
  输入: [k1 v1] [k2 v2] ... [kn vn] [k_query] ?
  输出: v_query（对应k_query的值）

参考: Mamba论文, Based论文

用法：
  python mqar.py
"""
import sys, os, math, json, argparse
import glob as _glob
import torch
import torch.nn as nn
import torch.nn.functional as F

for _f in _glob.glob('/root/**/triton_scan_v2.py', recursive=True) + \
          _glob.glob('/home/**/triton_scan_v2.py', recursive=True):
    with open(_f) as ff:
        if 'def mipt_scan' in ff.read():
            sys.path.insert(0, os.path.dirname(_f)); break
try:
    from triton_scan_v2 import mipt_scan as _triton_scan
    _USE_TRITON = True; print("✓ Triton OK")
except:
    _USE_TRITON = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")


# ══════════════════════════════════════════════
#  MQAR数据生成
# ══════════════════════════════════════════════

def make_mqar_data(n_samples=10000, seq_len=64,
                   n_kv_pairs=8, vocab_size=512,
                   n_queries=4, seed=42):
    """
    生成MQAR数据

    格式：[k1 v1 k2 v2 ... kn vn SEP q1 SEP q2 ... SEP qm]
    目标：每个query位置预测对应的value

    vocab分区：
      0-127: key tokens
      128-255: value tokens
      256: SEP token
      257+: padding
    """
    torch.manual_seed(seed)

    key_vocab = 128   # key的词表大小
    val_vocab = 128   # value的词表大小
    SEP = 256

    all_x, all_y = [], []

    for _ in range(n_samples):
        # 随机生成n_kv_pairs对key-value
        keys   = torch.randint(0,   key_vocab, (n_kv_pairs,))
        values = torch.randint(128, 128+val_vocab, (n_kv_pairs,))

        # 确保keys不重复
        while len(keys.unique()) < n_kv_pairs:
            keys = torch.randint(0, key_vocab, (n_kv_pairs,))

        # 构建序列：[k1 v1 k2 v2 ...]
        kv_seq = []
        for k, v in zip(keys.tolist(), values.tolist()):
            kv_seq.extend([k, v])

        # 随机选n_queries个key做查询
        q_indices = torch.randperm(n_kv_pairs)[:n_queries]
        q_keys    = keys[q_indices]
        q_values  = values[q_indices]

        # 构建完整序列：[kv_pairs... SEP q1 SEP q2 ...]
        seq = kv_seq.copy()
        tgt = [-1] * len(seq)  # -1表示不计算loss

        for qk, qv in zip(q_keys.tolist(), q_values.tolist()):
            seq.append(SEP)
            tgt.append(-1)
            seq.append(qk)    # query key
            tgt.append(qv)    # target: 对应的value

        # 截断或padding到seq_len
        seq = seq[:seq_len]
        tgt = tgt[:seq_len]

        # padding
        while len(seq) < seq_len:
            seq.append(257)  # padding
            tgt.append(-1)

        all_x.append(seq)
        all_y.append(tgt)

    X = torch.tensor(all_x, dtype=torch.long)
    Y = torch.tensor(all_y, dtype=torch.long)
    return X, Y, vocab_size


# ══════════════════════════════════════════════
#  模型定义
# ══════════════════════════════════════════════

def _fast_scan(p, theta, inp_re, inp_im):
    B, T, D = p.shape
    if _USE_TRITON and p.is_cuda:
        out = _triton_scan(p, theta, torch.cat([inp_re, inp_im], -1))
        return out[:,:,:D], p
    h_re = torch.zeros(B, D, device=p.device)
    h_im = torch.zeros(B, D, device=p.device)
    hl = []
    for t in range(T):
        c=torch.cos(theta[:,t]); s=torch.sin(theta[:,t])
        r=c*h_re-s*h_im; i=s*h_re+c*h_im
        h_re=(1-p[:,t])*r+p[:,t]*inp_re[:,t]
        h_im=(1-p[:,t])*i+p[:,t]*inp_im[:,t]
        hl.append(h_re.unsqueeze(1))
    return torch.cat(hl,1), p


class GivensRotation(nn.Module):
    def __init__(self, d, n=4):
        super().__init__()
        self.r=n
        pairs=[(k*2%d,(k*2+1)%d) for k in range(n)]
        self.register_buffer('pairs',torch.tensor(pairs))
        self.W=nn.Linear(d,n)
    def forward(self,h,e):
        angles=self.W(e); parts=list(h.unbind(-1))
        for k in range(self.r):
            i,j=int(self.pairs[k,0]),int(self.pairs[k,1])
            c=torch.cos(angles[...,k]); s=torch.sin(angles[...,k])
            parts[i],parts[j]=c*parts[i]-s*parts[j],s*parts[i]+c*parts[j]
        return torch.stack(parts,-1)


def causal_topk_mask(score, K, T):
    B=score.shape[0]
    s_exp=score.unsqueeze(1).expand(B,T,T)
    causal=torch.tril(torch.ones(T,T,device=score.device),diagonal=-1).bool()
    sc=s_exp.masked_fill(~causal,float('-inf'))
    _,idx=sc.topk(min(K,T),dim=-1)
    thr=sc.gather(-1,idx)[:,:,-1:]
    return causal & (s_exp >= thr-1e-6)


class MIPTForMQAR(nn.Module):
    """纯MIPT，无cache（baseline）"""
    def __init__(self, vocab_size, d_model=64, n_layers=2, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size+10, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            l = nn.Module()
            l.norm1 = nn.LayerNorm(d_model)
            l.W_p   = nn.Linear(d_model, d_model)
            l.W_t   = nn.Linear(d_model, d_model)
            l.W_r   = nn.Linear(d_model, d_model)
            l.W_i   = nn.Linear(d_model, d_model)
            l.norm2 = nn.LayerNorm(d_model)
            l.ffn   = nn.Sequential(
                nn.Linear(d_model,d_model*4), nn.GELU(),
                nn.Linear(d_model*4,d_model))
            nn.init.constant_(l.W_p.bias, -2.0)
            self.layers.append(l)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size+10)

    def forward(self, x):
        B,T = x.shape
        h = self.embed(x)
        for l in self.layers:
            res=h; hn=l.norm1(h)
            p=torch.sigmoid(l.W_p(hn))
            t=l.W_t(hn); ir=l.W_r(hn); ii=l.W_i(hn)
            out,_=_fast_scan(p,t,ir,ii)
            h=res+out; h=h+l.ffn(l.norm2(h))
        return self.head(self.norm_out(h))


class Hv2ForMQAR(nn.Module):
    """H-v2 + Cache，核心对比模型"""
    def __init__(self, vocab_size, d_model=64, n_layers=2,
                 cache_k=8, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d_model
        self.cache_k = cache_k
        self.embed = nn.Embedding(vocab_size+10, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            l = nn.Module()
            l.norm1 = nn.LayerNorm(d_model)
            l.W_p   = nn.Linear(d_model, d_model)
            l.W_t   = nn.Linear(d_model, d_model)
            l.W_r   = nn.Linear(d_model, d_model)
            l.W_i   = nn.Linear(d_model, d_model)
            l.givens_t = GivensRotation(d_model, 4)
            l.givens_h = GivensRotation(d_model, 4)
            l.delta_gate = nn.Parameter(torch.zeros(1))
            l.norm2 = nn.LayerNorm(d_model)
            l.ffn   = nn.Sequential(
                nn.Linear(d_model,d_model*4), nn.GELU(),
                nn.Linear(d_model*4,d_model))
            # cache
            l.norm_x = nn.LayerNorm(d_model)
            l.norm_h = nn.LayerNorm(d_model)
            l.Wk = nn.Linear(d_model, d_model)
            l.Wv = nn.Linear(d_model, d_model)
            l.Wq = nn.Linear(d_model, d_model)
            l.Wg = nn.Linear(d_model, d_model)
            nn.init.constant_(l.Wg.bias, -3.0)
            nn.init.constant_(l.W_p.bias, -2.0)
            self.layers.append(l)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size+10)

    def forward(self, x):
        B,T = x.shape; D=self.d
        h = self.embed(x)
        for l in self.layers:
            res=h; xn=l.norm_x(h)
            p=torch.sigmoid(l.W_p(xn))
            p_scalar = p.mean(-1)  # (B,T)
            theta=l.givens_t(l.W_t(xn),xn)
            ir,ii=l.W_r(xn),l.W_i(xn)
            pd=p_scalar.unsqueeze(-1).expand(-1,-1,D)
            h1,_=_fast_scan(pd,theta,ir,ii)
            delta=l.givens_h(h1,xn)-h1
            g=torch.sigmoid(l.delta_gate)
            h2,_=_fast_scan(pd,theta,ir+g*delta,ii)
            # cache
            if self.cache_k > 0:
                # p_scalar已在上面计算
                mask = causal_topk_mask(p_scalar, self.cache_k, T)
                hn2 = l.norm_h(h2)
                K_mat=l.Wk(xn); V_mat=l.Wv(xn); Q_mat=l.Wq(hn2)
                sc=torch.bmm(Q_mat,K_mat.transpose(1,2))/math.sqrt(D)
                sc=sc.masked_fill(~mask,float('-inf'))
                has=mask.any(-1,keepdim=True)
                at=torch.where(has,F.softmax(sc,-1).nan_to_num(0.),
                               torch.zeros_like(sc))
                cout=torch.bmm(at,V_mat)
                gate=torch.sigmoid(l.Wg(h2))
                h2=h2+gate*cout
            h=res+h2; h=h+l.ffn(l.norm2(h))
        return self.head(self.norm_out(h))


class TransformerForMQAR(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size+10, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model*4,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size+10)

    def forward(self, x):
        B,T=x.shape
        # 因果mask
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device)
        h = self.embed(x)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


# ══════════════════════════════════════════════
#  训练和评估
# ══════════════════════════════════════════════

def train_mqar(model, X, Y, vocab_size, epochs=50, lr=3e-3,
               bs=128, name='', print_every=10):
    model = model.to(device)
    X = X.to(device); Y = Y.to(device)
    n = len(X)
    vn = max(1, n//10)
    Xtr,ytr = X[vn:],Y[vn:]
    Xte,yte = X[:vn],Y[:vn]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc = 0.

    for ep in range(1, epochs+1):
        model.train()
        idx = torch.randperm(len(Xtr), device=device)
        tl,nb = 0.,0
        for i in range(0, len(Xtr), bs):
            xb = Xtr[idx[i:i+bs]]
            yb = ytr[idx[i:i+bs]]
            logits = model(xb)   # (B,T,V)
            # 只在target!=-1的位置计算loss
            mask = (yb != -1)
            if mask.sum() == 0: continue
            loss = F.cross_entropy(
                logits[mask], yb[mask].clamp(0, vocab_size+9))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl+=loss.item(); nb+=1
        sch.step()

        # 评估
        model.eval()
        with torch.no_grad():
            logits = model(Xte)
            mask   = (yte != -1)
            if mask.sum() > 0:
                preds = logits[mask].argmax(-1)
                acc   = (preds == yte[mask].clamp(0, vocab_size+9)).float().mean().item()
                best_acc = max(best_acc, acc)

        if ep % print_every == 0:
            print(f"  [{name:20s}] ep{ep:02d}/{epochs} "
                  f"loss={tl/max(nb,1):.3f} "
                  f"acc={acc:.3f}(best={best_acc:.3f})")

    return best_acc


# ══════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len',    type=int, default=64)
    parser.add_argument('--n_kv',      type=int, default=8,
                        help='key-value对数量')
    parser.add_argument('--n_queries', type=int, default=4)
    parser.add_argument('--d_model',   type=int, default=64)
    parser.add_argument('--epochs',    type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=10000)
    args = parser.parse_args()

    print(f"{'='*55}")
    print(f"MQAR: Multi-Query Associative Recall")
    print(f"seq_len={args.seq_len}  n_kv={args.n_kv}  "
          f"n_queries={args.n_queries}")
    print(f"{'='*55}\n")

    # 不同难度的测试
    configs = [
        dict(seq_len=64,  n_kv=8,  label='Easy(64tok,8kv)'),
        dict(seq_len=128, n_kv=16, label='Medium(128tok,16kv)'),
        dict(seq_len=256, n_kv=32, label='Hard(256tok,32kv)'),
    ]

    all_results = {}
    VOCAB = 512

    for cfg in configs:
        print(f"\n{'─'*50}")
        print(f"难度: {cfg['label']}")
        print(f"{'─'*50}")

        X, Y, vocab = make_mqar_data(
            n_samples=args.n_samples,
            seq_len=cfg['seq_len'],
            n_kv_pairs=cfg['n_kv'],
            vocab_size=VOCAB,
            n_queries=args.n_queries)

        task_results = {}
        D = args.d_model
        models = [
            ('TF',           TransformerForMQAR(vocab, D, 2)),
            ('MIPT(K=0)',    MIPTForMQAR(vocab, D, 2)),
            ('H-v2(K=4)',    Hv2ForMQAR(vocab, D, 2, cache_k=4)),
            ('H-v2(K=8)',    Hv2ForMQAR(vocab, D, 2, cache_k=8)),
            ('H-v2(K=16)',   Hv2ForMQAR(vocab, D, 2, cache_k=16)),
        ]

        for name, m in models:
            n_p = sum(p.numel() for p in m.parameters())
            print(f"\n[{name}]  params={n_p:,}")
            acc = train_mqar(m, X, Y, vocab,
                             epochs=args.epochs,
                             name=name, print_every=20)
            task_results[name] = acc
            print(f"  → {name}: {acc:.3f}")

        all_results[cfg['label']] = task_results

    # 汇总
    print(f"\n{'='*55}")
    print("MQAR 最终结果")
    print(f"{'='*55}")
    print(f"{'模型':20s}", end='')
    for cfg in configs:
        print(f"  {cfg['label'][:12]:12s}", end='')
    print()
    print('─'*70)
    for name in ['TF','MIPT(K=0)','H-v2(K=4)','H-v2(K=8)','H-v2(K=16)']:
        print(f"{name:20s}", end='')
        for cfg in configs:
            acc = all_results[cfg['label']].get(name, 0)
            print(f"  {acc:.3f}       ", end='')
        print()

    os.makedirs('results', exist_ok=True)
    with open('results/mqar.json','w') as f:
        json.dump(all_results, f, indent=2)
    print("\n保存至 results/mqar.json")


if __name__ == '__main__':
    main()
