"""
全息参考光解码 v4：多头正交参考光
====================================
基于v3发现：全息信息90%在第0层。
本实验只在第0层加多头正交参考光。

多头机制：
  h个头各自独立投影：gate_i = sigmoid(W_ref_i(x))
  正交惩罚：逼迫不同头学不同方向（提取不同语义）
  融合：h_out = h * (1/n_heads) * sum(gate_i)

正交损失：
  L_orth = sum_{i!=j} ||W_ref_i @ W_ref_j.T||_F^2
  迫使每个头的投影矩阵互相正交，提取正交方向的信息
"""
import sys, os, math, argparse, json
import glob as _glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

for _f in _glob.glob('/root/**/triton_scan_v2.py', recursive=True) + \
          _glob.glob('/home/**/triton_scan_v2.py', recursive=True):
    with open(_f) as ff:
        if 'def mipt_scan' in ff.read():
            sys.path.insert(0, os.path.dirname(_f)); break

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiHeadRefBeam(nn.Module):
    """
    多头正交参考光解码器（只用于第0层）
    
    n_heads个头，每头一个W_ref_i，全部零初始化。
    初始时所有gate=sigmoid(0)=0.5，融合后=h*0.5...
    
    等等，这样初始PPL会变。
    正确做法：用phase模式，tanh零初始化=0，1+0=1，h不变。
    
    融合：h * mean(1 + tanh(W_ref_i(x)))
    正交损失：逼迫不同头方向正交
    """
    def __init__(self, d_model, n_heads=4, orth_lambda=0.01):
        super().__init__()
        self.n_heads = n_heads
        self.orth_lambda = orth_lambda
        # 每头一个投影，全部零初始化
        self.heads = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=True)
            for _ in range(n_heads)
        ])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, h, x_ref):
        """
        h: (B, T, D) 全息底片
        x_ref: (B, T, D) 参考光
        返回: (B, T, D) 调制后的h, 正交损失标量
        """
        # 每头生成调制因子
        modulations = []
        for head in self.heads:
            mod = 1.0 + torch.tanh(head(x_ref))  # 零初始化时=1
            modulations.append(mod)

        # 平均融合（初始时每头=1，平均=1，h不变）
        avg_mod = sum(modulations) / self.n_heads
        h_out = h * avg_mod

        # 正交损失：逼迫不同头的W矩阵正交
        orth_loss = torch.tensor(0.0, device=h.device)
        if self.orth_lambda > 0 and self.training:
            for i in range(self.n_heads):
                for j in range(i+1, self.n_heads):
                    Wi = self.heads[i].weight  # (D, D)
                    Wj = self.heads[j].weight  # (D, D)
                    # Wi @ Wj.T的off-diagonal应该接近0
                    gram = Wi @ Wj.T  # (D, D)
                    orth_loss = orth_loss + gram.pow(2).mean()
            orth_loss = orth_loss * self.orth_lambda

        return h_out, orth_loss


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens; self.seq_len = seq_len
        self.n = (len(tokens)-1)//seq_len
    def __len__(self): return self.n
    def __getitem__(self, idx):
        s = idx*self.seq_len
        return self.tokens[s:s+self.seq_len], self.tokens[s+1:s+self.seq_len+1]


def patch_multihead(model, n_heads=4, orth_lambda=0.01):
    """只在第0层加多头正交参考光"""
    D = model.d
    dev = next(model.parameters()).device

    # 完全冻结base
    for p in model.parameters():
        p.requires_grad = False

    # 只在第0层加多头参考光
    model.mh_ref = MultiHeadRefBeam(D, n_heads, orth_lambda).to(dev)

    # 新norm_out
    model.norm_out_holo = nn.LayerNorm(D).to(dev)
    with torch.no_grad():
        model.norm_out_holo.weight.copy_(model.norm_out.weight)
        model.norm_out_holo.bias.copy_(model.norm_out.bias)

    import types

    def new_forward(self, tokens):
        from train_130m import causal_topk_mask
        B, T = tokens.shape
        if self.use_pos:
            pos = torch.arange(T, device=tokens.device).unsqueeze(0) % self.pos_emb.weight.shape[0]
            x = self.drop(self.embed(tokens) + self.pos_emb(pos))
        else:
            x = self.drop(self.embed(tokens))

        x_ref_layer0 = x.detach()  # 第0层的参考光

        t_idx    = torch.arange(T, device=tokens.device, dtype=x.dtype)
        pos_norm = t_idx / max(T-1, 1)
        pe       = torch.stack([torch.sin(math.pi*pos_norm),
                                torch.cos(math.pi*pos_norm)], dim=-1)
        pos_corr = self.W_pos(pe).squeeze(-1).unsqueeze(0).expand(B,-1)
        ls       = torch.log(torch.arange(2, T+2, device=tokens.device, dtype=x.dtype))

        total_orth_loss = torch.tensor(0.0, device=tokens.device)

        for i, layer in enumerate(self.layers):
            xn = self.norm_x[i](x) if self.cache_k > 0 else x
            ctx  = layer.causal_conv(xn.transpose(1,2))[:,:,:T].transpose(1,2)
            beta = layer.W_beta(torch.tanh(ctx)).squeeze(-1)
            seff = pos_corr + beta * ls.unsqueeze(0)
            p_wave = torch.sigmoid(seff + self.mu) * 0.5 + 1e-4
            h = layer(x, p_wave)

            if self.cache_k > 0:
                mask  = causal_topk_mask(seff, self.cache_k, T, self.local_window)
                hn    = self.norm_h[i](h)
                K_mat = self.Wk[i](xn); V_mat = self.Wv[i](xn); Q_mat = self.Wq[i](hn)
                sc    = torch.bmm(Q_mat, K_mat.transpose(1,2)) / math.sqrt(self.d)
                sc    = sc.masked_fill(~mask, float('-inf'))
                has   = mask.any(-1, keepdim=True)
                at    = torch.where(has, F.softmax(sc,-1).nan_to_num(0.), torch.zeros_like(sc))
                cout  = torch.bmm(at, V_mat)
                gate  = torch.sigmoid(self.Wg[i](h))
                x_out = h + gate * cout
            else:
                x_out = h

            # 只在第0层加多头参考光
            if i == 0:
                x_out, orth_loss = self.mh_ref(x_out, x_ref_layer0)
                total_orth_loss = total_orth_loss + orth_loss

            x = x_out

        # 把正交损失存到模型上，训练循环里取出来加到总loss
        self._last_orth_loss = total_orth_loss
        return self.lm_head(self.norm_out_holo(x))

    model.forward = types.MethodType(new_forward, model)
    return model


def evaluate(model, val_tokens, vocab_size, seq_len=1024, max_batches=200, batch_size=8):
    val_ds = TokenDataset(val_tokens, seq_len)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    total_loss = 0.; n = 0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(val_loader):
            if i >= max_batches: break
            xb = xb.to(device).clamp(0, vocab_size-1)
            yb = yb.to(device).clamp(0, vocab_size-1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(xb)
                if hasattr(out,'logits'): out=out.logits
                B,T,V = out.shape
                loss = F.cross_entropy(out.reshape(B*T,V), yb.reshape(B*T))
            total_loss += loss.item(); n += 1
    model.train()
    return math.exp(total_loss / max(n, 1))


def run(args):
    from train_130m import Hv2LM, CONFIGS

    print(f"\n{'='*60}")
    print(f"全息参考光 v4：多头正交（第0层）")
    print(f"n_heads={args.n_heads}  orth_lambda={args.orth_lambda}")
    print(f"{'='*60}\n")

    d = torch.load('./data/tokens_130m.pt')
    tokens = d['tokens'].clamp(0, d['vocab_size']-1)
    vocab_size = d['vocab_size']
    n_val = min(len(tokens)//20, 500_000)
    val_tokens = tokens[-n_val:]
    tr_tokens  = tokens[:-n_val]

    cfg = CONFIGS['130m'].copy(); cfg['max_len'] = 1040
    model = Hv2LM(vocab_size, local_window=args.local_window, **cfg)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt.get('model_state', ckpt), strict=False)
    base_ppl = ckpt.get('val_ppl', 0)
    print(f"✓ 加载完成  base PPL={base_ppl:.2f}")

    model = patch_multihead(model, n_heads=args.n_heads,
                            orth_lambda=args.orth_lambda).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable:,} ({trainable/1e6:.3f}M)")
    print(f"  {args.n_heads}头参考光: {args.n_heads}×332K = {args.n_heads*332/1000:.1f}M")

    print("\n评估初始状态（应=base PPL）...")
    init_ppl = evaluate(model, val_tokens, vocab_size)
    print(f"初始PPL={init_ppl:.2f}  base={base_ppl:.2f}  差异={init_ppl-base_ppl:+.2f}")
    if abs(init_ppl - base_ppl) > 0.5:
        print("⚠️  差异过大"); return

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9,0.95), weight_decay=0.1)
    def lr_lambda(s):
        w = 200
        if s < w: return s/w
        return max(0.1, 0.5*(1+math.cos(math.pi*(s-w)/max(1,args.max_steps-w))))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    tr_ds = TokenDataset(tr_tokens, args.seq_len)
    tr_loader = DataLoader(tr_ds, batch_size=8, shuffle=True, num_workers=0)
    tr_iter = iter(tr_loader)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    best_ppl = init_ppl; step = 0; log_data = []

    print(f"\n开始训练（{args.max_steps}步）...")
    print(f"{'步数':>8} {'lm_loss':>9} {'orth_loss':>10} {'val_ppl':>10} {'vs_base':>10}")

    model.train()
    while step < args.max_steps:
        try: xb, yb = next(tr_iter)
        except StopIteration:
            tr_iter = iter(tr_loader); xb, yb = next(tr_iter)

        xb = xb.to(device).clamp(0, vocab_size-1)
        yb = yb.to(device).clamp(0, vocab_size-1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(xb)
            if hasattr(out,'logits'): out=out.logits
            B,T,V = out.shape
            lm_loss = F.cross_entropy(out.reshape(B*T,V), yb.reshape(B*T))
            orth_loss = getattr(model, '_last_orth_loss',
                                torch.tensor(0.0, device=device))
            loss = lm_loss + orth_loss

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sch.step(); step += 1

        if step % args.eval_every == 0:
            val_ppl = evaluate(model, val_tokens, vocab_size)
            delta = val_ppl - base_ppl
            sign = "↓" if delta < 0 else "↑"
            print(f"{step:>8} {lm_loss.item():>9.4f} "
                  f"{orth_loss.item():>10.4f} "
                  f"{val_ppl:>10.2f} {delta:>+9.2f}{sign}")
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                torch.save({'step':step,'val_ppl':val_ppl,'n_heads':args.n_heads,
                            'orth_lambda':args.orth_lambda,'base_ppl':base_ppl,
                            'model_state':model.state_dict()},
                           f'checkpoints/fdm_holo_mh{args.n_heads}_best.pt')
                print(f"  → 新最优 PPL={val_ppl:.2f} (vs base {base_ppl-val_ppl:+.2f})")
            log_data.append({'step':step,'val_ppl':val_ppl,'delta':delta,
                             'orth_loss':orth_loss.item()})
            with open(f'logs/holo_mh{args.n_heads}.jsonl','a') as f:
                f.write(json.dumps(log_data[-1])+'\n')

    print(f"\n{'='*60}")
    print(f"base={base_ppl:.2f}  初始={init_ppl:.2f}  最优={best_ppl:.2f}")
    print(f"多头正交参考光效果: {base_ppl-best_ppl:+.2f} PPL")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/hv2_freeze_scan_cont_best.pt')
    parser.add_argument('--n_heads',      type=int,   default=4)
    parser.add_argument('--orth_lambda',  type=float, default=0.01)
    parser.add_argument('--max_steps',    type=int,   default=5000)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--seq_len',      type=int,   default=1024)
    parser.add_argument('--eval_every',   type=int,   default=500)
    parser.add_argument('--local_window', type=int,   default=256)
    args = parser.parse_args()
    run(args)
