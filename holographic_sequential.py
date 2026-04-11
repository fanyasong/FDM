"""
全息参考光解码 v3：逐层顺序训练
==================================
从最后一层往前，每层单独训练W_ref_i，
前一层收敛后冻结，再训练下一层。

类似Freeze-Scan的思路：解耦梯度，避免层间竞争。
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


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens; self.seq_len = seq_len
        self.n = (len(tokens)-1)//seq_len
    def __len__(self): return self.n
    def __getitem__(self, idx):
        s = idx*self.seq_len
        return self.tokens[s:s+self.seq_len], self.tokens[s+1:s+self.seq_len+1]


def patch_layerwise(model):
    """加入每层W_ref，全部零初始化，全部冻结（等待逐层解冻）"""
    D = model.d
    dev = next(model.parameters()).device

    for p in model.parameters():
        p.requires_grad = False

    model.W_ref_layers = nn.ModuleList([
        nn.Linear(D, D, bias=True) for _ in range(model.n_layers)
    ]).to(dev)
    for w in model.W_ref_layers:
        nn.init.zeros_(w.weight)
        nn.init.zeros_(w.bias)
        for p in w.parameters():
            p.requires_grad = False  # 先全部冻结

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

        t_idx    = torch.arange(T, device=tokens.device, dtype=x.dtype)
        pos_norm = t_idx / max(T-1, 1)
        pe       = torch.stack([torch.sin(math.pi*pos_norm),
                                torch.cos(math.pi*pos_norm)], dim=-1)
        pos_corr = self.W_pos(pe).squeeze(-1).unsqueeze(0).expand(B,-1)
        ls       = torch.log(torch.arange(2, T+2, device=tokens.device, dtype=x.dtype))

        for i, layer in enumerate(self.layers):
            x_ref_i = x.detach()
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

            # 参考光调制（零初始化时=恒等）
            modulated = x_out * (1.0 + torch.tanh(self.W_ref_layers[i](x_ref_i)))
            # 已冻结的层detach，避免梯度流回干扰前面层的激活
            if not any(p.requires_grad for p in self.W_ref_layers[i].parameters()):
                x = modulated.detach()
            else:
                x = modulated

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


def train_one_layer(model, layer_idx, tr_tokens, val_tokens, vocab_size,
                    args, base_ppl, prev_best):
    """训练单层的W_ref_i"""
    # 只解冻当前层
    for p in model.W_ref_layers[layer_idx].parameters():
        p.requires_grad = True
    # norm_out_holo也参与
    for p in model.norm_out_holo.parameters():
        p.requires_grad = True

    trainable_params = (list(model.W_ref_layers[layer_idx].parameters()) +
                        list(model.norm_out_holo.parameters()))

    opt = torch.optim.AdamW(trainable_params, lr=args.lr,
                            betas=(0.9,0.95), weight_decay=0.1)
    def lr_lambda(s):
        w = 100
        if s < w: return s/w
        return max(0.1, 0.5*(1+math.cos(math.pi*(s-w)/max(1,args.steps_per_layer-w))))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    tr_ds = TokenDataset(tr_tokens, args.seq_len)
    tr_loader = DataLoader(tr_ds, batch_size=8, shuffle=True, num_workers=0)
    tr_iter = iter(tr_loader)

    best_ppl = prev_best; step = 0
    print(f"\n  层 {layer_idx:2d}/{model.n_layers-1}  训练W_ref_{layer_idx}...")

    model.train()
    while step < args.steps_per_layer:
        try: xb, yb = next(tr_iter)
        except StopIteration:
            tr_iter = iter(tr_loader); xb, yb = next(tr_iter)

        xb = xb.to(device).clamp(0, vocab_size-1)
        yb = yb.to(device).clamp(0, vocab_size-1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model(xb)
            if hasattr(out,'logits'): out=out.logits
            B,T,V = out.shape
            loss = F.cross_entropy(out.reshape(B*T,V), yb.reshape(B*T))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step(); sch.step(); step += 1

        if step % args.eval_every == 0:
            val_ppl = evaluate(model, val_tokens, vocab_size)
            delta = val_ppl - base_ppl
            sign = "↓" if delta < 0 else "↑"
            print(f"    step={step:4d} loss={loss.item():.4f} "
                  f"ppl={val_ppl:.2f} vs_base={delta:+.2f}{sign}")
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                torch.save(
                    {'layer': layer_idx, 'val_ppl': val_ppl,
                     'base_ppl': base_ppl, 'model_state': model.state_dict()},
                    f'checkpoints/fdm_holo_seq_layer{layer_idx:02d}_best.pt')
                print(f"    → 层{layer_idx}最优 PPL={val_ppl:.2f}")

    # 冻结当前层，进入下一层
    for p in model.W_ref_layers[layer_idx].parameters():
        p.requires_grad = False

    return best_ppl


def run(args):
    from train_130m import Hv2LM, CONFIGS

    print(f"\n{'='*60}")
    print(f"全息参考光解码 v3：逐层顺序训练（从前往后，波态→粒子态）")
    print(f"每层步数={args.steps_per_layer}  lr={args.lr}")
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

    model = patch_layerwise(model).to(device)

    print("评估初始状态...")
    init_ppl = evaluate(model, val_tokens, vocab_size)
    print(f"初始PPL={init_ppl:.2f}  base={base_ppl:.2f}  差异={init_ppl-base_ppl:+.2f}")
    if abs(init_ppl - base_ppl) > 0.5:
        print("⚠️  差异过大"); return

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    current_best = init_ppl
    layer_results = []

    # 从第一层往后训练（波态→粒子态方向）
    for layer_idx in range(0, model.n_layers):
        prev_best = current_best
        current_best = train_one_layer(
            model, layer_idx, tr_tokens, val_tokens, vocab_size,
            args, base_ppl, current_best)

        improvement = prev_best - current_best
        layer_results.append({'layer': layer_idx, 'ppl': current_best,
                               'improvement': improvement})
        print(f"  层{layer_idx}完成: PPL={current_best:.2f} "
              f"(本层提升={improvement:+.2f})")

        with open('logs/holo_sequential.jsonl', 'a') as f:
            f.write(json.dumps(layer_results[-1]) + '\n')

    print(f"\n{'='*60}")
    print(f"逐层训练完成！")
    print(f"base={base_ppl:.2f}  初始={init_ppl:.2f}  最终={current_best:.2f}")
    print(f"总提升: {base_ppl - current_best:+.2f} PPL")
    print(f"\n各层贡献：")
    for r in sorted(layer_results, key=lambda x: x['improvement'], reverse=True):
        print(f"  层{r['layer']:2d}: {r['improvement']:+.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/hv2_freeze_scan_cont_best.pt')
    parser.add_argument('--steps_per_layer', type=int,   default=2000)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--seq_len',         type=int,   default=1024)
    parser.add_argument('--eval_every',      type=int,   default=500)
    parser.add_argument('--local_window',    type=int,   default=256)
    args = parser.parse_args()
    run(args)
