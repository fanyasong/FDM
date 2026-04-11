"""
标准NLP Benchmark评估
======================
评估FDM和Transformer在标准任务上的zero-shot性能：

1. LAMBADA: 长程依赖，预测段落最后一个词
   - 指标：accuracy（准确率）
   - 数据集：~5K测试样本

2. HellaSwag: 常识推理，从4个选项选最合理的续写
   - 指标：accuracy
   - 数据集：~10K测试样本

3. PIQA: 物理常识，判断两种方法哪个更合理
   - 指标：accuracy
   - 数据集：~1.8K测试样本

评估方法：language model scoring
  对每个选项计算log-likelihood，选最高分的选项
"""
import sys, os, math, argparse, json
import glob as _glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

for _f in _glob.glob('/root/**/triton_scan_v2.py', recursive=True) + \
          _glob.glob('/home/**/triton_scan_v2.py', recursive=True):
    with open(_f) as ff:
        if 'def mipt_scan' in ff.read():
            sys.path.insert(0, os.path.dirname(_f)); break

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(ckpt_path, model_type='hv2', vocab_size=100277):
    from train_130m import Hv2LM, TransformerLM, CONFIGS
    cfg = CONFIGS['130m'].copy(); cfg['max_len'] = 1040

    if model_type == 'hv2':
        model = Hv2LM(vocab_size, local_window=256, **cfg)
    else:
        model = TransformerLM(vocab_size, **cfg)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt.get('model_state', ckpt), strict=False)
        ppl = ckpt.get('val_ppl', 0)
        print(f"✓ 加载: {ckpt_path}  PPL={ppl:.1f}")
    else:
        print(f"⚠️  Checkpoint不存在: {ckpt_path}")

    return model.to(device).eval()


def get_tokenizer():
    """加载tiktoken BPE分词器（和训练时一致）"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return enc
    except ImportError:
        print("安装tiktoken: pip install tiktoken")
        return None


@torch.no_grad()
def score_sequence(model, input_ids, target_ids, vocab_size):
    """
    计算target_ids的log-likelihood
    input_ids: context tokens
    target_ids: tokens to score
    """
    all_ids = torch.cat([input_ids, target_ids], dim=0)
    if len(all_ids) > 1024:
        all_ids = all_ids[-1024:]  # 截断到最大长度

    x = all_ids[:-1].unsqueeze(0).to(device).clamp(0, vocab_size-1)
    y = all_ids[1:].unsqueeze(0).to(device).clamp(0, vocab_size-1)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x)
        if hasattr(logits, 'logits'): logits = logits.logits

    # 只计算target部分的loss
    context_len = len(input_ids) - 1
    target_logits = logits[0, context_len:context_len+len(target_ids)]
    target_labels = y[0, context_len:context_len+len(target_ids)]

    if len(target_logits) == 0:
        return float('-inf')

    log_probs = F.log_softmax(target_logits, dim=-1)
    token_log_probs = log_probs[
        torch.arange(len(target_labels)), target_labels]
    return token_log_probs.sum().item()


# ══════════════════════════════════════════════
# LAMBADA评估
# ══════════════════════════════════════════════

def eval_lambada(model, enc, vocab_size, max_samples=5000):
    """
    LAMBADA: 给定段落上文，预测最后一个词
    zero-shot accuracy
    """
    print("\n评估 LAMBADA...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("lambada", split="test",
                               trust_remote_code=True)
    except Exception as e:
        print(f"  加载LAMBADA失败: {e}")
        print("  尝试: pip install datasets")
        return None

    correct = 0; total = 0
    model.eval()

    for item in dataset:
        if total >= max_samples: break

        text = item['text']
        # 分割：最后一个词作为target
        words = text.strip().split()
        if len(words) < 2: continue

        last_word = words[-1]
        context = ' '.join(words[:-1])

        ctx_ids  = torch.tensor(enc.encode(context), dtype=torch.long)
        # 注意：最后一个词前面有空格
        tgt_ids  = torch.tensor(
            enc.encode(' ' + last_word), dtype=torch.long)

        # 预测：取context最后token的logit，看last_word是否是top-1
        if len(ctx_ids) == 0: continue
        x = ctx_ids[-1023:].unsqueeze(0).to(device).clamp(0, vocab_size-1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            if hasattr(logits, 'logits'): logits = logits.logits

        pred_id = logits[0, -1].argmax().item()
        # 检查pred_id是否对应last_word的第一个token
        if len(tgt_ids) > 0 and pred_id == tgt_ids[0].item():
            correct += 1
        total += 1

        if total % 500 == 0:
            print(f"  进度: {total}/{max_samples}  "
                  f"acc={correct/total:.3f}")

    acc = correct / max(total, 1)
    print(f"  LAMBADA: {correct}/{total} = {acc:.4f} ({acc*100:.1f}%)")
    return acc


# ══════════════════════════════════════════════
# HellaSwag评估
# ══════════════════════════════════════════════

def eval_hellaswag(model, enc, vocab_size, max_samples=1000):
    """
    HellaSwag: 4选1，选最合理的续写
    用language model scoring
    """
    print("\n评估 HellaSwag...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("hellaswag", split="validation",
                               trust_remote_code=True)
    except Exception as e:
        print(f"  加载HellaSwag失败: {e}")
        return None

    correct = 0; total = 0
    model.eval()

    for item in dataset:
        if total >= max_samples: break

        ctx = item['ctx']
        endings = item['endings']
        label = int(item['label'])

        ctx_ids = torch.tensor(enc.encode(ctx), dtype=torch.long)
        scores = []
        for ending in endings:
            end_ids = torch.tensor(enc.encode(' ' + ending), dtype=torch.long)
            score = score_sequence(model, ctx_ids, end_ids, vocab_size)
            # 归一化：除以token数避免长度偏差
            score = score / max(len(end_ids), 1)
            scores.append(score)

        pred = scores.index(max(scores))
        if pred == label: correct += 1
        total += 1

        if total % 200 == 0:
            print(f"  进度: {total}/{max_samples}  "
                  f"acc={correct/total:.3f}")

    acc = correct / max(total, 1)
    print(f"  HellaSwag: {correct}/{total} = {acc:.4f} ({acc*100:.1f}%)")
    return acc


# ══════════════════════════════════════════════
# PIQA评估
# ══════════════════════════════════════════════

def eval_piqa(model, enc, vocab_size, max_samples=1838):
    """
    PIQA: 物理常识，2选1
    """
    print("\n评估 PIQA...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("piqa", split="validation",
                               trust_remote_code=True)
    except Exception as e:
        print(f"  加载PIQA失败: {e}")
        return None

    correct = 0; total = 0
    model.eval()

    for item in dataset:
        if total >= max_samples: break

        goal = item['goal']
        sol1 = item['sol1']
        sol2 = item['sol2']
        label = int(item['label'])

        ctx_ids = torch.tensor(enc.encode(goal), dtype=torch.long)

        scores = []
        for sol in [sol1, sol2]:
            sol_ids = torch.tensor(enc.encode(' ' + sol), dtype=torch.long)
            score = score_sequence(model, ctx_ids, sol_ids, vocab_size)
            score = score / max(len(sol_ids), 1)
            scores.append(score)

        pred = scores.index(max(scores))
        if pred == label: correct += 1
        total += 1

        if total % 200 == 0:
            print(f"  进度: {total}/{max_samples}  "
                  f"acc={correct/total:.3f}")

    acc = correct / max(total, 1)
    print(f"  PIQA: {correct}/{total} = {acc:.4f} ({acc*100:.1f}%)")
    return acc


# ══════════════════════════════════════════════
# WinoGrande评估（代词消解，测长程依赖）
# ══════════════════════════════════════════════

def eval_winogrande(model, enc, vocab_size, max_samples=1267):
    """
    WinoGrande: 2选1，填空题，测试常识推理
    FDM在这种需要长程依赖的任务上应该有优势
    """
    print("\n评估 WinoGrande...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("winogrande", "winogrande_xl",
                               split="validation",
                               trust_remote_code=True)
    except Exception as e:
        print(f"  加载WinoGrande失败: {e}")
        return None

    correct = 0; total = 0
    model.eval()

    for item in dataset:
        if total >= max_samples: break

        sentence = item['sentence']
        opt1 = item['option1']
        opt2 = item['option2']
        answer = item['answer']  # '1' or '2'

        scores = []
        for opt in [opt1, opt2]:
            filled = sentence.replace('_', opt)
            ids = torch.tensor(enc.encode(filled), dtype=torch.long)
            if len(ids) < 2:
                scores.append(float('-inf')); continue
            x = ids[:-1].unsqueeze(0).to(device).clamp(0, vocab_size-1)
            y = ids[1:].unsqueeze(0).to(device).clamp(0, vocab_size-1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                if hasattr(logits,'logits'): logits=logits.logits
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1), reduction='mean')
            scores.append(-loss.item())  # 负loss作为score

        pred = scores.index(max(scores))
        label = int(answer) - 1  # '1'->'0', '2'->'1'
        if pred == label: correct += 1
        total += 1

        if total % 200 == 0:
            print(f"  进度: {total}/{max_samples}  "
                  f"acc={correct/total:.3f}")

    acc = correct / max(total, 1)
    print(f"  WinoGrande: {correct}/{total} = {acc:.4f} ({acc*100:.1f}%)")
    return acc


# ══════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════

def run(args):
    print(f"\n{'='*60}")
    print(f"NLP Benchmark评估")
    print(f"模型: {args.model_type}  Checkpoint: {args.ckpt}")
    print(f"{'='*60}\n")

    # 安装依赖
    os.system("pip install datasets tiktoken -q")

    enc = get_tokenizer()
    if enc is None: return

    vocab_size = enc.n_vocab
    print(f"词表大小: {vocab_size}")

    model = load_model(args.ckpt, args.model_type, vocab_size)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {params:.1f}M\n")

    results = {}

    if 'lambada' in args.tasks or 'all' in args.tasks:
        acc = eval_lambada(model, enc, vocab_size, args.max_samples)
        if acc: results['lambada'] = acc

    if 'hellaswag' in args.tasks or 'all' in args.tasks:
        acc = eval_hellaswag(model, enc, vocab_size, args.max_samples)
        if acc: results['hellaswag'] = acc

    if 'piqa' in args.tasks or 'all' in args.tasks:
        acc = eval_piqa(model, enc, vocab_size, args.max_samples)
        if acc: results['piqa'] = acc

    if 'winogrande' in args.tasks or 'all' in args.tasks:
        acc = eval_winogrande(model, enc, vocab_size, args.max_samples)
        if acc: results['winogrande'] = acc

    print(f"\n{'='*60}")
    print(f"评估完成！{args.model_type} 结果汇总：")
    print(f"{'='*60}")
    for task, acc in results.items():
        print(f"  {task:15s}: {acc:.4f} ({acc*100:.1f}%)")

    # 随机猜测基线
    print(f"\n随机猜测基线：")
    print(f"  LAMBADA:     ~0.000 (开放词表)")
    print(f"  HellaSwag:   0.250 (4选1)")
    print(f"  PIQA:        0.500 (2选1)")
    print(f"  WinoGrande:  0.500 (2选1)")

    os.makedirs('results', exist_ok=True)
    out_path = f'results/benchmark_{args.model_type}.json'
    with open(out_path, 'w') as f:
        json.dump({'model': args.model_type, 'ckpt': args.ckpt,
                   'results': results}, f, indent=2)
    print(f"\n结果保存至: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/hv2_freeze_scan_cont_best.pt')
    parser.add_argument('--model_type', default='hv2', choices=['hv2','tf'])
    parser.add_argument('--tasks', nargs='+',
                        default=['lambada','hellaswag','piqa','winogrande'])
    parser.add_argument('--max_samples', type=int, default=1000)
    args = parser.parse_args()
    run(args)
