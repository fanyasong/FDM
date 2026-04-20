"""
FDM Genomic Pretraining
arXiv: 2604.07716

Trains FDM_Entropy_LM autoregressively on the hg38 reference genome.

Usage
-----
python -m fdm.train \
    --genome_dir  /path/to/hg38_fa_gz/ \
    --output_dir  ./checkpoints/entropy_pretrain \
    --steps       200000 \
    --seq_len     512 \
    --batch       8 \
    --lr          3e-4 \
    --warmup      2000
"""

import argparse
import gzip
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import FDM_Entropy_LM

# ─── DNA Tokenisation ────────────────────────────────────────────────────────

DNA_VOCAB = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4,
             'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4}
PAD_ID = 5

TRAIN_CHROMS = [
    'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7',
    'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
    'chr16', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX',
]
VAL_CHROM = 'chr17'


def load_fasta_gz(path: str) -> str:
    parts = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('>'): continue
            parts.append(line.strip().upper())
    return ''.join(parts)


def encode(seq: str) -> list:
    return [DNA_VOCAB.get(b, 4) for b in seq]


# ─── Dataset ─────────────────────────────────────────────────────────────────

class GenomeDataset:
    """Randomly samples fixed-length windows from the genome."""

    def __init__(self, seqs: dict, seq_len: int, seed: int = 42):
        self.seqs    = seqs          # {chrom: np.uint8 array of token ids}
        self.seq_len = seq_len
        self.rng     = random.Random(seed)

    def __iter__(self):
        while True:
            chrom = self.rng.choice(list(self.seqs.keys()))
            seq   = self.seqs[chrom]
            if len(seq) < self.seq_len: continue
            s = self.rng.randint(0, len(seq) - self.seq_len)
            yield seq[s: s + self.seq_len]


def make_batch(dataset_iter, batch_size: int, device: str) -> torch.Tensor:
    seqs = [next(dataset_iter) for _ in range(batch_size)]
    return torch.tensor(np.stack(seqs), dtype=torch.long, device=device)


# ─── Training ────────────────────────────────────────────────────────────────

def compute_ppl(model, val_seqs: dict, seq_len: int, n_batches: int,
                batch: int, device: str) -> float:
    model.eval()
    ds = GenomeDataset(val_seqs, seq_len, seed=0)
    it = iter(ds)
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x = make_batch(it, batch, device)
            logits = model(x)
            loss = nn.CrossEntropyLoss()(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
            )
            total_loss += loss.item()
    model.train()
    return float(np.exp(total_loss / n_batches))


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)

    # ── Load genome ──────────────────────────────────────────────────────
    print("Loading genome…")
    genome_dir = Path(args.genome_dir)
    train_seqs, val_seqs = {}, {}
    for chrom in TRAIN_CHROMS + [VAL_CHROM]:
        fa = genome_dir / f"{chrom}.fa.gz"
        if not fa.exists():
            print(f"  skip {chrom} (file not found)")
            continue
        raw = load_fasta_gz(str(fa))
        arr = np.array(encode(raw), dtype=np.uint8)
        mbp = len(arr) / 1e6
        label = "(val)" if chrom == VAL_CHROM else ""
        print(f"  {chrom}{label}: {mbp:.0f} Mbp")
        if chrom == VAL_CHROM:
            val_seqs[chrom] = arr
        else:
            train_seqs[chrom] = arr

    # ── Model ────────────────────────────────────────────────────────────
    model = FDM_Entropy_LM().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params/1e6:.1f}M")
    print(f"Device:     {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best_ppl   = float('inf')

    # Resume from checkpoint if present
    ckpt_path = out_dir / 'best.pt'
    if ckpt_path.exists() and not args.restart:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state'], strict=False)
        start_step = state.get('step', 0)
        best_ppl   = state.get('ppl',  float('inf'))
        print(f"Resumed from step={start_step}  ppl={best_ppl:.4f}")

    # ── Optimiser ────────────────────────────────────────────────────────
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                weight_decay=0.01)
    scheduler = CosineAnnealingLR(opt, T_max=args.steps - args.warmup,
                                  eta_min=args.lr * 0.1)

    def warmup_lr(step):
        if step < args.warmup:
            opt.param_groups[0]['lr'] = args.lr * (step + 1) / args.warmup

    # ── Training loop ────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    ds_iter   = iter(GenomeDataset(train_seqs, args.seq_len))
    model.train()

    print(f"\nTraining step={start_step} → {args.steps}")
    t0 = time.time()

    for step in range(start_step, args.steps):
        warmup_lr(step)
        x      = make_batch(ds_iter, args.batch, device)
        logits = model(x)
        loss   = criterion(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step >= args.warmup:
            scheduler.step()

        if step % 1000 == 0 and step > start_step:
            elapsed = time.time() - t0
            print(f"  step={step:7d}  loss={loss.item():.4f}  "
                  f"ppl={float(np.exp(loss.item())):.2f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"elapsed={elapsed/3600:.1f}h")

        if step % 5000 == 0 and step > start_step:
            ppl = compute_ppl(model, val_seqs, args.seq_len,
                              n_batches=50, batch=args.batch, device=device)
            print(f"  [VAL] step={step}  ppl={ppl:.4f}")
            if ppl < best_ppl:
                best_ppl = ppl
                torch.save({'model_state': model.state_dict(),
                            'step': step, 'ppl': ppl},
                           ckpt_path)
                print(f"  Saved best checkpoint  ppl={ppl:.4f}")

    # Final validation
    ppl = compute_ppl(model, val_seqs, args.seq_len,
                      n_batches=100, batch=args.batch, device=device)
    print(f"\nFinal val PPL: {ppl:.4f}")
    torch.save({'model_state': model.state_dict(),
                'step': args.steps, 'ppl': ppl},
               out_dir / 'final.pt')


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='FDM genomic pretraining')
    p.add_argument('--genome_dir',  required=True)
    p.add_argument('--output_dir',  default='./checkpoints/entropy_pretrain')
    p.add_argument('--steps',       type=int,   default=200_000)
    p.add_argument('--seq_len',     type=int,   default=512)
    p.add_argument('--batch',       type=int,   default=8)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--warmup',      type=int,   default=2000)
    p.add_argument('--restart',     action='store_true')
    train(p.parse_args())


if __name__ == '__main__':
    main()
