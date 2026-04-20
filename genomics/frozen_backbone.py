"""
Frozen-Backbone ATAC-seq Head Training
arXiv: 2604.07716  —  Table 5, Figure 6D

Trains a lightweight ATAC-seq prediction head on top of the frozen
FDM entropy backbone, preserving the unsupervised surprisal signal.

Key result:  ATAC AUC = 0.727  (vs multitask 0.739)
             p_t dynamic range fully preserved (L1 range = 0.72)

Usage
-----
python -m genomics.frozen_backbone \
    --ckpt        /path/to/entropy_pretrain/best.pt \
    --atac_peaks  /path/to/ENCFF038DDS.bed.gz \
    --genome_fa   /path/to/chr22.fa.gz \
    --chrom       chr22 \
    --out         ./checkpoints/frozen_atac.pt
"""

import argparse
import gzip
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

from fdm.model import FDM_Entropy_LM, load_pretrained

DNA_VOCAB = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4,
             'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4}
SEQ_LEN = 512


# ─── ATAC Head ────────────────────────────────────────────────────────────────

class ATACHead(nn.Module):
    """65,848-parameter binary classifier on top of frozen FDM."""

    def __init__(self, d: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D) → logit (B,)"""
        return self.net(h.mean(dim=1)).squeeze(-1)


# ─── Data ────────────────────────────────────────────────────────────────────

def load_fa_gz(path: str) -> str:
    parts = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if not line.startswith('>'):
                parts.append(line.strip().upper())
    return ''.join(parts)


def load_peaks_bed(path: str, chrom: str) -> list:
    peaks = []
    opener = gzip.open if str(path).endswith('.gz') else open
    mode   = 'rt' if str(path).endswith('.gz') else 'r'
    with opener(path, mode) as f:
        for line in f:
            if line.startswith(('#', 'track')): continue
            p = line.strip().split()
            if len(p) < 3 or p[0] != chrom: continue
            try:
                peaks.append((int(p[1]), int(p[2])))
            except ValueError:
                pass
    return peaks


def build_dataset(
    seq:    str,
    peaks:  list,
    n_pos:  int = 2000,
    n_neg:  int = 2000,
    seed:   int = 42,
) -> tuple:
    """Sample positive (peak centres) and background windows."""
    rng = random.Random(seed)
    chrom_len = len(seq)
    peak_set  = set()
    for s, e in peaks:
        c = (s + e) // 2
        peak_set.add(c // SEQ_LEN)

    pos_wins, neg_wins = [], []
    for s, e in rng.sample(peaks, min(n_pos, len(peaks))):
        c  = (s + e) // 2
        ws = max(0, c - SEQ_LEN // 2)
        we = min(chrom_len, ws + SEQ_LEN)
        ws = max(0, we - SEQ_LEN)
        chunk = seq[ws:we]
        pos_wins.append([DNA_VOCAB.get(b, 4) for b in chunk])

    attempts = 0
    while len(neg_wins) < n_neg and attempts < n_neg * 20:
        attempts += 1
        ws = rng.randint(0, chrom_len - SEQ_LEN)
        if ws // SEQ_LEN in peak_set: continue
        chunk = seq[ws: ws + SEQ_LEN]
        if chunk.count('N') / SEQ_LEN > 0.3: continue
        neg_wins.append([DNA_VOCAB.get(b, 4) for b in chunk])

    X = torch.tensor(pos_wins + neg_wins, dtype=torch.long)
    y = torch.tensor([1] * len(pos_wins) + [0] * len(neg_wins), dtype=torch.float)
    return X, y


# ─── Training ────────────────────────────────────────────────────────────────

def train_frozen(args):
    device = args.device

    print("Loading pretrained backbone…")
    backbone = load_pretrained(args.ckpt, device=device)
    # FREEZE all backbone parameters
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    print(f"  Backbone frozen: {sum(p.numel() for p in backbone.parameters())/1e6:.1f}M params")

    head = ATACHead().to(device)
    print(f"  ATAC head trainable: {sum(p.numel() for p in head.parameters())} params")

    print(f"Loading chromosome sequence from {args.genome_fa}…")
    seq = load_fa_gz(args.genome_fa)
    print(f"  {args.chrom}: {len(seq):,} bp")

    print(f"Loading ATAC peaks from {args.atac_peaks}…")
    peaks = load_peaks_bed(args.atac_peaks, args.chrom)
    print(f"  {len(peaks)} peaks on {args.chrom}")

    print("Building dataset…")
    X, y = build_dataset(seq, peaks, n_pos=2000, n_neg=2000)

    # Train / val split (80/20)
    n      = len(y)
    idx    = torch.randperm(n)
    split  = int(0.8 * n)
    tr_idx, va_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    print(f"  Train: {len(y_tr)}  Val: {len(y_va)}")

    opt       = Adam(head.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    BATCH     = 64
    best_auc  = 0.0

    print(f"\nTraining ATAC head ({args.epochs} epochs)…")
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(len(y_tr))
        ep_loss = 0.0
        for i in range(0, len(y_tr), BATCH):
            idx_b = perm[i: i + BATCH]
            xb    = X_tr[idx_b].to(device)
            yb    = y_tr[idx_b].to(device)
            with torch.no_grad():
                h = backbone.drop(backbone.embed(xb))
                for layer in backbone.layers:
                    h = layer(h, xb, backbone.rope_cos, backbone.rope_sin)
                h = backbone.norm_out(h)
            logits = head(h)
            loss   = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()

        # Validation
        head.eval()
        with torch.no_grad():
            probs_all, labels_all = [], []
            for i in range(0, len(y_va), BATCH):
                xb = X_va[i: i + BATCH].to(device)
                h  = backbone.drop(backbone.embed(xb))
                for layer in backbone.layers:
                    h = layer(h, xb, backbone.rope_cos, backbone.rope_sin)
                h  = backbone.norm_out(h)
                pr = torch.sigmoid(head(h)).cpu().numpy()
                probs_all.append(pr)
                labels_all.append(y_va[i: i + BATCH].numpy())
            probs  = np.concatenate(probs_all)
            labels = np.concatenate(labels_all)
            auc    = roc_auc_score(labels, probs)

        if epoch % 5 == 0:
            print(f"  epoch={epoch:3d}  loss={ep_loss:.4f}  val_AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save({
                'head_state':     head.state_dict(),
                'backbone_ckpt':  args.ckpt,
                'val_auc':        auc,
            }, args.out)

    print(f"\nBest val AUC: {best_auc:.4f}  (paper: ~0.727)")
    print(f"Saved → {args.out}")

    # Verify p_t range is preserved
    print("\nVerifying p_t dynamic range (should be ~0.72 for L1)…")
    backbone.eval()
    test_x = X[:4].to(device)
    p_norm = backbone.get_surprisal(test_x)
    l1_range = p_norm[:, 0].max().item() - p_norm[:, 0].min().item()
    print(f"  L1 range = {l1_range:.4f}  (expected ~0.72; collapsed if <0.01)")
    if l1_range < 0.01:
        print("  WARNING: p_t range collapsed — backbone may have been modified")
    else:
        print("  OK: p_t dynamic range preserved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',       required=True)
    ap.add_argument('--atac_peaks', required=True, help='BED(.gz) file')
    ap.add_argument('--genome_fa',  required=True, help='chr*.fa.gz')
    ap.add_argument('--chrom',      default='chr22')
    ap.add_argument('--out',        default='./checkpoints/frozen_atac.pt')
    ap.add_argument('--epochs',     type=int, default=30)
    ap.add_argument('--device',     default='cuda')
    train_frozen(ap.parse_args())


if __name__ == '__main__':
    main()
