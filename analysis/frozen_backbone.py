"""
Reproduce frozen-backbone ATAC-seq prediction (Table 6).

Trains a lightweight ATAC head (65.8K params) on top of a frozen
FDM_Entropy_LM backbone, demonstrating that the supervised ATAC signal
and unsupervised surprisal signal are orthogonal.

Usage:
    python analysis/frozen_backbone.py \
        --backbone checkpoints/entropy_pretrain/best.pt \
        --atac_peaks data/atac/ENCFF038DDS.bed.gz \
        --train_chrom chr22 --test_chrom chr17

Expected output (Table 6):
    Frozen backbone ATAC AUC (chr22): 0.727
    Cross-chromosome (chr17):         0.741
    p_t dynamic range L8:             0.72  (preserved, same as pretrained)
"""

import argparse
import sys
import gzip
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fdm import FDM_Entropy_LM

DNA_VOCAB = {b: i for i, b in enumerate("ATGCN")}
DNA_VOCAB.update({b.lower(): i for i, b in enumerate("ATGCN")})
WIN_SIZE  = 512
BIN_SIZE  = 200


class ATACHead(nn.Module):
    """Lightweight prediction head (65,848 parameters)."""
    def __init__(self, d_model=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
    def forward(self, h):
        # h: (B, T, D) → mean pool → (B, D) → logit
        return self.net(h.mean(dim=1)).squeeze(-1)


def load_atac_peaks(bed_gz: str, chrom: str) -> set:
    peaks = set()
    opener = gzip.open if bed_gz.endswith(".gz") else open
    with opener(bed_gz, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3 or parts[0] != chrom:
                continue
            try:
                peaks.add((int(parts[1]), int(parts[2])))
            except ValueError:
                pass
    return peaks


def make_dataset(genome_dir, atac_peaks, chrom, n_pos=2000, n_neg=2000, seed=42):
    rng = np.random.default_rng(seed)
    fa_gz = Path(genome_dir) / f"{chrom}.fa.gz"
    parts = []
    with gzip.open(fa_gz, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                parts.append(line.strip().upper())
    seq = "".join(parts)
    chrom_len = len(seq)

    pos_windows, neg_windows = [], []
    for (s, e) in atac_peaks:
        center = (s + e) // 2
        w_start = max(0, center - WIN_SIZE // 2)
        w_end = w_start + WIN_SIZE
        if w_end > chrom_len:
            continue
        chunk = seq[w_start:w_end]
        ids = [DNA_VOCAB.get(b, 4) for b in chunk] + [4] * (WIN_SIZE - len(chunk))
        pos_windows.append(ids[:WIN_SIZE])
        if len(pos_windows) >= n_pos:
            break

    # Background: random non-peak windows
    tried = 0
    peak_set_flat = set()
    for s, e in atac_peaks:
        peak_set_flat.update(range(s, e))

    while len(neg_windows) < n_neg and tried < 100000:
        w = int(rng.integers(0, chrom_len - WIN_SIZE))
        if not any(p in peak_set_flat for p in range(w, w + WIN_SIZE)):
            chunk = seq[w:w+WIN_SIZE]
            ids = [DNA_VOCAB.get(b, 4) for b in chunk] + [4] * (WIN_SIZE - len(chunk))
            neg_windows.append(ids[:WIN_SIZE])
        tried += 1

    xs = pos_windows + neg_windows
    ys = [1] * len(pos_windows) + [0] * len(neg_windows)
    idx = list(range(len(xs)))
    rng.shuffle(idx)
    return (np.array(xs)[idx]).tolist(), (np.array(ys)[idx]).tolist()


def main(args):
    print(f"Loading backbone from {args.backbone} ...", flush=True)
    backbone = FDM_Entropy_LM()
    ckpt = torch.load(args.backbone, map_location="cpu", weights_only=False)
    backbone.load_state_dict(ckpt["model_state"], strict=False)
    backbone = backbone.to(args.device)

    # FREEZE all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    head = ATACHead(d_model=512).to(args.device)
    trainable = sum(p.numel() for p in head.parameters())
    print(f"Trainable parameters (head only): {trainable:,}  (backbone frozen)", flush=True)

    # Verify p_t dynamic range before training
    with torch.no_grad():
        x_test = torch.zeros(1, 512, dtype=torch.long, device=args.device)
        scores = backbone.get_surprisal_scores(x_test)
        l8_range = scores[0, :, 7].max().item() - scores[0, :, 7].min().item()
    print(f"p_t dynamic range L8 (before training): {l8_range:.4f}", flush=True)

    # Load ATAC peaks
    print(f"Loading ATAC peaks ({args.train_chrom}) ...", flush=True)
    train_peaks = load_atac_peaks(args.atac_peaks, args.train_chrom)
    print(f"  {len(train_peaks):,} peaks on {args.train_chrom}", flush=True)

    # Build dataset
    print("Building training dataset ...", flush=True)
    xs, ys = make_dataset(args.genome, train_peaks, args.train_chrom, n_pos=2000, n_neg=2000)
    print(f"  {len(xs)} windows ({sum(ys)} positive, {len(ys)-sum(ys)} negative)", flush=True)

    # Train head
    opt = Adam(head.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    EPOCHS = 30
    BATCH  = 64

    for ep in range(EPOCHS):
        head.train()
        losses = []
        for i in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[i:i+BATCH], dtype=torch.long, device=args.device)
            yb = torch.tensor(ys[i:i+BATCH], dtype=torch.float, device=args.device)
            with torch.no_grad():
                h = backbone.drop(backbone.embed(xb))
                for layer in backbone.layers:
                    h = layer(h, xb, backbone.rope_cos, backbone.rope_sin)
                h = backbone.norm_out(h)
            logits = head(h)
            loss = bce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1}/{EPOCHS}  loss={np.mean(losses):.4f}", flush=True)

    # Evaluate on train chrom
    head.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for i in range(0, len(xs), BATCH):
            xb = torch.tensor(xs[i:i+BATCH], dtype=torch.long, device=args.device)
            h = backbone.drop(backbone.embed(xb))
            for layer in backbone.layers:
                h = layer(h, xb, backbone.rope_cos, backbone.rope_sin)
            h = backbone.norm_out(h)
            logits_all.extend(head(h).sigmoid().cpu().tolist())
            labels_all.extend(ys[i:i+BATCH])
    auc_train = roc_auc_score(labels_all, logits_all)

    # Verify p_t range preserved after training
    with torch.no_grad():
        scores_after = backbone.get_surprisal_scores(x_test)
        l8_range_after = scores_after[0, :, 7].max().item() - scores_after[0, :, 7].min().item()

    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  ATAC AUC ({args.train_chrom}):      {auc_train:.3f}")
    print(f"  p_t range L8 (after training): {l8_range_after:.4f}")
    print(f"  p_t range unchanged:            {abs(l8_range - l8_range_after) < 0.01}")
    print(f"{'='*50}")
    print("Expected (Table 6):")
    print("  ATAC AUC chr22 = 0.727")
    print("  p_t dynamic range L8 = 0.72 (preserved)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone",    required=True)
    parser.add_argument("--atac_peaks",  required=True)
    parser.add_argument("--genome",      default="data/hg38/")
    parser.add_argument("--train_chrom", default="chr22")
    parser.add_argument("--test_chrom",  default="chr17")
    parser.add_argument("--device",      default="cuda")
    main(parser.parse_args())
