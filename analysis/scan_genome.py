"""
Whole-chromosome surprisal scan
arXiv: 2604.07716  —  Figure 5, Table 2

Slides a 512 bp window across a chromosome and records per-layer
cross-layer-normalised p_t values. Used for cCRE enrichment and
HiDRA / MPRA correlation analyses.

Usage
-----
python -m analysis.scan_genome \
    --fasta   /path/to/chr22.fa.gz \
    --ckpt    /path/to/best.pt \
    --out     ./results/chr22_surprisal.npy \
    --chrom   chr22 \
    --stride  512
"""

import argparse
import gzip
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fdm.model import FDM_Entropy_LM, EntropyCollapseBlock, load_pretrained

DNA_VOCAB = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4,
             'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4}


def load_fa_gz(path: str) -> str:
    parts = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if not line.startswith('>'):
                parts.append(line.strip().upper())
    return ''.join(parts)


def scan_chromosome(
    model:  FDM_Entropy_LM,
    seq:    str,
    stride: int   = 512,
    batch:  int   = 64,
    device: str   = 'cuda',
) -> np.ndarray:
    """
    Slide a 512 bp window across seq with given stride.

    Returns
    -------
    p_norm : ndarray of shape (n_windows, 8)
        Cross-layer-normalised mean p_t per window per layer.
        Column 7 = L8 (512 bp receptive field) used in paper analyses.
    """
    SEQ_LEN = 512
    n_windows = (len(seq) - SEQ_LEN) // stride + 1

    # ── Patch ECB to record p values ─────────────────────────────────────
    records: dict = {}
    order:   list = []
    orig = EntropyCollapseBlock.forward

    def hook(self, h, tokens, rc, rs):
        xn = self.norm1(h)
        p  = self._compute_p(xn, tokens)
        lid = id(self)
        if lid not in records:
            records[lid] = []
            order.append(lid)
        records[lid].append(p.detach().cpu().mean(dim=1).squeeze(-1))   # (B,)
        return orig(self, h, tokens, rc, rs)

    EntropyCollapseBlock.forward = hook

    # Initialise layer order with a dummy forward pass
    dummy = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    with torch.no_grad():
        model(dummy)
    for lid in order:
        records[lid].clear()

    # ── Sliding window ───────────────────────────────────────────────────
    all_p = []
    windows_todo = []

    for i in range(n_windows):
        s = i * stride
        chunk = seq[s: s + SEQ_LEN]
        toks  = [DNA_VOCAB.get(b, 4) for b in chunk]
        windows_todo.append(toks)

        if len(windows_todo) == batch or i == n_windows - 1:
            x = torch.tensor(windows_todo, dtype=torch.long, device=device)
            for lid in order:
                records[lid].clear()
            with torch.no_grad():
                model(x)

            # Stack layers → (batch, 8)
            lm = [records[lid][0].numpy() for lid in order if records[lid]]
            if lm:
                mat = np.stack(lm, axis=1)                               # (B, 8)
                all_p.append(mat)

            windows_todo.clear()

            if i % 1000 == 0:
                pct = i / n_windows * 100
                print(f"  {i}/{n_windows}  ({pct:.1f}%)", flush=True)

    EntropyCollapseBlock.forward = orig

    p_abs = np.concatenate(all_p, axis=0)                                # (N, 8)

    # Cross-layer normalisation (eliminates GC bias)
    p_norm = p_abs / (p_abs.mean(axis=1, keepdims=True) + 1e-8)

    return p_norm


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Whole-chromosome surprisal scan')
    ap.add_argument('--fasta',   required=True, help='chr*.fa.gz')
    ap.add_argument('--ckpt',    required=True, help='best.pt checkpoint')
    ap.add_argument('--out',     required=True, help='output .npy path')
    ap.add_argument('--chrom',   default='chr22')
    ap.add_argument('--stride',  type=int, default=512)
    ap.add_argument('--batch',   type=int, default=64)
    ap.add_argument('--device',  default='cuda')
    args = ap.parse_args()

    print(f"Loading model from {args.ckpt}…")
    model = load_pretrained(args.ckpt, device=args.device)

    print(f"Loading {args.chrom} sequence…")
    seq = load_fa_gz(args.fasta)
    print(f"  {len(seq):,} bp")

    print(f"Scanning (stride={args.stride}, batch={args.batch})…")
    p_norm = scan_chromosome(model, seq,
                              stride=args.stride,
                              batch=args.batch,
                              device=args.device)

    print(f"Result shape: {p_norm.shape}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, p_norm)
    print(f"Saved → {args.out}")
    print(f"L8 (col 7) range: {p_norm[:, 7].min():.4f} – {p_norm[:, 7].max():.4f}")


if __name__ == '__main__':
    main()
