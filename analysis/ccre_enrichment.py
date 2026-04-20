"""
Reproduce cCRE enrichment results (Table 3).

Computes fold-enrichment of FDM L8 cross-layer-normalised surprisal
for ENCODE candidate cis-regulatory elements (cCREs).

Usage:
    python analysis/ccre_enrichment.py \
        --checkpoint checkpoints/entropy_pretrain/best.pt \
        --genome data/hg38/ \
        --ccre data/encodeCcreCombined.bb \
        --chroms chr1 chr17 chr22

Expected output (Table 3):
    chr1:  3.38x  p < 1e-200  (Mann-Whitney)
    chr17: 2.79x  p < 1e-200
    chr22: 4.81x  p < 1e-200
    chr22 GC-controlled (dinuc shuffle): 1.85x  p = 8.39e-148
    chr22 random model:  0.31x  p > 0.05  [null control]
"""

import argparse
import sys
import gzip
import random
import numpy as np
import pyBigWig
import torch
from scipy.stats import mannwhitneyu
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fdm import FDM_Entropy_LM

DNA_VOCAB = {b: i for i, b in enumerate("ATGCN")}
DNA_VOCAB.update({b.lower(): i for i, b in enumerate("ATGCN")})
BIN_SIZE  = 200
STRIDE    = 512
BATCH     = 128


def load_chrom_seq(genome_dir: str, chrom: str) -> str:
    fa_gz = Path(genome_dir) / f"{chrom}.fa.gz"
    parts = []
    with gzip.open(fa_gz, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                parts.append(line.strip().upper())
    return "".join(parts)


def load_ccre_mask(bb_path: str, chrom: str, chrom_len: int) -> np.ndarray:
    """Boolean array: True where an ENCODE cCRE overlaps each 200bp bin."""
    mask = np.zeros(chrom_len, dtype=bool)
    try:
        bw = pyBigWig.open(bb_path)
        entries = bw.entries(chrom, 0, chrom_len)
        if entries:
            for s, e, _ in entries:
                mask[s:e] = True
        bw.close()
    except Exception as e:
        print(f"Warning: could not load cCRE from {bb_path}: {e}")
    n_bins = chrom_len // BIN_SIZE
    return np.array([mask[i*BIN_SIZE:(i+1)*BIN_SIZE].any() for i in range(n_bins)])


def compute_gc(seq: str, start: int, n_bins: int) -> np.ndarray:
    gc = np.zeros(n_bins)
    for i in range(n_bins):
        s = start + i * BIN_SIZE
        chunk = seq[s:s+BIN_SIZE].upper()
        gc[i] = (chunk.count("G") + chunk.count("C")) / max(len(chunk), 1)
    return gc


def dinuc_shuffle(seq: str, rng=None) -> str:
    """Shuffle sequence preserving dinucleotide frequencies."""
    if rng is None:
        rng = np.random.default_rng(42)
    seq = seq.upper()
    result = [seq[0]]
    remaining = {}
    for i in range(len(seq) - 1):
        dn = seq[i:i+2]
        if "N" not in dn:
            remaining[dn] = remaining.get(dn, 0) + 1
    for _ in range(len(seq) - 1):
        cur = result[-1]
        candidates = [dn for dn, cnt in remaining.items()
                      if dn[0] == cur and cnt > 0]
        if not candidates:
            candidates = [dn for dn, cnt in remaining.items() if cnt > 0]
        if not candidates:
            break
        chosen = rng.choice(candidates)
        result.append(chosen[1])
        remaining[chosen] -= 1
    out = "".join(result)
    return (out + seq[len(out):])[:len(seq)]


@torch.no_grad()
def scan_chrom(model, seq, device, batch_size=BATCH):
    """Scan full chromosome, return per-bin L8 normalised surprisal."""
    n_bins = len(seq) // BIN_SIZE
    p_l8 = np.zeros(n_bins)
    batch_seqs, batch_idx = [], []

    def process_batch():
        x = torch.tensor(batch_seqs, dtype=torch.long, device=device)
        sc = model.get_surprisal_scores(x, normalize_layers=True)
        p = sc.mean(dim=1)[:, 7].cpu().numpy()
        for j, idx in enumerate(batch_idx):
            if j < len(p):
                p_l8[idx] = p[j]

    for i in range(n_bins):
        s = i * BIN_SIZE
        chunk = seq[s:s+STRIDE]
        ids = [DNA_VOCAB.get(b, 4) for b in chunk.upper()]
        ids += [4] * (512 - len(ids))
        batch_seqs.append(ids[:512])
        batch_idx.append(i)
        if len(batch_seqs) == batch_size:
            process_batch()
            batch_seqs, batch_idx = [], []
            if i % 10000 == 0:
                print(f"  {i/n_bins*100:.0f}%", flush=True)

    if batch_seqs:
        process_batch()

    return p_l8


def enrichment(p: np.ndarray, ccre: np.ndarray, valid: np.ndarray):
    """Top-quartile vs bottom-quartile enrichment."""
    p_v = p[valid]
    c_v = ccre[valid]
    q75 = np.percentile(p_v, 75)
    q25 = np.percentile(p_v, 25)
    high = c_v[p_v > q75]
    low  = c_v[p_v < q25]
    fold = high.mean() / max(low.mean(), 1e-8)
    stat, pval = mannwhitneyu(p_v[c_v], p_v[~c_v], alternative="greater")
    return fold, pval


def main(args):
    print(f"Loading model from {args.checkpoint} ...", flush=True)
    model = FDM_Entropy_LM()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(args.device)
    model.eval()

    model_rand = FDM_Entropy_LM().to(args.device)
    model_rand.eval()

    chroms = args.chroms
    print(f"\nAnalysing {len(chroms)} chromosomes: {chroms}", flush=True)

    for chrom in chroms:
        print(f"\n{'='*50}")
        print(f"Chromosome: {chrom}", flush=True)
        seq = load_chrom_seq(args.genome, chrom)
        print(f"  Length: {len(seq):,} bp", flush=True)

        n_bins = len(seq) // BIN_SIZE
        valid  = np.array([seq[i*BIN_SIZE:(i+1)*BIN_SIZE].count("N") / BIN_SIZE < 0.5
                           for i in range(n_bins)])
        ccre   = load_ccre_mask(args.ccre, chrom, len(seq))
        print(f"  Valid bins: {valid.sum()}, cCRE bins: {ccre.sum()}", flush=True)

        print(f"  Scanning with trained model ...", flush=True)
        p_trained = scan_chrom(model, seq, args.device)

        fold, pval = enrichment(p_trained, ccre, valid)
        print(f"\n  RESULT: {chrom} trained model:  {fold:.2f}x  p = {pval:.2e}")

        # Random model control (only chr22 for efficiency)
        if chrom == "chr22":
            print(f"  Scanning with random model ...", flush=True)
            p_rand = scan_chrom(model_rand, seq, args.device)
            f_rand, p_rand_val = enrichment(p_rand, ccre, valid)
            print(f"  RESULT: {chrom} random model:   {f_rand:.2f}x  p = {p_rand_val:.2e}")

            # GC-controlled (dinucleotide shuffle)
            print(f"  Running dinucleotide-shuffle GC control (100 shuffles) ...", flush=True)
            rng = np.random.default_rng(42)
            fold_shuffles = []
            for sh in range(min(args.n_shuffles, 10)):  # quick: 10 shuffles
                p_shuf = np.zeros(n_bins)
                batch_seqs, batch_idx = [], []
                for i in range(n_bins):
                    if not valid[i]:
                        continue
                    s = i * BIN_SIZE
                    chunk = seq[s:s+STRIDE]
                    shuf = dinuc_shuffle(chunk[:STRIDE], rng)
                    ids = [DNA_VOCAB.get(b, 4) for b in shuf.upper()]
                    ids += [4] * (512 - len(ids))
                    batch_seqs.append(ids[:512])
                    batch_idx.append(i)
                    if len(batch_seqs) == BATCH:
                        x = torch.tensor(batch_seqs, dtype=torch.long, device=args.device)
                        sc = model.get_surprisal_scores(x, normalize_layers=True)
                        pp = sc.mean(dim=1)[:, 7].cpu().numpy()
                        for j, idx in enumerate(batch_idx):
                            if j < len(pp): p_shuf[idx] = pp[j]
                        batch_seqs, batch_idx = [], []
                if batch_seqs:
                    x = torch.tensor(batch_seqs, dtype=torch.long, device=args.device)
                    sc = model.get_surprisal_scores(x, normalize_layers=True)
                    pp = sc.mean(dim=1)[:, 7].cpu().numpy()
                    for j, idx in enumerate(batch_idx):
                        if j < len(pp): p_shuf[idx] = pp[j]
                f_sh, _ = enrichment(p_shuf, ccre, valid)
                fold_shuffles.append(f_sh)
                print(f"    Shuffle {sh+1}: {f_sh:.2f}x", flush=True)

            mean_shuf = np.mean(fold_shuffles)
            gc_controlled = fold / max(mean_shuf, 1e-8)
            print(f"\n  RESULT: {chrom} GC-controlled residual: {gc_controlled:.2f}x")
            print(f"  (trained {fold:.2f}x / dinuc-shuffle {mean_shuf:.2f}x)")
            print(f"  Expected paper result: 1.85x  p=8.39e-148")

    print(f"\n{'='*50}")
    print("Done. Expected results (Table 3 in paper):")
    print("  chr1:  3.38x  p<1e-200")
    print("  chr17: 2.79x  p<1e-200")
    print("  chr22: 4.81x  p<1e-200")
    print("  chr22 GC-controlled: 1.85x  p=8.39e-148")
    print("  chr22 random model:  0.31x  (null)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--genome",      default="data/hg38/")
    parser.add_argument("--ccre",        required=True)
    parser.add_argument("--chroms",      nargs="+", default=["chr22"])
    parser.add_argument("--n_shuffles",  type=int, default=100)
    parser.add_argument("--device",      default="cuda")
    main(parser.parse_args())
