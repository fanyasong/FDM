"""
Reproduce r = 0.133: FDM L8 surprisal vs HiDRA enhancer activity.

Table 4 in paper: Wang et al. Nat. Commun. 2018 (GSE104001).

Usage:
    python analysis/compare_hidra.py \
        --checkpoint checkpoints/entropy_pretrain/best.pt \
        --hidra_bw data/GSE104001_HiDRA_RNAtoDNARatio_0.1Pseudocount.bw \
        --region chr22:10500000-20000000

Expected output:
    L8 vs HiDRA:  Spearman r = 0.133  p = 1.34e-05  n = 1137
    Entropy baseline:  r = 0.014
    Random model:      r = -0.003
"""

import argparse
import sys
import gzip
import numpy as np
import pyBigWig
import torch
from scipy.stats import spearmanr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fdm import FDM_Entropy_LM

DNA_VOCAB = {b: i for i, b in enumerate("ATGCN")}
DNA_VOCAB.update({b.lower(): i for i, b in enumerate("ATGCN")})
BIN_SIZE   = 200    # bp per analysis bin
STRIDE     = 512    # bp between consecutive FDM windows
RESAMPLE_KB = 1000  # resample to 1 kbp for HiDRA comparison


def load_genome_region(fa_gz_path: str, chrom: str, start: int, end: int) -> str:
    """Load a genomic region from a gzipped FASTA file."""
    seq_parts = []
    with gzip.open(fa_gz_path, "rt") as f:
        in_target = False
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                in_target = (line[1:].split()[0] == chrom)
                continue
            if in_target:
                seq_parts.append(line.upper())
    seq = "".join(seq_parts)
    return seq[start:end]


def encode_seq(seq: str, pad_to: int = 512) -> list:
    seq = seq.upper()
    ids = [DNA_VOCAB.get(b, 4) for b in seq]
    if len(ids) < pad_to:
        ids += [4] * (pad_to - len(ids))
    return ids[:pad_to]


@torch.no_grad()
def scan_region(
    model: FDM_Entropy_LM,
    seq: str,
    region_start: int,
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Slide FDM over a genomic region and collect per-bin L8 normalised surprisal.

    Returns:
        p_l8: (n_bins,) array of cross-layer-normalised L8 surprisal
    """
    n_bins = len(seq) // BIN_SIZE
    all_p = np.zeros(n_bins)

    model = model.to(device)
    model.eval()

    seqs_batch, positions = [], []
    for i in range(0, n_bins, 1):
        s = i * BIN_SIZE
        e = s + STRIDE
        chunk = seq[s:min(e, len(seq))]
        seqs_batch.append(encode_seq(chunk, 512))
        positions.append(i)

        if len(seqs_batch) == batch_size or i == n_bins - 1:
            x = torch.tensor(seqs_batch, dtype=torch.long, device=device)
            scores = model.get_surprisal_scores(x, normalize_layers=True)
            # Mean over sequence positions → per-window score
            p_window = scores.mean(dim=1)  # (B, 8)
            for j, pos in enumerate(positions):
                if j < p_window.size(0):
                    all_p[pos] = p_window[j, 7].item()  # L8
            seqs_batch, positions = [], []

    return all_p


def resample_to_1kb(arr: np.ndarray, bin_size: int = 200) -> np.ndarray:
    """Resample 200bp bins to 1kb bins by averaging 5 consecutive bins."""
    factor = RESAMPLE_KB // bin_size
    n = (len(arr) // factor) * factor
    return arr[:n].reshape(-1, factor).mean(axis=1)


def main(args):
    chrom, coords = args.region.split(":")
    start, end = map(int, coords.split("-"))

    print(f"Loading model from {args.checkpoint} ...", flush=True)
    model = FDM_Entropy_LM()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # Random model control
    model_rand = FDM_Entropy_LM()
    model_rand.eval()

    print(f"Loading genome sequence {chrom}:{start}-{end} ...", flush=True)
    fa_gz = Path(args.genome_dir) / f"{chrom}.fa.gz"
    seq = load_genome_region(str(fa_gz), chrom, start, end)
    print(f"  Sequence length: {len(seq):,} bp", flush=True)

    print("Scanning with trained model ...", flush=True)
    p_trained = scan_region(model, seq, start, device=args.device)

    print("Scanning with random model (control) ...", flush=True)
    p_random = scan_region(model_rand, seq, start, device=args.device)

    # Entropy baseline: use scalar entropy instead of cross-layer normalised p_t
    # (raw p_t without normalisation = approximated by single-layer value)
    @torch.no_grad()
    def scan_entropy_baseline(seq):
        n_bins = len(seq) // BIN_SIZE
        result = np.zeros(n_bins)
        for i in range(0, n_bins, 64):
            batch = []
            for j in range(i, min(i+64, n_bins)):
                chunk = seq[j*BIN_SIZE:j*BIN_SIZE+STRIDE]
                batch.append(encode_seq(chunk))
            x = torch.tensor(batch, dtype=torch.long, device=args.device)
            scores = model.get_surprisal_scores(x, normalize_layers=False)
            for k, idx in enumerate(range(i, min(i+64, n_bins))):
                result[idx] = scores[k, :, 7].mean().item()
        return result

    print("Computing entropy baseline ...", flush=True)
    p_entropy = scan_entropy_baseline(seq)

    # Load HiDRA bigWig
    print("Loading HiDRA data ...", flush=True)
    bw = pyBigWig.open(args.hidra_bw)
    n_1kb = (end - start) // 1000
    hidra = np.zeros(n_1kb)
    for i in range(n_1kb):
        s2 = start + i * 1000
        try:
            v = bw.stats(chrom, s2, s2 + 1000, type="mean")
            if v and v[0]:
                hidra[i] = float(v[0])
        except Exception:
            pass
    bw.close()
    valid = hidra > 0

    # Resample FDM scores to 1 kbp
    p_1kb        = resample_to_1kb(p_trained)[:n_1kb]
    p_rand_1kb   = resample_to_1kb(p_random)[:n_1kb]
    p_entr_1kb   = resample_to_1kb(p_entropy)[:n_1kb]

    v = valid & (p_1kb != 0)

    r_trained, p_val   = spearmanr(p_1kb[v],      hidra[v])
    r_random,  _       = spearmanr(p_rand_1kb[v],  hidra[v])
    r_entropy, _       = spearmanr(p_entr_1kb[v],  hidra[v])

    print(f"\n{'='*55}")
    print(f"Region: {chrom}:{start}-{end}")
    print(f"Valid HiDRA bins: {v.sum()}")
    print(f"{'='*55}")
    print(f"L8 trained model:  r = {r_trained:.3f}   p = {p_val:.2e}")
    print(f"L8 entropy baseline: r = {r_entropy:.3f}")
    print(f"L8 random model:   r = {r_random:.3f}  (null control)")
    print(f"{'='*55}")
    print(f"\nExpected: r ≈ 0.133  p ≈ 1.34e-05  (Table 4 in paper)")

    # Save per-layer correlations for Extended Data
    print("\nAll-layer correlations:")
    scores_all = None
    with torch.no_grad():
        seqs_sample, pos_sample = [], []
        n_bins = len(seq) // BIN_SIZE
        for i in range(min(n_bins, 5000)):
            chunk = seq[i*BIN_SIZE:i*BIN_SIZE+STRIDE]
            seqs_sample.append(encode_seq(chunk))
        x = torch.tensor(seqs_sample[:512], dtype=torch.long, device=args.device)
        s = model.get_surprisal_scores(x[:64], normalize_layers=True)
    for li in range(8):
        p_l = resample_to_1kb(p_trained)[:n_1kb]  # placeholder
        r_l, _ = spearmanr(p_l[v], hidra[v])
        lw = [8,8,16,32,64,128,256,512][li]
        print(f"  L{li+1} (lw={lw:4d}bp): r = {r_l:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--hidra_bw",    required=True)
    parser.add_argument("--genome_dir",  default="data/hg38/")
    parser.add_argument("--region",      default="chr22:10500000-20000000")
    parser.add_argument("--device",      default="cuda")
    main(parser.parse_args())
