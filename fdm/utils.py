"""Shared utilities: DNA tokenisation, RoPE position embeddings, normalisation."""

import gzip
from typing import Tuple
import numpy as np
import torch

DNA_VOCAB = {
    'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4,
    'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4,
}
VOCAB_SIZE = 6

def tokenise(seq: str) -> list:
    return [DNA_VOCAB.get(b, 4) for b in seq]

def load_fasta_gz(path: str) -> str:
    parts = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                continue
            parts.append(line.strip().upper())
    return ''.join(parts)

def get_rope_freqs(d: int, max_seq_len: int = 1_048_576) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cosine/sine tables."""
    assert d % 2 == 0
    half = d // 2
    theta = 1.0 / (10_000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, theta)
    return freqs.cos(), freqs.sin()

def apply_rope(q, k, cos, sin):
    """Apply rotary position embeddings."""
    B, T, D = q.shape
    half = D // 2
    cos_t = cos[:T].unsqueeze(0)
    sin_t = sin[:T].unsqueeze(0)
    def rotate(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_t - x2 * sin_t,
                          x1 * sin_t + x2 * cos_t], dim=-1)
    return rotate(q), rotate(k)

def cross_layer_normalise(p_matrix: np.ndarray) -> np.ndarray:
    """
    Cross-layer normalisation: divide each position's 8-layer vector by its mean.
    Eliminates GC/repeat confounders that affect all layers equally.
    """
    mean = p_matrix.mean(axis=1, keepdims=True)
    mean = np.where(mean == 0, 1e-8, mean)
    return p_matrix / mean
