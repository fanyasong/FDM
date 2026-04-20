"""Rotary Position Embedding (RoPE) utilities."""
import torch
import torch.nn as nn


def get_rope_freqs(d_model: int, max_seq_len: int = 1_048_576):
    """
    Precompute RoPE cos/sin buffers.

    Args:
        d_model: hidden dimension (must be even)
        max_seq_len: maximum sequence length
    Returns:
        cos, sin: each shape (max_seq_len, d_model // 2)
    """
    half = d_model // 2
    theta = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos   = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, theta)          # (T, half)
    return freqs.cos(), freqs.sin()


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple:
    """
    Apply rotary embeddings to queries and keys.

    Args:
        q, k: (B, T, D)
        cos, sin: (T_max, D//2)
    Returns:
        q_rot, k_rot: (B, T, D)
    """
    T = q.size(1)
    c = cos[:T].unsqueeze(0)   # (1, T, D//2)
    s = sin[:T].unsqueeze(0)

    def rotate(x):
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

    return rotate(q), rotate(k)
