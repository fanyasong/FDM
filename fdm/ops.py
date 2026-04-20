"""
Triton scan kernel and fast ops.

If Triton is not available, the sequential fallback in layers.py is used.
For production, Triton provides ~10x speedup on wave scan.
"""
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    mipt_scan = None

import torch

if TRITON_AVAILABLE:
    @triton.jit
    def _mipt_scan_kernel(
        p_ptr, state_ptr, x_ptr, out_ptr,
        B, T, D,
        stride_b, stride_t, stride_d,
        BLOCK_D: tl.constexpr,
    ):
        """Parallelised linear recurrent scan with gating."""
        b = tl.program_id(0)
        d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = d < D

        state = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t in range(T):
            idx = b * stride_b + t * stride_t + d
            p   = tl.load(p_ptr   + idx, mask=mask, other=0.0)
            x   = tl.load(x_ptr   + idx, mask=mask, other=0.0)
            state = (1.0 - p) * state + p * x
            tl.store(out_ptr + idx, state, mask=mask)

    def mipt_scan(
        p: torch.Tensor,
        state: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear recurrent scan: state_t = (1-p_t)*state_{t-1} + p_t*x_t

        Args:
            p, state, x: (B, T, D) float32 on CUDA
        Returns:
            out: (B, T, D)
        """
        B, T, D = p.shape
        out = torch.empty_like(x)
        BLOCK_D = min(64, D)
        grid = (B, (D + BLOCK_D - 1) // BLOCK_D)
        _mipt_scan_kernel[grid](
            p, state, x, out, B, T, D,
            p.stride(0), p.stride(1), p.stride(2),
            BLOCK_D=BLOCK_D,
        )
        return out


def causal_topk_mask(
    scores: torch.Tensor,
    k: int,
    T: int,
    win_size: int,
) -> torch.Tensor:
    """
    Build causal (local window + global top-K) boolean attention mask.

    Args:
        scores: (B, T) content scores
        k: number of global slots
        T: sequence length
        win_size: local window width
    Returns:
        mask: (B, T, T) bool
    """
    B = scores.size(0)
    device = scores.device
    mask = torch.zeros(B, T, T, dtype=torch.bool, device=device)

    # Local causal window
    row = torch.arange(T, device=device)
    col = torch.arange(T, device=device)
    local = (col.unsqueeze(0) >= (row.unsqueeze(1) - win_size)) & (col.unsqueeze(0) < row.unsqueeze(1))
    mask = mask | local.unsqueeze(0)

    # Global top-K
    for i in range(1, T):
        k_ = min(k, i)
        topk = scores[:, :i].topk(k_, dim=-1).indices
        mask[:, i].scatter_(1, topk, True)

    return mask
