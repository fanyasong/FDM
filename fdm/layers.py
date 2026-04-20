"""
EntropyCollapseBlock (ECB) — core layer of FDM_Entropy_LM.

Each ECB implements:
  1. Per-token surprisal computation s_t = -log P(x_t | x_{<t})
  2. Surprise-conditioned gating parameter p_t
  3. Wave channel: Givens-rotation recurrent scan (norm-preserving)
  4. Cache channel: sparse attention over W local + K global slots

The gating formula (Eq. 3 in paper):
  p_t = [sigma(W_p(x_t + f(s_t)) + mu) * 0.5 * p_scale
        + sigma(10 * s_t) * 0.5 * p_scale
        + 1e-4]  clamped to [1e-4, 0.99]

Cross-layer normalisation for regulatory analysis:
  p_norm_t^l = p_t^l / mean_l(p_t^l)
This removes GC-content confound shared across all layers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class EntropyCollapseBlock(nn.Module):
    """
    Single layer of FDM_Entropy_LM.

    Args:
        d_model (int): Hidden dimension.
        p_scale (float): Maximum p_t value for this layer.
            Layer 1 (lw=8bp):   p_scale=0.99
            Layer 8 (lw=512bp): p_scale=0.004
        win_size (int): Local attention window width W.
        top_k (int): Number of global attention slots K.
        vocab_size (int): Vocabulary size for per-layer LM head.
    """

    def __init__(
        self,
        d_model:    int,
        p_scale:    float,
        win_size:   int,
        top_k:      int,
        vocab_size: int = 6,
    ):
        super().__init__()
        self.d_model   = d_model
        self.p_scale   = p_scale
        self.win_size  = win_size
        self.top_k     = top_k

        # Normalisation
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Wave channel
        self.W_r          = nn.Linear(d_model, d_model)
        self.W_rot        = nn.Linear(d_model, d_model)
        self.delta_gate   = nn.Parameter(torch.zeros(d_model))

        # Entropy-conditioned p_t gate
        self.surprise_proj = nn.Linear(1, d_model)
        self.W_p           = nn.Linear(d_model, 1)
        self.mu            = nn.Parameter(torch.zeros(1))

        # Per-layer LM head for surprisal computation
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Cache channel (sparse attention)
        self.Wq         = nn.Linear(d_model, d_model)
        self.Wk         = nn.Linear(d_model, d_model)
        self.Wv         = nn.Linear(d_model, d_model)
        self.W_content  = nn.Linear(d_model, 1)   # content score for global slot selection
        self.Wg         = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Storage for regulatory analysis hooks
        self._last_p: torch.Tensor = None

    def _compute_surprisal(
        self,
        xn: torch.Tensor,
        tokens: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Compute per-token surprisal s_t = -log P(x_t | x_{<t}).

        Args:
            xn: normalised hidden states (B, T, D)
            tokens: input token ids (B, T)
        Returns:
            sn: surprisal, mean-subtracted, shape (B, T, 1)
        """
        B, T, D = xn.shape
        with torch.no_grad():
            logits = self.lm_head(xn)                        # (B, T, V)
            log_p  = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
            tgt    = tokens[:, 1:].clamp(0, logits.size(-1) - 1)  # (B, T-1)
            s = torch.zeros(B, T, device=xn.device)
            s[:, 1:] = -log_p.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            sn = (s - s.mean(1, keepdim=True)).unsqueeze(-1)  # (B, T, 1)
        return sn

    def _compute_p(
        self,
        xn: torch.Tensor,
        sn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Entropy-conditioned gating parameter p_t (Eq. 3 in paper).

        Args:
            xn: normalised hidden states (B, T, D)
            sn: surprisal scores (B, T, 1)
        Returns:
            p: gating values in [1e-4, 0.99], shape (B, T, 1)
        """
        sf = self.surprise_proj(sn)                          # (B, T, D)
        p = (
            torch.sigmoid(self.W_p(xn + sf) + self.mu) * 0.5 * self.p_scale
            + torch.sigmoid(sn * 10)                         * 0.5 * self.p_scale
            + 1e-4
        )
        return p.clamp(1e-4, 0.99)

    def _wave_scan(
        self,
        h: torch.Tensor,
        p: torch.Tensor,
        xn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Wave channel: norm-preserving recurrent scan.
        Uses the Triton kernel when available; falls back to sequential.
        """
        try:
            from .ops import mipt_scan
            wave_in = torch.cat([self.W_r(xn), torch.zeros_like(xn)], dim=-1)
            wo = mipt_scan(
                p.expand_as(h),
                torch.zeros_like(h),
                wave_in,
            )[..., :self.d_model].nan_to_num(0.0)
        except ImportError:
            # Sequential fallback (slower but correct)
            wo = self._wave_scan_sequential(h, p, xn)

        return wo + torch.sigmoid(self.delta_gate) * (self.W_rot(wo) - wo) + self.delta_gate * h

    def _wave_scan_sequential(self, h, p, xn):
        """Sequential fallback for wave scan (no Triton required)."""
        B, T, D = h.shape
        state = torch.zeros(B, D, device=h.device, dtype=h.dtype)
        out = []
        x_proj = self.W_r(xn)
        for t in range(T):
            pt = p[:, t, 0:1]  # (B, 1)
            state = (1 - pt) * state + pt * x_proj[:, t, :]
            out.append(state)
        return torch.stack(out, dim=1)

    def _cache_attention(
        self,
        hw: torch.Tensor,
        xn: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cache channel: sparse local + global attention.

        Local window W: attends to previous W positions.
        Global K slots: selected by content score s_eff, reusing W_content.
        """
        if self.top_k <= 0:
            return hw

        B, T, D = hw.shape
        q = self.Wq(self.norm2(hw))
        k = self.Wk(xn)
        v = self.Wv(xn)
        q, k = apply_rope(q, k, rope_cos, rope_sin)

        # Build causal local+global mask
        content_scores = self.W_content(xn).squeeze(-1)  # (B, T)
        mask = self._build_sparse_mask(content_scores, T, hw.device)

        # Masked attention
        scale = math.sqrt(D)
        attn = torch.bmm(q, k.transpose(1, 2)) / scale    # (B, T, T)
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = torch.where(
            mask.any(-1, keepdim=True),
            F.softmax(attn, dim=-1).nan_to_num(0.0),
            torch.zeros_like(attn),
        )
        return hw + torch.sigmoid(self.Wg(hw)) * torch.bmm(attn, v)

    def _build_sparse_mask(
        self,
        content_scores: torch.Tensor,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Causal mask = local window union top-K global slots."""
        B = content_scores.size(0)
        mask = torch.zeros(B, T, T, dtype=torch.bool, device=device)

        # Local window
        for i in range(T):
            lo = max(0, i - self.win_size)
            mask[:, i, lo:i] = True

        # Global top-K
        for i in range(T):
            if i > 0:
                scores_so_far = content_scores[:, :i]          # (B, i)
                k = min(self.top_k, i)
                topk_idx = scores_so_far.topk(k, dim=-1).indices  # (B, k)
                mask[:, i].scatter_(1, topk_idx, True)

        return mask

    def forward(
        self,
        h: torch.Tensor,
        tokens: torch.LongTensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: hidden states (B, T, D)
            tokens: input ids (B, T)
            rope_cos, rope_sin: RoPE buffers from parent model
        Returns:
            h: updated hidden states (B, T, D)
        """
        xn = self.norm1(h)

        # Surprisal and p_t
        sn = self._compute_surprisal(xn, tokens)
        p  = self._compute_p(xn, sn)

        # Store for regulatory analysis hooks
        self._last_p = p.detach().squeeze(-1)  # (B, T)

        # Wave channel
        hw = self._wave_scan(h, p, xn)

        # Cache channel
        hw = self._cache_attention(hw, xn, rope_cos, rope_sin)

        # FFN
        return hw + self.ffn(self.norm3(hw))
