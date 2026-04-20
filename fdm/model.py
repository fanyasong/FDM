"""
FDM: Fan Duality Model - Core Architecture
==========================================
FDM_Entropy_LM: entropy-conditioned variant used for all regulatory analyses.

Architecture:
  - 8x EntropyCollapseBlock (ECB) layers
  - Layer-specific p_scale: [0.99, 0.50, 0.25, 0.125, 0.063, 0.031, 0.016, 0.004]
  - Receptive fields: [8, 8, 16, 32, 64, 128, 256, 512] bp
  - 29.5M parameters total

Reference: arXiv:2604.07716
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EntropyCollapseBlock
from .rope import get_rope_freqs


class FDM_Entropy_LM(nn.Module):
    """
    Entropy-architecture FDM pretrained on hg38.

    This is the model used for all regulatory analyses in the paper:
      - cCRE enrichment (Table 3)
      - HiDRA correlation r=0.133 (Table 4)
      - MPRA correlation (Table 5)
      - Frozen-backbone ATAC (Table 6)

    Args:
        vocab_size (int): DNA vocabulary size. Default 6 = {A, T, G, C, N, PAD}.
        d_model (int): Hidden dimension. Default 512.
        n_layers (int): Number of ECB layers. Default 8.
        dropout (float): Dropout rate. Default 0.1.
    """

    # Layer-specific p_scale values (geometric decrease with depth)
    P_SCALES  = [0.99, 0.50, 0.25, 0.125, 0.063, 0.031, 0.016, 0.004]
    # Attention window widths (local receptive fields)
    WIN_SIZES = [8,    8,    16,   32,    64,    128,   256,   512  ]
    # Attention top-K global slots
    K_VALS    = [128,  64,   32,   16,    8,     4,     4,     4    ]

    def __init__(
        self,
        vocab_size: int = 6,
        d_model: int = 512,
        n_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert n_layers == 8, "P_SCALES/WIN_SIZES defined for 8 layers"

        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop  = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EntropyCollapseBlock(
                d_model    = d_model,
                p_scale    = self.P_SCALES[i],
                win_size   = self.WIN_SIZES[i],
                top_k      = self.K_VALS[i],
                vocab_size = vocab_size,
            )
            for i in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        cos, sin = get_rope_freqs(d_model, max_seq_len=1_048_576)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, T) integer token ids in [0, vocab_size)
        Returns:
            logits: (B, T, vocab_size)
        """
        h = self.drop(self.embed(tokens))
        for layer in self.layers:
            h = layer(h, tokens, self.rope_cos, self.rope_sin)
        return self.lm_head(self.norm_out(h))

    @torch.no_grad()
    def get_surprisal_scores(
        self,
        tokens: torch.LongTensor,
        normalize_layers: bool = True,
    ) -> torch.Tensor:
        """
        Compute per-position, per-layer surprisal scores for regulatory analysis.

        This is the core function used in Tables 3-6.

        Args:
            tokens: (B, T) integer token ids
            normalize_layers: if True, apply cross-layer normalisation
                              (eliminates GC-content confound).
        Returns:
            scores: (B, T, n_layers) float32.
                    scores[:, :, 7] = L8 channel (512 bp receptive field)
                    used for HiDRA (r=0.133) and MPRA correlations.
        """
        p_per_layer = []

        def make_hook(li):
            def hook(module, args, output):
                if hasattr(module, "_last_p") and module._last_p is not None:
                    p_per_layer.append(module._last_p.detach())
            return hook

        hooks = [layer.register_forward_hook(make_hook(i))
                 for i, layer in enumerate(self.layers)]
        try:
            self.forward(tokens)
        finally:
            for h in hooks:
                h.remove()

        if not p_per_layer:
            raise RuntimeError("No p_t collected. Ensure model is FDM_Entropy_LM.")

        scores = torch.stack(p_per_layer, dim=-1)  # (B, T, L)

        if normalize_layers:
            scores = scores / scores.mean(dim=-1, keepdim=True).clamp(min=1e-8)

        return scores
