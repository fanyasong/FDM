# FDM: Fan Duality Model

**A Wave-Cache Architecture for Genomic Sequence Modeling with Unsupervised Regulatory Signal Extraction**

[![arXiv](https://img.shields.io/badge/arXiv-2604.07716-b31b1b.svg)](https://arxiv.org/abs/2604.07716)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Yasong Fan · Independent Researcher

---

## Overview

FDM separates global state propagation (wave channel) from sparse local retrieval (cache channel),
achieving fixed O(1) inference memory independent of sequence length N.
This enables whole-chromosome genomic scanning on a single GPU.

**Key results:**

| Experiment | Result |
|---|---|
| WikiText-103 PPL | 33.75 (Transformer ref ~36–38) |
| Memory at N=65536 | 397 MB (Transformer: 1812 MB) |
| Promoter AUC | 0.945 ± 0.001 (HyenaDNA: 0.856) |
| chr22 cCRE enrichment | 4.81× (GC-controlled: 1.85×, p=8.39e-148) |
| mm10 zero-shot | 5.80× (no mouse training) |
| HiDRA r (L8) | 0.133, p=1.34e-05 (entropy baseline: 0.014) |
| MPRA HepG2 GC-strat. r | 0.096 (n=61,463); K562: 0.110 (n=2,123) |

---

## Repository Structure

```
FDM/
├── fdm/
│   ├── __init__.py
│   ├── model.py            # FDM_Entropy_LM
│   ├── layers.py           # EntropyCollapseBlock
│   ├── ops.py              # Triton scan kernel
│   └── rope.py             # RoPE
├── scripts/
│   ├── pretrain_entropy.py
│   ├── pretrain_multitask.py
│   └── finetune_genomic.py
├── analysis/
│   ├── compare_hidra.py    # Reproduce r=0.133
│   ├── ccre_enrichment.py  # Reproduce cCRE enrichment
│   ├── mpra_correlation.py # Reproduce MPRA r
│   └── frozen_backbone.py  # Frozen backbone ATAC
├── configs/
│   ├── entropy_pretrain.yaml
│   └── lm_wikitext.yaml
├── generate_figs.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YasongFan/FDM.git
cd FDM
pip install -r requirements.txt
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.1, CUDA ≥ 11.8, Triton ≥ 2.0

---

## Pretrained Weights

| Model | Params | Val PPL | Link |
|---|---|---|---|
| FDM-Entropy-hg38 | 29.5M | 3.88 (chr17) | HuggingFace (coming soon) |
| FDM-Multitask-hg38 | 29.5M | 3.13 (chr17) | HuggingFace (coming soon) |

```python
from fdm import FDM_Entropy_LM
import torch

model = FDM_Entropy_LM()
ckpt = torch.load("checkpoints/entropy_pretrain/best.pt", weights_only=False)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()
```

---

## Reproducing Key Results

### r = 0.133 (HiDRA, Table 4)

```bash
# Data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104001
python analysis/compare_hidra.py \
    --checkpoint checkpoints/entropy_pretrain/best.pt \
    --hidra_bw data/GSE104001_HiDRA_RNAtoDNARatio_0.1Pseudocount.bw \
    --region chr22:10500000-20000000
# Expected: L8 r=0.133  p=1.34e-05  n=1137
```

### cCRE Enrichment (Table 3)

```bash
# ENCODE cCRE: https://www.encodeproject.org/annotations/ENCSR890HXX/
python analysis/ccre_enrichment.py \
    --checkpoint checkpoints/entropy_pretrain/best.pt \
    --genome data/hg38/ \
    --ccre data/encodeCcreCombined.bb \
    --chroms chr1 chr17 chr22
# Expected: chr22 4.81x; GC-controlled 1.85x p=8.39e-148
```

### MPRA Correlation (Table 5)

```bash
# Data: https://zenodo.org/records/10558183
python analysis/mpra_correlation.py \
    --checkpoint checkpoints/entropy_pretrain/best.pt \
    --seq_table data/mpra/table_s3.xlsx \
    --act_table data/mpra/table_s6.xlsx \
    --cell_type HepG2
# Expected: raw r=0.176  GC-stratified r=0.096  n=61463
```

### Frozen Backbone ATAC (Table 6)

```bash
# Data: https://www.encodeproject.org/files/ENCFF038DDS/
python analysis/frozen_backbone.py \
    --backbone checkpoints/entropy_pretrain/best.pt \
    --atac_peaks data/atac/ENCFF038DDS.bed.gz \
    --train_chrom chr22 --test_chrom chr17
# Expected: AUC=0.727; p_t range L8=0.72 (preserved)
```

---

## Training From Scratch

### Entropy Architecture Pretraining

```bash
# hg38: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/
python scripts/pretrain_entropy.py \
    --genome_dir data/hg38/ \
    --val_chrom chr17 \
    --output_dir checkpoints/entropy_pretrain/ \
    --steps 100000 --batch_size 8 --seq_len 512 --lr 3e-4
# Hardware: 1x RTX 5090, ~12h  Expected PPL: ~3.88
```

### WikiText-103

```bash
python scripts/pretrain_lm.py \
    --data_dir data/wikitext103/ \
    --output_dir checkpoints/lm/ \
    --phase0_steps 23000 --phase1_steps 17000
# Expected PPL: 33.75
```

---

## Data Sources

| Dataset | Source | Usage |
|---|---|---|
| hg38 | UCSC | Pretraining |
| ENCODE cCRE | ENCSR890HXX | cCRE enrichment |
| GM12878 ATAC | ENCFF038DDS | ATAC prediction |
| HiDRA | GSE104001 | Enhancer activity (Wang et al. Nat. Commun. 2018) |
| lentiMPRA | Zenodo 10558183 | MPRA (Agarwal et al. Nature 2025) |
| PhyloP 100way | UCSC | Conservation |

---

## Citation

```bibtex
@article{fan2026fdm,
  title   = {A Wave-Cache Architecture for Genomic Sequence Modeling
             with Unsupervised Regulatory Signal Extraction},
  author  = {Fan, Yasong},
  journal = {arXiv preprint arXiv:2604.07716},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
