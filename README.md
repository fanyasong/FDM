# FDM: Fan Duality Model

**A Wave-Cache Architecture with Phase-Aware Training for Language and Genomic Sequence Modeling**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Chinese Patent No. 2026104740169

---

## Overview

FDM separates sequence processing into two dedicated channels:

- **Wave channel** — norm-preserving recurrent propagation via Givens-rotation phase operators; compresses long-range history into a fixed-dimensional complex state
- **Cache channel** — sparse associative retrieval over W local + K global slots, independent of sequence length N; maintains a fixed-size memory regardless of context length

A central finding is that jointly training these two channels produces a **gradient sink**: scan parameters dominate gradient flow, leaving the cache severely undertrained. **Phase-aware staged training** resolves this by first freezing the wave channel to allow cache specialisation, then resuming joint optimisation after an automatic gradient-ratio criterion is met.

---

## Results

### Language Modeling — WikiText-103

| Model | Val PPL |
|-------|---------|
| Transformer (reference, ~137M) | ~36–38 |
| FDM — joint training | 487.5 |
| FDM — Freeze-Scan (Phase 0 only) | 64.9 |
| **FDM — phase-aware (full)** | **33.75** |

All FDM variants share the same architecture; differences reflect optimisation strategy.

### Genomic Benchmarks — AUC (mean ± std, 3 seeds)

| Task | Seq. len | FDM-HG38 | HyenaDNA | Δ |
|------|----------|----------|----------|---|
| human_nontata_promoters | 251 bp | **0.945 ± 0.001** | 0.856 | +0.089 |
| human_enhancers_ensembl | 479 bp | **0.905 ± 0.012** | 0.706 | +0.199 |
| human_enhancers_cohn    | 500 bp | **0.775 ± 0.002** | 0.711 | +0.064 |
| human_ocr_ensembl       | 330 bp | **0.811 ± 0.009** | 0.806 | +0.005 |

### Variant Effect Prediction — ClinVar chr17, 5-fold CV

| Method | AUC |
|--------|-----|
| FDM zero-shot | 0.51 |
| CADD (reference) | 0.869 |
| **FDM fine-tuned** | **0.861 ± 0.011** |

### Cross-Species Transfer — Drosophila Enhancers

| Model | AUC |
|-------|-----|
| CNN (from scratch) | 0.736 |
| **FDM-HG38 (human pretrained)** | **0.792** |

---

## Repository Structure

```
FDM/
├── README.md
├── LICENSE
├── requirements.txt
├── train_130m.py           # FDM language model + phase-aware training
├── genomics_hg38.py        # Genomic model + hg38 pretraining
├── triton_scan_v2.py       # Triton kernel for Givens-rotation scan
├── genomic_multiseed.py    # Multi-seed genomic benchmark evaluation
├── vep_finetune.py         # ClinVar variant effect prediction fine-tuning
├── scripts/
│   ├── pretrain_lm.sh
│   ├── pretrain_genomic.sh
│   └── eval_benchmarks.sh
├── figures/
│   ├── generate_figures.py
│   └── fig1–fig4 + supp_memory (PDF)
├── paper/
│   ├── main.tex
│   └── cover_letter.pdf
└── checkpoints/
    └── README.md
```

---

## Installation

```bash
git clone https://github.com/YasongFan/FDM.git
cd FDM
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.1+, Triton 3.0+, CUDA 12+

---

## Usage

### Phase-Aware Language Model Training

```bash
# Phase 0: freeze wave channel, train cache only
python train_130m.py --phase 0 --steps 23000 --lr 1e-4 \
  --data data/tokens_130m.pt --save checkpoints/fdm_phase0.pt

# Phase 1: joint optimisation
python train_130m.py --phase 1 --steps 40000 --lr 3e-5 \
  --resume checkpoints/fdm_phase0.pt \
  --save checkpoints/fdm_phase_aware.pt
```

### Genomic Pretraining

```bash
python genomics_hg38.py \
  --genome_dir data/hg38_full/ --steps 95000 \
  --d 512 --n_layers 8 --cache_k 32 --local_window 128 \
  --save checkpoints/fdm_hg38.pt
```

### Benchmark Evaluation

```bash
python genomic_multiseed.py \
  --checkpoint checkpoints/fdm_hg38_step54k.pt \
  --seeds 42 123 456
```

### Reproduce Figures

```bash
cd figures/ && python generate_figures.py
```

---

## Pretrained Checkpoints

Hosted on HuggingFace: `https://huggingface.co/YasongFan/FDM` (available upon paper publication)

| Checkpoint | Val PPL |
|------------|---------|
| `fdm_phase_aware_lm.pt` | 33.75 (WikiText-103) |
| `fdm_hg38_step95k.pt` | 3.23 (hg38) |
| `fdm_hg38_step54k.pt` | 3.37 (used in benchmark experiments) |

See `checkpoints/README.md` for loading instructions.

---

## Citation

Paper under review. If you use this code, please cite:

```bibtex
@article{fan2026fdm,
  title={A Wave-Cache Architecture with Phase-Aware Training
         for Language and Genomic Sequence Modeling},
  author={Fan, Yasong},
  year={2026},
  note={Preprint. Chinese Patent No. 2026104740169}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
Chinese Patent No. 2026104740169.
