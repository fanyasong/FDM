# Reproducibility Guide

> **arXiv: 2604.07716**  
> All headline numbers in the paper can be reproduced with the steps below.

---

## Environment

```bash
git clone https://github.com/YasongFan/FDM.git
cd FDM
pip install -r requirements.txt
```

Tested on:
- Python 3.10 / 3.11
- PyTorch 2.1–2.3
- CUDA 12.1
- RTX 5090 (24 GB) or A100 (40 GB)

---

## Step 0: Download pretrained weights

Weights will be uploaded to Zenodo on paper acceptance.
In the meantime, train from scratch (Step 1) or contact the author.

| File | Steps | Val PPL |
|------|-------|---------|
| `checkpoints/entropy_pretrain/best.pt` | 100k | 3.88 |
| `checkpoints/multitask_v2/best.pt`     | 185k | 3.13 |
| `checkpoints/entropy_frozen_atac.pt`   | —    | —    |

---

## Step 1: Pretraining (optional — skip if using pretrained weights)

```bash
# Download hg38 first
bash scripts/download_data.sh --data_dir ./data

# Train entropy architecture (~12 h on RTX 5090)
python -m fdm.train \
    --genome_dir ./data/hg38_full \
    --output_dir ./checkpoints/entropy_pretrain \
    --steps      100000 \
    --seq_len    512 \
    --batch      8 \
    --lr         3e-4
# Expected val PPL ≈ 3.88
```

---

## Step 2: Reproduce HiDRA correlation r = 0.133

This is the key result (Table 3, Figure 6B).

```bash
# Scan chr22 (5–10 min on RTX 5090)
python -m analysis.scan_genome \
    --fasta  ./data/hg38_full/chr22.fa.gz \
    --ckpt   ./checkpoints/entropy_pretrain/best.pt \
    --out    ./results/chr22_surprisal.npy \
    --stride 512

# Correlate with HiDRA
python -m analysis.compare_hidra \
    --surprisal    ./results/chr22_surprisal.npy \
    --hidra_bw     ./data/GSE104001_HiDRA_RNAtoDNARatio_0.1Pseudocount.bw \
    --region_start 10500000 \
    --region_end   20000000
```

**Expected output:**
```
L8 (lw= 512 bp):  r=+0.133  p=1.34e-05  ← paper result
```

Alternatively, use the pre-computed array:
```bash
# If you have results/entropy_layer_relative.npy from the 6000A server:
python -m analysis.compare_hidra \
    --surprisal ./results/entropy_layer_relative.npy \
    --hidra_bw  ./data/GSE104001_HiDRA_RNAtoDNARatio_0.1Pseudocount.bw
```

---

## Step 3: Reproduce cCRE Enrichment (Table 2)

```bash
python -m analysis.ccre_enrichment \
    --surprisal ./results/chr22_surprisal.npy \
    --ccre_bb   ./data/encodeCcreCombined.bb \
    --fasta     ./data/hg38_full/chr22.fa.gz \
    --chrom     chr22
```

**Expected output:**
```
Layer L8   Fold enrichment (Q75/Q25): 4.81×
           GC-stratified AUC: 0.xxx
```

---

## Step 4: Reproduce MPRA Correlation (Table 4)

```bash
# HepG2  (n=61,463)
python -m analysis.mpra_analysis \
    --ckpt       ./checkpoints/entropy_pretrain/best.pt \
    --seqs       "./data/mpra/Table S3 - sequences and coordinates.xlsx" \
    --activity   "./data/mpra/Table S6 - folds and performance.xlsx" \
    --cell_type  HepG2 \
    --random_ctrl
```

**Expected output:**
```
[trained model]  n=61463
  Raw Spearman r        = 0.176
  GC-stratified r       = 0.096
  GC-corrected residual = 0.083  (p=2.39e-94)

[random model]
  GC-stratified r       = -0.049
```

---

## Step 5: Reproduce Frozen Backbone (Table 5)

```bash
python -m genomics.frozen_backbone \
    --ckpt        ./checkpoints/entropy_pretrain/best.pt \
    --atac_peaks  ./data/atac/ENCFF038DDS.bed.gz \
    --genome_fa   ./data/hg38_full/chr22.fa.gz \
    --out         ./checkpoints/frozen_atac.pt
# Expected: val AUC ≈ 0.727, p_t L1 range ≈ 0.72
```

---

## Step 6: Reproduce Genomic Benchmarks (Table 1)

```bash
pip install genomic-benchmarks scikit-learn

for TASK in human_nontata_promoters human_enhancers_ensembl \
            human_enhancers_cohn human_ocr_ensembl; do
    python -m genomics.finetune \
        --task    ${TASK} \
        --ckpt    ./checkpoints/entropy_pretrain/best.pt \
        --seeds   42 123 456
done
```

**Expected AUC (mean ± std, 3 seeds):**

| Task | FDM | HyenaDNA |
|------|-----|----------|
| human_nontata_promoters | 0.945 ± 0.001 | 0.856 |
| human_enhancers_ensembl | 0.905 ± 0.012 | 0.706 |
| human_enhancers_cohn    | 0.775 ± 0.002 | 0.711 |
| human_ocr_ensembl       | 0.811 ± 0.009 | 0.806 |

---

## One-Command Reproduction

```bash
# Sets CKPT, DATA, RESULTS env vars and runs steps 2–4
bash scripts/reproduce_all.sh \
    --ckpt ./checkpoints/entropy_pretrain/best.pt \
    --data ./data
```

---

## Hardware Requirements

| Analysis | GPU VRAM | Time (RTX 5090) |
|----------|----------|-----------------|
| chr22 scan | 9.3 GB | ~5 min |
| MPRA scan (61k seqs) | 9.3 GB | ~15 min |
| Pretraining (100k steps) | 9.3 GB | ~6 h |
| Fine-tuning (3 seeds) | 4 GB | ~30 min |

FDM memory saturates above moderate context lengths.
A Transformer would OOM on the chr22 whole-chromosome scan.

---

## Known Issues

- The sequential scan fallback in `model.py` is ~10× slower than the
  Triton kernel used during actual training. Results are numerically identical.
- `compare_hidra.py` uses hg19 coordinates for HiDRA (consistent with GSE104001);
  chr22 is the same length in hg19 and hg38 so no liftover is needed.
- MPRA Table S3/S6 column names may differ between Zenodo versions;
  the parser checks for the standard headers documented in Agarwal et al. 2025.
