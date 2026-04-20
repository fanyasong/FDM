#!/usr/bin/env bash
# =============================================================================
# reproduce_all.sh  —  FDM key results reproduction
# arXiv: 2604.07716
#
# Reproduces the three headline numbers from the paper:
#   1. HiDRA correlation  r = 0.133  (p = 1.34e-05)
#   2. chr22 cCRE enrichment  ~4.81×
#   3. MPRA GC-stratified r  ~0.096  (HepG2)
#
# Prerequisites
# -------------
#   pip install -r requirements.txt
#   # Download data (see docs/REPRODUCIBILITY.md) or set paths below
#
# Usage
# -----
#   bash scripts/reproduce_all.sh
#   bash scripts/reproduce_all.sh --ckpt /custom/path/best.pt
# =============================================================================

set -euo pipefail

# ─── Paths (override via env vars or args) ───────────────────────────────────
CKPT="${CKPT:-./checkpoints/entropy_pretrain/best.pt}"
DATA="${DATA:-./data}"
RESULTS="${RESULTS:-./results}"

# Parse optional --ckpt flag
while [[ $# -gt 0 ]]; do
  case $1 in
    --ckpt) CKPT="$2"; shift 2;;
    --data) DATA="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

GENOME="${DATA}/hg38_full"
CCRE_BB="${DATA}/encodeCcreCombined.bb"
HIDRA_BW="${DATA}/GSE104001_HiDRA_RNAtoDNARatio_0.1Pseudocount.bw"
MPRA_SEQS="${DATA}/mpra/table_s3.xlsx"
MPRA_ACT="${DATA}/mpra/Table_S6.xlsx"

mkdir -p "${RESULTS}"

echo "============================================================"
echo "FDM Reproduction Script"
echo "Checkpoint: ${CKPT}"
echo "============================================================"

# ─── 1. Whole-chromosome scan ─────────────────────────────────────────────────
echo ""
echo "[1/3] Scanning chr22 (takes ~5 min on RTX 5090)…"

SCAN_OUT="${RESULTS}/chr22_surprisal.npy"
if [[ ! -f "${SCAN_OUT}" ]]; then
  python -m analysis.scan_genome \
    --fasta   "${GENOME}/chr22.fa.gz" \
    --ckpt    "${CKPT}" \
    --out     "${SCAN_OUT}" \
    --chrom   chr22 \
    --stride  512 \
    --batch   64
else
  echo "  Found existing ${SCAN_OUT}, skipping scan."
fi

# ─── 2. HiDRA correlation ─────────────────────────────────────────────────────
echo ""
echo "[2/3] HiDRA correlation (expected: L8 r=0.133 p=1.34e-05)…"

python -m analysis.compare_hidra \
  --surprisal    "${SCAN_OUT}" \
  --hidra_bw     "${HIDRA_BW}" \
  --chrom        chr22 \
  --region_start 10500000 \
  --region_end   20000000

# ─── 3. cCRE enrichment ───────────────────────────────────────────────────────
echo ""
echo "[3/3] cCRE enrichment (expected: ~4.81×, GC-controlled ~1.85×)…"

python -m analysis.ccre_enrichment \
  --surprisal    "${SCAN_OUT}" \
  --ccre_bb      "${CCRE_BB}" \
  --fasta        "${GENOME}/chr22.fa.gz" \
  --chrom        chr22 \
  --layer        7

# ─── 4. MPRA correlation (optional, requires openpyxl) ───────────────────────
echo ""
echo "[4/3] MPRA HepG2 correlation (expected: GC-stratified r~0.096)…"

python -m analysis.mpra_analysis \
  --ckpt        "${CKPT}" \
  --seqs        "${MPRA_SEQS}" \
  --activity    "${MPRA_ACT}" \
  --cell_type   HepG2 \
  --random_ctrl

echo ""
echo "============================================================"
echo "Done. Results saved to ${RESULTS}/"
echo "============================================================"
