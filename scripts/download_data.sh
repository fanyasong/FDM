#!/usr/bin/env bash
# =============================================================================
# download_data.sh  —  Download all datasets for FDM reproduction
# arXiv: 2604.07716
#
# Usage
# -----
#   bash scripts/download_data.sh --data_dir ./data
#
# Downloads
# ---------
#   hg38 reference genome (chr1–22, X)   ~3.1 GB
#   ENCODE cCRE annotations               ~50 MB
#   GM12878 ATAC-seq peaks                ~10 MB
#   HiDRA bigWig (GSE104001)              ~154 MB
#   lentiMPRA data (Zenodo 10558183)      ~60 MB
# =============================================================================

set -euo pipefail

DATA_DIR="${1:-./data}"
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir) DATA_DIR="$2"; shift 2;;
    *) shift;;
  esac
done

mkdir -p "${DATA_DIR}/hg38_full" \
         "${DATA_DIR}/mpra" \
         "${DATA_DIR}/atac"

echo "Downloading to: ${DATA_DIR}"

# ─── hg38 reference genome ───────────────────────────────────────────────────
echo ""
echo "[1/5] hg38 reference genome (chroms used in paper)…"
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7
        chr9 chr10 chr11 chr12 chr13 chr14 chr15
        chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)

for CHR in "${CHROMS[@]}"; do
  OUT="${DATA_DIR}/hg38_full/${CHR}.fa.gz"
  if [[ -f "${OUT}" ]]; then
    echo "  ${CHR}: already exists, skipping"
    continue
  fi
  echo "  downloading ${CHR}…"
  URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/${CHR}.fa.gz"
  curl -L -o "${OUT}" "${URL}" --silent --show-error || \
    wget -q -O "${OUT}" "${URL}"
done

# ─── ENCODE cCRE ──────────────────────────────────────────────────────────────
echo ""
echo "[2/5] ENCODE cCRE annotations (hg38)…"
OUT="${DATA_DIR}/encodeCcreCombined.bb"
if [[ ! -f "${OUT}" ]]; then
  URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/encodeCcreCombined.bb"
  curl -L -o "${OUT}" "${URL}" --silent --show-error || \
    wget -q -O "${OUT}" "${URL}"
  echo "  saved → ${OUT}"
else
  echo "  already exists, skipping"
fi

# ─── GM12878 ATAC-seq ────────────────────────────────────────────────────────
echo ""
echo "[3/5] GM12878 ATAC-seq peaks (ENCODE ENCFF038DDS)…"
OUT="${DATA_DIR}/atac/ENCFF038DDS.bed.gz"
if [[ ! -f "${OUT}" ]]; then
  URL="https://www.encodeproject.org/files/ENCFF038DDS/@@download/ENCFF038DDS.bed.gz"
  curl -L -o "${OUT}" "${URL}" --silent --show-error || \
    wget -q -O "${OUT}" "${URL}"
  echo "  saved → ${OUT}"
else
  echo "  already exists, skipping"
fi

# ─── HiDRA (GSE104001) ───────────────────────────────────────────────────────
echo ""
echo "[4/5] HiDRA bigWig (GSE104001, GM12878)…"
echo "  Manual download required (NCBI GEO access):"
echo "  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104001"
echo "  File: GSE104001_HiDRA_RNAtoDNARatio_0.1Pseudocount.bw"
echo "  Save to: ${DATA_DIR}/GSE104001_HiDRA_RNAtoDNARatio_0.1Pseudocount.bw"

# ─── lentiMPRA (Zenodo 10558183) ─────────────────────────────────────────────
echo ""
echo "[5/5] lentiMPRA data (Zenodo 10558183; Agarwal et al. Nature 2025)…"
ZENODO_BASE="https://zenodo.org/records/10558183/files"

for FNAME in "Table S3 - sequences and coordinates.xlsx" \
             "Table S6 - folds and performance.xlsx"; do
  ENCODED=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${FNAME}'))")
  OUT="${DATA_DIR}/mpra/${FNAME}"
  if [[ ! -f "${OUT}" ]]; then
    echo "  downloading '${FNAME}'…"
    curl -L -o "${OUT}" "${ZENODO_BASE}/${ENCODED}" --silent --show-error || \
      wget -q -O "${OUT}" "${ZENODO_BASE}/${ENCODED}"
    echo "  saved → ${OUT}"
  else
    echo "  '${FNAME}': already exists, skipping"
  fi
done

# ─── PhyloP (optional, large) ────────────────────────────────────────────────
echo ""
echo "[optional] PhyloP 100-way conservation (hg38, ~10 GB):"
echo "  wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw"
echo "  Save to: ${DATA_DIR}/hg38.phyloP100way.bw"

echo ""
echo "============================================================"
echo "Data download complete."
echo "Required for paper results:"
echo "  ${DATA_DIR}/hg38_full/chr22.fa.gz          (cCRE / HiDRA / MPRA)"
echo "  ${DATA_DIR}/encodeCcreCombined.bb           (cCRE enrichment)"
echo "  ${DATA_DIR}/GSE104001_HiDRA_*.bw            (HiDRA correlation)"
echo "  ${DATA_DIR}/mpra/Table S3 *.xlsx            (MPRA sequences)"
echo "  ${DATA_DIR}/mpra/Table S6 *.xlsx            (MPRA activity)"
echo "============================================================"
