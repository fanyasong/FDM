#!/bin/bash
# Run all genomic benchmarks

CKPT=${1:-checkpoints/fdm_hg38_step54k.pt}

python genomic_multiseed.py \
  --checkpoint $CKPT \
  --seeds 42 123 456 \
  --tasks human_nontata_promoters human_enhancers_cohn \
          human_enhancers_ensembl human_ocr_ensembl
