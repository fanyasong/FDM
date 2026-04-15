#!/bin/bash
# Genomic pretraining on hg38

python genomics_hg38.py \
  --genome_dir data/hg38_full/ \
  --steps 95000 --lr 1e-4 --batch 8 --seq_len 512 \
  --d 512 --n_layers 8 --cache_k 32 --local_window 128 \
  --save checkpoints/fdm_hg38.pt
