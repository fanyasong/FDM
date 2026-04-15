#!/bin/bash
# Phase-aware language model pretraining

python train_130m.py --phase 0 --steps 23000 --lr 1e-4 \
  --data data/tokens_130m.pt --save checkpoints/fdm_phase0.pt

python train_130m.py --phase 1 --steps 40000 --lr 3e-5 \
  --resume checkpoints/fdm_phase0.pt --save checkpoints/fdm_phase_aware.pt
