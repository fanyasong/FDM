"""
Genomic Benchmark Fine-tuning
arXiv: 2604.07716  —  Table 1

Fine-tunes FDM_Entropy_LM on genomic_benchmarks tasks.

Usage
-----
python -m genomics.finetune \
    --task    human_nontata_promoters \
    --ckpt    /path/to/best.pt \
    --out_dir ./checkpoints/finetune/promoter \
    --seeds   42 123 456
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from fdm.model import FDM_Entropy_LM, load_pretrained

DNA_VOCAB = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4,
             'a': 0, 't': 1, 'g': 2, 'c': 3, 'n': 4}


# ─── Task registry ────────────────────────────────────────────────────────────

TASKS = {
    'human_nontata_promoters': {'seq_len': 251,  'n_classes': 2},
    'human_enhancers_ensembl': {'seq_len': 479,  'n_classes': 2},
    'human_enhancers_cohn':    {'seq_len': 500,  'n_classes': 2},
    'human_ocr_ensembl':       {'seq_len': 330,  'n_classes': 2},
}


def encode_seq(seq: str, max_len: int) -> list:
    ids = [DNA_VOCAB.get(b, 4) for b in seq[:max_len]]
    ids += [4] * (max_len - len(ids))
    return ids


# ─── Classifier Head ──────────────────────────────────────────────────────────

class FDMClassifier(nn.Module):
    def __init__(self, backbone: FDM_Entropy_LM, n_classes: int = 2, d: int = 512):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, n_classes),
        )
        # Freeze first 4 layers and embedding
        for param in self.backbone.embed.parameters():
            param.requires_grad = False
        for layer in self.backbone.layers[:4]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone.drop(self.backbone.embed(x))
        for layer in self.backbone.layers:
            h = layer(h, x, self.backbone.rope_cos, self.backbone.rope_sin)
        h = self.backbone.norm_out(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


# ─── Training & Evaluation ───────────────────────────────────────────────────

def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    from sklearn.metrics import roc_auc_score
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return float(roc_auc_score(labels, probs))


def run_seed(args, seed: int) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = args.device

    # Load task data via genomic_benchmarks
    try:
        from genomic_benchmarks.dataset_getters.pytorch_datasets import \
            get_dataset as get_gb_dataset
    except ImportError:
        raise ImportError(
            "Install genomic_benchmarks: "
            "pip install genomic-benchmarks"
        )

    task_cfg = TASKS[args.task]
    seq_len  = task_cfg['seq_len']

    train_ds, test_ds = (
        get_gb_dataset(args.task, split, version=0)
        for split in ('train', 'test')
    )

    def collate(batch):
        xs = torch.tensor(
            [encode_seq(s, seq_len) for s, _ in batch], dtype=torch.long
        )
        ys = torch.tensor([int(y) for _, y in batch], dtype=torch.long)
        return xs, ys

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        collate_fn=collate, num_workers=4,
    )
    test_loader  = torch.utils.data.DataLoader(
        test_ds,  batch_size=args.batch, shuffle=False,
        collate_fn=collate, num_workers=4,
    )

    # Model
    backbone = load_pretrained(args.ckpt, device=device)
    model    = FDMClassifier(backbone, n_classes=2).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable params: {n_trainable/1e6:.2f}M")

    opt = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    sched     = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    patience  = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, criterion, device)
        auc  = evaluate(model, test_loader, device)
        sched.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch={epoch:3d}  loss={loss:.4f}  AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    return best_auc


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='FDM genomic benchmark fine-tuning')
    ap.add_argument('--task',    required=True, choices=list(TASKS))
    ap.add_argument('--ckpt',    required=True)
    ap.add_argument('--out_dir', default='./checkpoints/finetune')
    ap.add_argument('--seeds',   type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--epochs',  type=int,   default=30)
    ap.add_argument('--batch',   type=int,   default=32)
    ap.add_argument('--lr',      type=float, default=5e-5)
    ap.add_argument('--patience',type=int,   default=6)
    ap.add_argument('--device',  default='cuda')
    args = ap.parse_args()

    print(f"\nTask: {args.task}")
    aucs = []
    for seed in args.seeds:
        print(f"  seed={seed}")
        auc = run_seed(args, seed)
        aucs.append(auc)
        print(f"  → AUC = {auc:.4f}")

    print(f"\nFinal:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
          f"(seeds={args.seeds})")
    print(f"Paper:  {TASKS[args.task].get('paper_auc', 'N/A')}")


if __name__ == '__main__':
    main()
