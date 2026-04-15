# Pretrained Checkpoints

## Available

| Checkpoint | Description | Val PPL |
|------------|-------------|---------|
| `fdm_phase_aware_lm.pt` | Language model, phase-aware, WikiText-103 | 33.75 |
| `fdm_hg38_step95k.pt` | Genomic model, hg38, 95K steps | 3.23 |
| `fdm_hg38_step54k.pt` | Genomic model used in benchmark experiments | 3.37 |

Hosted at: https://huggingface.co/YasongFan/FDM *(coming soon)*

## Loading

```python
import torch
from genomics_hg38 import FDM_HG38

ckpt = torch.load('fdm_hg38_step54k.pt', map_location='cpu')
model = FDM_HG38(vocab_size=6, d=512, n_layers=8,
                  cache_k=32, local_window=128, max_len=4096)
model.load_state_dict(ckpt['model_state'])
model.eval()
```
