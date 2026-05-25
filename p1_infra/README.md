# p1_infra — Shared Infrastructure (Shayan)

Reusable modules for Steps 4, 5, 7, 8.

## Usage

```python
import sys
sys.path.insert(0, "/content/project/p1_infra")

from cityscapes_loader import CityscapesValDataset
from class_mapping     import coco_pred_to_cityscapes, COCO_MODEL_IDX_TO_CITY
from miou_eval         import new_conf, update_conf, miou, print_report, CLASSES, FAIR_CLASSES
from eomt_loader       import (
    load_eomt, build_eomt,
    windowed_semantic_inference,
    to_per_pixel_logits,
)
```

## Step 4 baseline results (validated)

| Model | mIoU (19 cls) | mIoU (17 cls fair) |
|---|---|---|
| EoMT-COCO (zero-shot, mapped) | 55.00 | 61.47 |
| EoMT-Cityscapes               | 81.45 | 82.78 |

## Notes for P2 (Step 5) and P4 (Step 8)

- Images must be passed in `[0, 255]` range — `windowed_semantic_inference` handles `/255` internally
- For semantic eval: use `windowed_semantic_inference()` (returns `[C,H,W]` logits at native resolution)
- For anomaly eval (P4): use `to_per_pixel_logits()` to get per-pixel C-channel logits, then apply MSP/MaxLogit/MaxEntropy/RbA on top
- The `network.` prefix in checkpoints is stripped automatically by `load_eomt()`
