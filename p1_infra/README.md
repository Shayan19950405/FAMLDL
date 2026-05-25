# p1_infra — Shared Infrastructure (from Step 4, Shayan's work)

Reusable modules for Steps 4, 5, 7, 8.

```python
import sys
sys.path.insert(0, "/content/project/p1_infra")
from cityscapes_loader import CityscapesValDataset
from class_mapping     import coco_pred_to_cityscapes
from miou_eval         import new_conf, update_conf, print_report, CLASSES
from eomt_loader       import load_eomt, build_eomt, windowed_semantic_inference, to_per_pixel_logits
```

## Step 4 baseline results

| Model | mIoU (19 cls) | mIoU (17 cls fair) |
|---|---|---|
| EoMT-COCO (zero-shot, mapped) | 55.00 | 61.47 |
| EoMT-Cityscapes               | 81.45 | 82.78 |
