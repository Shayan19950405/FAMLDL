"""COCO→Cityscapes class mapping — sourced from EoMT's CLASS_MAPPING."""
import sys, torch
sys.path.insert(0, "/content/project/eomt")
from datasets.coco_panoptic import CLASS_MAPPING as COCO_CLASS_MAPPING

IDX_TO_COCO_ID = {v: k for k, v in COCO_CLASS_MAPPING.items()}

COCO_MODEL_IDX_TO_CITY = {
    0: 11, 1: 18, 2: 13, 3: 17, 5: 15, 6: 16, 7: 14, 9: 6, 11: 7,
    82: 2, 91: 2, 100: 0, 109: 3, 110: 3, 111: 3, 112: 3,
    116: 8, 117: 4, 119: 10, 123: 1, 125: 9, 129: 2, 131: 3,
}

_mapping = torch.full((133,), 255, dtype=torch.long)
for mi, ci in COCO_MODEL_IDX_TO_CITY.items():
    _mapping[mi] = ci

def coco_pred_to_cityscapes(pred):
    return _mapping.to(pred.device)[pred]
