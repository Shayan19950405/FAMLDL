"""Cityscapes val DataLoader — native 1024x2048, raw [0,255] images.
EoMT performs ImageNet normalization internally; we must NOT normalize here.
Windowed inference handles the resize, so we keep native resolution.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np

class CityscapesValDataset(Dataset):
    CLASSES = [
        "road","sidewalk","building","wall","fence","pole",
        "traffic light","traffic sign","vegetation","terrain","sky",
        "person","rider","car","truck","bus","train","motorcycle","bicycle"
    ]
    ID_TO_TRAINID = {
        0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255,
        7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255,
        15:255, 16:255, 17:5, 18:255, 19:6, 20:7, 21:8, 22:9,
        23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255,
        30:255, 31:16, 32:17, 33:18, -1:255,
    }

    def __init__(self, root="/content/cityscapes", split="val"):
        self.root = Path(root)
        img_dir = self.root / "leftImg8bit" / split
        gt_dir  = self.root / "gtFine" / split
        self.samples = []
        for ip in sorted(img_dir.rglob("*_leftImg8bit.png")):
            city = ip.parent.name
            stem = ip.stem.replace("_leftImg8bit","")
            gp = gt_dir / city / f"{stem}_gtFine_labelIds.png"
            if gp.exists(): self.samples.append((ip, gp))
        print(f"CityscapesValDataset: {len(self.samples)} samples ({split})")

    def __len__(self): return len(self.samples)

    def _to_trainids(self, m):
        out = np.full_like(m, 255)
        for lid, tid in self.ID_TO_TRAINID.items():
            out[m == lid] = tid
        return out

    def __getitem__(self, idx):
        ip, gp = self.samples[idx]
        # Image stays in [0, 255] uint8-as-float — EoMT normalizes internally
        img  = np.array(Image.open(ip).convert("RGB"))
        mask = np.array(Image.open(gp), dtype=np.int32)
        img_t  = torch.from_numpy(img).permute(2,0,1).float()       # [3,1024,2048] in [0,255]
        mask_t = torch.from_numpy(self._to_trainids(mask)).long()   # [1024,2048]
        return img_t, mask_t, str(ip)
