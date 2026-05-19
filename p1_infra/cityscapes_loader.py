import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms as T

class CityscapesValDataset(Dataset):
    CLASSES = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic light", "traffic sign", "vegetation",
        "terrain", "sky", "person", "rider", "car", "truck",
        "bus", "train", "motorcycle", "bicycle"
    ]

    ID_TO_TRAINID = {
        0: 255, 1: 255, 2: 255, 3: 255, 4: 255,
        5: 255, 6: 255, 7: 0,   8: 1,   9: 255,
        10: 255, 11: 2, 12: 3,  13: 4,  14: 255,
        15: 255, 16: 255, 17: 5, 18: 255, 19: 6,
        20: 7,  21: 8,  22: 9,  23: 10, 24: 11,
        25: 12, 26: 13, 27: 14, 28: 15, 29: 255,
        30: 255, 31: 16, 32: 17, 33: 18, -1: 255
    }

    def __init__(self, root="/content/cityscapes", split="val", size=(512, 1024)):
        self.root  = Path(root)
        self.split = split
        self.size  = size

        self.img_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        img_dir = self.root / "leftImg8bit" / split
        gt_dir  = self.root / "gtFine"      / split

        self.samples = []
        for img_path in sorted(img_dir.rglob("*_leftImg8bit.png")):
            city = img_path.parent.name
            stem = img_path.stem.replace("_leftImg8bit", "")
            gt_path = gt_dir / city / f"{stem}_gtFine_labelIds.png"
            if gt_path.exists():
                self.samples.append((img_path, gt_path))

        print(f"CityscapesValDataset: {len(self.samples)} samples ({split})")

    def __len__(self):
        return len(self.samples)

    def _convert_labels(self, mask_np):
        out = np.full_like(mask_np, 255)
        for lid, tid in self.ID_TO_TRAINID.items():
            out[mask_np == lid] = tid
        return out

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        img  = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(gt_path), dtype=np.int32)
        img_tensor  = self.img_transform(img)
        mask_tensor = torch.from_numpy(
            self._convert_labels(mask)).long()
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=self.size, mode="nearest"
        ).squeeze().long()
        return img_tensor, mask_tensor, str(img_path)


def get_val_loader(root="/content/cityscapes", batch_size=4, num_workers=2):
    dataset = CityscapesValDataset(root=root, split="val")
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers)
