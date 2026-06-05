"""mIoU pipeline — 19-class standard + 17-class fair (excludes pole/rider)."""
import torch

NUM_CLASSES = 19
IGNORE = 255
FAIR_EXCLUDE = [5, 12]
FAIR_CLASSES = [c for c in range(NUM_CLASSES) if c not in FAIR_EXCLUDE]
CLASSES = ["road","sidewalk","building","wall","fence","pole",
           "traffic light","traffic sign","vegetation","terrain","sky",
           "person","rider","car","truck","bus","train","motorcycle","bicycle"]

def new_conf(device="cuda"):
    return torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long, device=device)

def update_conf(conf, pred, gt):
    pred = pred.flatten(); gt = gt.flatten()
    valid = (gt != IGNORE) & (pred != IGNORE)
    p, g = pred[valid].long(), gt[valid].long()
    mask = (g >= 0) & (g < NUM_CLASSES) & (p >= 0) & (p < NUM_CLASSES)
    idx = g[mask] * NUM_CLASSES + p[mask]
    conf += torch.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    return conf

def miou(conf, subset=None):
    tp = conf.diag().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    iou = tp / (tp + fp + fn + 1e-10)
    iou_list = iou.cpu().tolist()
    if subset is None:
        present = (conf.sum(1) > 0).cpu().tolist()
        valid = [v for v, p in zip(iou_list, present) if p]
    else:
        valid = [iou_list[c] for c in subset]
    return (sum(valid)/len(valid) if valid else 0.0), iou_list

def print_report(conf, name="Model"):
    m19, ious = miou(conf)
    m17, _    = miou(conf, subset=FAIR_CLASSES)
    print(f"\n== {name} ==")
    print(f"  mIoU (19 cls) : {m19*100:.2f}")
    print(f"  mIoU (17 cls) : {m17*100:.2f}  (fair)")
    for c, n in enumerate(CLASSES):
        tag = " [excluded]" if c in FAIR_EXCLUDE else ""
        print(f"    {c:>2} {n:<14} {ious[c]*100:>6.2f}{tag}")
    return {"miou_19": m19, "miou_17": m17, "iou_per_class": ious}
