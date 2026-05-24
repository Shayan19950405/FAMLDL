# ── STEP 8 — EoMT Mask-Based Anomaly Baselines 
# Owner: Shabab (P4) · Branch: p4-eval
#
# This runs MSP, MaxLogit, MaxEntropy, and RbA on:
#   - EoMT-COCO        (133 classes, 200 queries)
#   - EoMT-Cityscapes  (19 classes,  100 queries)
#   - EoMT-Finetuned   (19 classes,  100 queries) ← from P2
# on all 5 anomaly datasets.

import os, sys, glob, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# ── Path setup (works on Colab) ──
sys.path.insert(0, '/content/project/eomt')
sys.path.insert(0, '/content/project/p1_infra')

from eomt_loader import load_eomt, to_per_pixel_logits, windowed_semantic_inference

# ── Image transforms ──
IMG_SIZE = (512, 1024)
input_transform = Compose([
    Resize(IMG_SIZE, Image.BILINEAR),
    ToTensor(),
])
target_transform = Compose([
    Resize(IMG_SIZE, Image.NEAREST),
])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Dataset paths ──
ANOMALY_ROOT = Path('/content/anomaly_datasets')
DATASETS = {
    'SMIYC_RA21':    ANOMALY_ROOT / 'RoadObsticle21/images/*.webp',
    'SMIYC_RO21':    ANOMALY_ROOT / 'RoadAnomaly21/images/*.png',
    'FS_LostFound':  ANOMALY_ROOT / 'LostAndFound/images/*.png',
    'FS_Static':     ANOMALY_ROOT / 'fs_static/images/*.jpg',
    'RoadAnomaly':   ANOMALY_ROOT / 'RoadAnomaly/images/*.jpg',
}

# ── Metrics ──
def fpr_at_95_tpr(scores, labels):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]

def compute_metrics(anomaly_scores, ood_gts):
    ood_gts = np.array(ood_gts)
    anomaly_scores = np.array(anomaly_scores)
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)
    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]
    val_out   = np.concatenate([ind_out, ood_out])
    val_label = np.concatenate([np.zeros(len(ind_out)), np.ones(len(ood_out))])
    auprc = average_precision_score(val_label, val_out)
    fpr   = fpr_at_95_tpr(val_out, val_label)
    return auprc * 100, fpr * 100

# ── GT mask loading ──
def load_gt_mask(path, pathGT):
    mask = Image.open(pathGT)
    mask = target_transform(mask)
    ood_gts = np.array(mask)
    if "RoadAnomaly" in pathGT:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)
    if "LostAndFound" in pathGT:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    if "Streethazard" in pathGT:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)
    return ood_gts

# ── Anomaly scoring methods ──
def score_msp(pixel_logits):
    """1 - max softmax probability. High = anomaly."""
    probs = pixel_logits.softmax(dim=0)
    return (1 - probs.max(dim=0).values).cpu().numpy()

def score_maxlogit(pixel_logits):
    """-max raw logit. High = anomaly."""
    return (-pixel_logits.max(dim=0).values).cpu().numpy()

def score_maxentropy(pixel_logits):
    """Entropy of softmax. High = anomaly."""
    probs = pixel_logits.softmax(dim=0)
    entropy = -(probs * probs.log()).sum(dim=0)
    return entropy.cpu().numpy()

def score_rba(mask_logits, class_logits):
    """
    RbA: Rejected by All (Nayal et al. 2023)
    For each pixel: max over known classes of (mask_prob * class_prob)
    Low = anomaly (nobody claims it) → we negate it so High = anomaly
    mask_logits:  [Q, H, W]
    class_logits: [Q, C+1]
    """
    cls = class_logits.softmax(dim=-1)[:, :-1]   # [Q, C] drop no-object
    msk = mask_logits.sigmoid()                    # [Q, H, W]
    # For each query: best class score * mask score
    best_cls = cls.max(dim=-1).values              # [Q]
    # Expand to [Q, H, W]
    best_cls = best_cls[:, None, None].expand_as(msk)
    query_scores = best_cls * msk                  # [Q, H, W]
    # RbA = max over queries per pixel → negate so high = anomaly
    rba = -query_scores.max(dim=0).values
    return rba.cpu().numpy()

# ── Main eval function for one model on one dataset ──
def eval_model_on_dataset(model, dataset_name, glob_pattern,
                           crop_size, num_classes, num_q,
                           is_coco=False):
    paths = sorted(glob.glob(str(glob_pattern)))
    if len(paths) == 0:
        print(f'  [WARN] No images found for {dataset_name}')
        return None

    scores = {m: [] for m in ['msp', 'maxlogit', 'maxentropy', 'rba']}
    gts = []

    for path in tqdm(paths, desc=f'{dataset_name}', leave=False):
        # Load image
        img = input_transform(Image.open(path).convert('RGB')).to(DEVICE)

        # Load GT mask
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
        if "fs_static"      in pathGT: pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly"    in pathGT: pathGT = pathGT.replace("jpg", "png")
        if not os.path.exists(pathGT):
            continue
        ood_gts = load_gt_mask(path, pathGT)
        if 1 not in np.unique(ood_gts):
            continue

        # Run EoMT — get raw mask and class logits
        with torch.no_grad():
            img_input = img.unsqueeze(0)             # [1, 3, H, W]
            mask_list, class_list = model(img_input / 255.0)
            mask_logits_raw  = mask_list[-1][0]      # [Q, h, w]
            class_logits_raw = class_list[-1][0]     # [Q, C+1]

        # Upsample mask logits to image size
        mask_logits = F.interpolate(
            mask_logits_raw.unsqueeze(0),
            size=IMG_SIZE, mode='bilinear', align_corners=False
        ).squeeze(0)                                 # [Q, H, W]

        # Per-pixel logits [C, H, W]
        pixel_logits = to_per_pixel_logits(
            mask_logits.unsqueeze(0),
            class_logits_raw.unsqueeze(0)
        ).squeeze(0)                                 # [C, H, W]

        # Score all methods
        scores['msp'].append(score_msp(pixel_logits))
        scores['maxlogit'].append(score_maxlogit(pixel_logits))
        scores['maxentropy'].append(score_maxentropy(pixel_logits))
        scores['rba'].append(score_rba(mask_logits, class_logits_raw))
        gts.append(ood_gts)

        torch.cuda.empty_cache()

    # Compute metrics
    results = {}
    for method, score_list in scores.items():
        auprc, fpr = compute_metrics(
            np.concatenate([s.flatten() for s in score_list]),
            np.concatenate([g.flatten() for g in gts])
        )
        results[method] = {'auprc': auprc, 'fpr95': fpr}
        print(f'  {method:<12} AuPRC={auprc:.1f}  FPR95={fpr:.1f}')

    return results

# ── Run everything ──
def run_all(checkpoints, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for ckpt_name, ckpt_cfg in checkpoints.items():
        print(f'\n{"="*60}')
        print(f'Model: {ckpt_name}')
        print(f'{"="*60}')

        model = load_eomt(
            ckpt_cfg['path'],
            num_classes=ckpt_cfg['num_classes'],
            num_q=ckpt_cfg['num_q'],
            img_size=ckpt_cfg['img_size'],
            device=DEVICE
        )

        all_results[ckpt_name] = {}
        for ds_name, ds_glob in DATASETS.items():
            print(f'\n  Dataset: {ds_name}')
            res = eval_model_on_dataset(
                model, ds_name, ds_glob,
                crop_size=ckpt_cfg['img_size'][0],
                num_classes=ckpt_cfg['num_classes'],
                num_q=ckpt_cfg['num_q'],
                is_coco=ckpt_cfg.get('is_coco', False)
            )
            if res:
                all_results[ckpt_name][ds_name] = res

        # Save after each model
        with open(output_dir / f'{ckpt_name}_results.json', 'w') as f:
            json.dump(all_results[ckpt_name], f, indent=2)
        print(f'\nSaved {ckpt_name} results')

        del model
        torch.cuda.empty_cache()

    # Save combined
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\n All results saved!')
    return all_results


if __name__ == '__main__':
    CHECKPOINTS = {
        'EoMT_COCO': {
            'path': '/content/checkpoints/eomt_coco.bin',
            'num_classes': 133, 'num_q': 200,
            'img_size': (640, 640), 'is_coco': True,
        },
        'EoMT_Cityscapes': {
            'path': '/content/checkpoints/eomt_cityscapes.bin',
            'num_classes': 19, 'num_q': 100,
            'img_size': (1024, 1024),
        },
        # Add finetuned when P2 is done:
        # 'EoMT_Finetuned': {
        #     'path': '/content/drive/MyDrive/FAMLDL/checkpoints/exp_5_3_lora.bin',
        #     'num_classes': 19, 'num_q': 100,
        #     'img_size': (1024, 1024)
    }

    results = run_all(
        CHECKPOINTS,
        output_dir='/content/drive/MyDrive/FAMLDL/results/step8'
    )