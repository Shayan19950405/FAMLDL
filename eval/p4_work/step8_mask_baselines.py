# ── STEP 8 — EoMT Mask-Based Anomaly Baselines ──
import os, sys, glob, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.metrics import average_precision_score, roc_curve
from tqdm import tqdm

os.chdir('/content/project/eomt')
sys.path.insert(0, '/content/project/eomt')
from models.eomt import EoMT
from models.vit import ViT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ANOMALY_ROOT = Path('/content/anomaly_datasets/Validation_Dataset')
DATASETS = {
    'SMIYC_RA21':   ANOMALY_ROOT / 'RoadObsticle21/images/*.webp',
    'SMIYC_RO21':   ANOMALY_ROOT / 'RoadAnomaly21/images/*.png',
    'FS_LostFound': ANOMALY_ROOT / 'FS_LostFound_full/images/*.png',
    'FS_Static':    ANOMALY_ROOT / 'fs_static/images/*.jpg',
    'RoadAnomaly':  ANOMALY_ROOT / 'RoadAnomaly/images/*.jpg',
}

def load_eomt(ckpt_path, num_classes, num_q, img_size):
    enc = ViT(img_size=img_size, patch_size=16,
              backbone_name='vit_base_patch14_reg4_dinov2')
    model = EoMT(encoder=enc, num_classes=num_classes, num_q=num_q,
                 num_blocks=3, masked_attn_enabled=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    clean = {k[len('network.'):]: v for k, v in sd.items()
             if k.startswith('network.')}
    model.load_state_dict(clean, strict=False)
    return model.eval().to(DEVICE)

@torch.no_grad()
def to_per_pixel_logits(mask_logits, class_logits):
    cls = class_logits.softmax(dim=-1)[..., :-1]
    return torch.einsum('bqhw,bqc->bchw', mask_logits.sigmoid(), cls)

def score_msp(pl):
    return (1 - pl.softmax(dim=0).max(dim=0).values).cpu().numpy()

def score_maxlogit(pl):
    return (-pl.max(dim=0).values).cpu().numpy()

def score_maxentropy(pl):
    p = pl.softmax(dim=0)
    return (-(p * p.log()).sum(dim=0)).cpu().numpy()

def score_rba(ml, cl):
    cls = cl.softmax(dim=-1)[:, :-1].max(dim=-1).values
    msk = ml.sigmoid()
    scores = cls[:, None, None].expand_as(msk) * msk
    return (-scores.max(dim=0).values).cpu().numpy()

def fpr_at_95_tpr(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr[np.argmin(np.abs(tpr - 0.95))]

def compute_metrics(score_list, gt_list):
    scores = np.concatenate([s.flatten() for s in score_list])
    gts    = np.concatenate([g.flatten() for g in gt_list])
    ood = scores[gts == 1]
    ind = scores[gts == 0]
    val_out   = np.concatenate([ind, ood])
    val_label = np.concatenate([np.zeros(len(ind)), np.ones(len(ood))])
    auprc = average_precision_score(val_label, val_out) * 100
    fpr   = fpr_at_95_tpr(val_out, val_label) * 100
    return auprc, fpr

def load_gt(path, pathGT, img_size):
    mask = np.array(
        Image.open(pathGT).resize(
            (img_size[1], img_size[0]), Image.NEAREST))
    if 'RoadAnomaly' in pathGT:
        mask = np.where(mask == 2, 1, mask)
    if 'LostAndFound' in pathGT or 'FS_LostFound' in pathGT:
        mask = np.where(mask == 0, 255, mask)
        mask = np.where(mask == 1, 0, mask)
        mask = np.where((mask > 1) & (mask < 201), 1, mask)
    return mask

def eval_one(model, glob_pattern, img_size):
    paths = sorted(glob.glob(str(glob_pattern)))
    if not paths:
        print(f'    [WARN] No images found: {glob_pattern}')
        return None

    input_tf  = Compose([Resize(img_size, Image.BILINEAR), ToTensor()])
    scores = {m: [] for m in ['msp','maxlogit','maxentropy','rba']}
    gts = []

    for path in tqdm(paths, leave=False):
        pathGT = path.replace('images', 'labels_masks')
        if 'RoadObsticle21' in pathGT: pathGT = pathGT.replace('webp','png')
        if 'fs_static'      in pathGT: pathGT = pathGT.replace('jpg','png')
        if 'RoadAnomaly'    in pathGT: pathGT = pathGT.replace('jpg','png')
        if not os.path.exists(pathGT): continue
        gt = load_gt(path, pathGT, img_size)
        if 1 not in np.unique(gt): continue

        img = input_tf(Image.open(path).convert('RGB')).to(DEVICE)
        with torch.no_grad():
            ml, cl = model(img.unsqueeze(0) / 255.0)
            ml = F.interpolate(ml[-1], img_size,
                               mode='bilinear', align_corners=False)
            cl = cl[-1]
            pl = to_per_pixel_logits(ml, cl).squeeze(0)
            ml0 = ml.squeeze(0)
            cl0 = cl.squeeze(0)

        scores['msp'].append(score_msp(pl))
        scores['maxlogit'].append(score_maxlogit(pl))
        scores['maxentropy'].append(score_maxentropy(pl))
        scores['rba'].append(score_rba(ml0, cl0))
        gts.append(gt)
        torch.cuda.empty_cache()

    if not gts:
        print('    [WARN] No valid images found')
        return None

    results = {}
    for m, sl in scores.items():
        auprc, fpr = compute_metrics(sl, gts)
        results[m] = {'auprc': round(auprc,2), 'fpr95': round(fpr,2)}
        print(f'    {m:<12} AuPRC={auprc:.1f}  FPR95={fpr:.1f}')
    return results

# ── Checkpoints ──
CHECKPOINTS = {
    'EoMT_COCO': {
        'path': '/content/checkpoints/eomt_coco.bin',
        'num_classes': 133, 'num_q': 200, 'img_size': (640, 640),
    },
    'EoMT_Cityscapes': {
        'path': '/content/checkpoints/eomt_cityscapes.bin',
        'num_classes': 19, 'num_q': 100, 'img_size': (1024, 1024),
    },
    'EoMT_Finetuned': {
        'path': '/content/drive/MyDrive/FAMLDL/checkpoints/exp_5_2_partial.bin',
        'num_classes': 19, 'num_q': 100, 'img_size': (1024, 1024),
    },
}

OUT = Path('/content/drive/MyDrive/FAMLDL/results/step8')
OUT.mkdir(parents=True, exist_ok=True)
all_results = {}

for ckpt_name, cfg in CHECKPOINTS.items():
    print(f'\n{"="*50}')
    print(f'Model: {ckpt_name}')
    print(f'{"="*50}')
    if not Path(cfg['path']).exists():
        print(f'[SKIP] checkpoint not found: {cfg["path"]}')
        continue
    model = load_eomt(cfg['path'], cfg['num_classes'],
                      cfg['num_q'], cfg['img_size'])
    all_results[ckpt_name] = {}
    for ds_name, ds_glob in DATASETS.items():
        print(f'\n  Dataset: {ds_name}')
        res = eval_one(model, ds_glob, cfg['img_size'])
        if res:
            all_results[ckpt_name][ds_name] = res
    with open(OUT / f'{ckpt_name}.json', 'w') as f:
        json.dump(all_results[ckpt_name], f, indent=2)
    print(f'\n {ckpt_name} results saved')
    del model
    torch.cuda.empty_cache()

with open(OUT / 'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\n ALL EXPERIMENTS COMPLETE!')