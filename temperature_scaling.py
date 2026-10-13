"""
temperature_scaling.py — Temperature Scaling for Anomaly Segmentation
======================================================================
Adapted from Guo et al. (2017) / gpleiss/temperature_scaling for use
with pixel-wise segmentation models (ERFNet + Cityscapes).

Key differences from the original classification version:
  - Works on saved .npy logit maps (C, H, W) rather than batched tensors
  - apply_temperature() rescales saved logit arrays directly — no GPU needed
  - score_msp_t() computes MSP anomaly score at a given temperature T
  - find_best_temperature() does a grid search over T values to maximise AuPRC
  - No DataLoader required: we operate on pre-saved .npy files (the PRO TIP
    from the assignment: run the forward pass ONCE, then reuse saved logits)
"""

import os
import glob
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve


# ── Temperature application ────────────────────────────────────────────────

def softmax(x, axis=0):
    """Numerically stable softmax along `axis`."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def score_msp_t(logits_chw: np.ndarray, temperature: float) -> np.ndarray:
    """
    MSP anomaly score with temperature scaling.

    temperature < 1 → sharper softmax → more confident on in-dist pixels
                    → better separation between in-dist and OOD
    temperature > 1 → softer softmax → flattens confidence everywhere
    temperature = 1 → standard MSP (no scaling)

    Args:
        logits_chw: raw logits array of shape (C, H, W), float32
        temperature: scalar T > 0

    Returns:
        anomaly_score: (H, W) float32, higher = more anomalous
    """
    scaled = logits_chw / temperature           # (C, H, W)
    probs  = softmax(scaled, axis=0)            # (C, H, W)
    return 1.0 - probs.max(axis=0)             # (H, W)


# ── Loading helpers ────────────────────────────────────────────────────────

def load_logits(logits_dir: str) -> dict:
    """
    Load all saved raw-logit .npy files from a directory.

    Expects files saved as  <stem>_logits.npy  (shape C×H×W, float32).

    Returns dict: {stem: logits_array}
    """
    result = {}
    for p in sorted(glob.glob(os.path.join(logits_dir, '*_logits.npy'))):
        stem = os.path.basename(p).replace('_logits.npy', '')
        result[stem] = np.load(p).astype(np.float32)
    if not result:
        raise FileNotFoundError(
            f'No *_logits.npy files found in {logits_dir}.\n'
            'Run the "Save raw logits" cell in the notebook first.')
    return result


def load_masks(masks_dir: str, stems: list, mask_ext: str = None) -> dict:
    """
    Load ground-truth mask PNGs matching `stems`.

    Mask convention: 255 = void/ignore, >0 = anomaly, 0 = normal.
    Returns dict: {stem: mask_array (H, W) uint8}
    """
    if mask_ext is None:
        for ext in ('png', 'jpg', 'jpeg', 'webp'):
            candidate = os.path.join(masks_dir, f'{stems[0]}.{ext}')
            if os.path.isfile(candidate):
                mask_ext = ext
                break
        mask_ext = mask_ext or 'png'

    result = {}
    for stem in stems:
        p = os.path.join(masks_dir, f'{stem}.{mask_ext}')
        if os.path.isfile(p):
            result[stem] = np.array(Image.open(p))
        else:
            print(f'  [SKIP] mask not found: {p}')
    return result


# ── Metric helpers ─────────────────────────────────────────────────────────

def compute_metrics(scores_dict: dict, masks_dict: dict):
    """
    Given dicts of {stem: score_map} and {stem: mask}, compute AuPRC + FPR95.

    Returns (auprc, fpr95) both in [0, 1].
    """
    S, L = [], []
    for stem, score in scores_dict.items():
        if stem not in masks_dict:
            continue
        mask  = masks_dict[stem]
        # Resize score to mask shape if needed
        if score.shape != mask.shape:
            score = np.array(
                Image.fromarray(score).resize(
                    (mask.shape[1], mask.shape[0]), Image.BILINEAR))
        valid = mask != 255
        if valid.sum() == 0:
            continue
        S.append(score[valid].ravel())
        L.append((mask[valid] > 0).ravel().astype(np.int32))

    if not S:
        raise RuntimeError('No valid (score, mask) pairs found.')

    S_cat = np.concatenate(S)
    L_cat = np.concatenate(L)

    auprc = float(average_precision_score(L_cat, S_cat))
    fpr_arr, tpr_arr, _ = roc_curve(L_cat, S_cat)
    idx   = np.searchsorted(tpr_arr, 0.95)
    fpr95 = float(fpr_arr[min(idx, len(fpr_arr) - 1)])

    return auprc, fpr95


# ── Grid search ────────────────────────────────────────────────────────────

def find_best_temperature(
    logits_dict: dict,
    masks_dict:  dict,
    temperatures: list = None,
    metric: str = 'auprc',
) -> dict:
    """
    Grid search over temperature values. Returns results for every T.

    Args:
        logits_dict:  {stem: logits (C,H,W)}
        masks_dict:   {stem: mask   (H,W)}
        temperatures: list of T values to try
                      (default: [0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0])
        metric:       'auprc' or 'fpr95' — which metric to optimise

    Returns:
        results dict:
          {
            't': [...],
            'auprc': [...],
            'fpr95': [...],
            'best_t': float,
            'best_auprc': float,
            'best_fpr95': float,
          }
    """
    if temperatures is None:
        temperatures = [0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0]

    ts, auprcs, fpr95s = [], [], []

    for t in temperatures:
        scores_dict = {
            stem: score_msp_t(logits, t)
            for stem, logits in logits_dict.items()
        }
        auprc, fpr95 = compute_metrics(scores_dict, masks_dict)
        ts.append(t)
        auprcs.append(auprc)
        fpr95s.append(fpr95)
        print(f'  T={t:.4f}  AuPRC={auprc*100:.2f}%  FPR95={fpr95*100:.2f}%')

    # Best T: maximise AuPRC (or minimise FPR95)
    if metric == 'auprc':
        best_idx = int(np.argmax(auprcs))
    else:
        best_idx = int(np.argmin(fpr95s))

    return {
        't':          ts,
        'auprc':      auprcs,
        'fpr95':      fpr95s,
        'best_t':     ts[best_idx],
        'best_auprc': auprcs[best_idx],
        'best_fpr95': fpr95s[best_idx],
    }


# ── Convenience: run everything for one dataset ────────────────────────────

def evaluate_temperature_scaling(
    logits_dir:   str,
    masks_dir:    str,
    fixed_temps:  list = None,
    search_temps: list = None,
    metric:       str  = 'auprc',
    verbose:      bool = True,
) -> dict:
    """
    Full pipeline for one dataset:
      1. Load saved logits
      2. Evaluate fixed T values  (0.5, 0.75, 1.0=MSP-baseline, 1.1)
      3. Grid-search for best T
      4. Return all results

    Args:
        logits_dir:   directory containing *_logits.npy files
        masks_dir:    directory containing ground-truth masks
        fixed_temps:  T values required by the assignment table
                      default = [0.5, 0.75, 1.0, 1.1]
        search_temps: finer grid for best-T search
                      default = 30 values from 0.1 to 3.0
        metric:       optimisation criterion for best T

    Returns:
        dict with keys 'fixed' and 'search', each containing
        t / auprc / fpr95 / best_t / best_auprc / best_fpr95
    """
    if fixed_temps  is None:
        fixed_temps  = [0.5, 0.75, 1.0, 1.1]
    if search_temps is None:
        search_temps = list(np.round(np.linspace(0.1, 3.0, 30), 3))

    logits_dict = load_logits(logits_dir)
    stems       = list(logits_dict.keys())
    masks_dict  = load_masks(masks_dir, stems)

    if verbose:
        print(f'Loaded {len(logits_dict)} logit maps, {len(masks_dict)} masks.')
        print('\n── Fixed temperatures ─────────────────────────────────────')

    fixed_results  = find_best_temperature(logits_dict, masks_dict,
                                           fixed_temps, metric)

    if verbose:
        print('\n── Temperature grid search ────────────────────────────────')

    search_results = find_best_temperature(logits_dict, masks_dict,
                                           search_temps, metric)

    if verbose:
        print(f'\nBest T (grid search): {search_results["best_t"]:.3f}  '
              f'→  AuPRC={search_results["best_auprc"]*100:.2f}%  '
              f'FPR95={search_results["best_fpr95"]*100:.2f}%')

    return {'fixed': fixed_results, 'search': search_results}
