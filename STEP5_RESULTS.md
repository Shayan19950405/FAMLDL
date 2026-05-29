# Step 5 — Fine-tuning EoMT-COCO on Cityscapes

Three fine-tuning strategies were applied to adapt the EoMT-COCO checkpoint
(640×640, 200 queries, 133+1 classes) to Cityscapes semantic segmentation
(1024×1024, 100 queries, 19+1 classes). Starting point identical across all
three experiments: `eomt_coco_for_cityscapes.bin` (193 tensors copied verbatim
from COCO, 1 pos_embed interpolated 40×40→64×64, 3 incompatible heads freshly
re-initialized).

All experiments use EoMT's own Lightning training infrastructure
(`MaskClassificationSemantic` + `MaskClassificationLoss`) untouched. Only the
freezing strategy changes between runs.

## Results

| Experiment | mIoU (19 cls) | mIoU (17 cls fair) | Trainable % | Epochs | Wall time |
|------------|---------------|--------------------|-------------|--------|-----------|
| EoMT-COCO zero-shot (mapped)     | 55.00 | 61.47 |   0%   |  —  |  —  |
| **5.1 head-only**                | 75.52 | 77.09 |  6.9%  | 107 | 13.77 h |
| **5.2 partial (last-2 blocks)**  | 79.75 | 81.41 | 21.8%  |  60 |  7.18 h |
| **5.3 LoRA (rank 8, α=16)**      | 79.34 | 80.83 |  7.2%  |  60 |  ~5.8 h |
| EoMT-Cityscapes (paper ref)      | 81.45 | 82.78 | 100%   |  —  |  —  |

mIoU (17 cls fair) excludes `pole` and `rider`, which have no COCO panoptic
equivalents and unfairly penalize the COCO zero-shot baseline. Used as a
secondary metric for transparency.

## Headline finding

LoRA (7.2% trainable, attention QKV adapters in all 12 ViT blocks) matches
partial unfreeze (21.8% trainable) within 0.41 mIoU at 1/3 the parameter
budget. LoRA wins on 14 of 19 individual classes; partial wins decisively
only on `train` (+15.7 mIoU) and `bus` (+6.5), the rare large-vehicle classes
that benefit most from deep feature adaptation.

## Per-class IoU (19-class)

| ID | Class           | 5.1 head | 5.2 partial | 5.3 LoRA | best |
|----|-----------------|----------|-------------|----------|------|
|  0 | road            |   97.92  |   97.82     |  98.02   | LoRA |
|  1 | sidewalk        |   83.82  |   82.86     |  84.79   | LoRA |
|  2 | building        |   93.07  |   92.96     |  93.67   | LoRA |
|  3 | wall            |   61.80  |   65.75     |  62.06   | partial |
|  4 | fence           |   59.54  |   60.89     |  64.21   | LoRA |
|  5 | pole            |   61.64  |   62.76     |  67.02   | LoRA |
|  6 | traffic light   |   70.47  |   70.88     |  73.12   | LoRA |
|  7 | traffic sign    |   76.89  |   77.86     |  80.12   | LoRA |
|  8 | vegetation      |   92.47  |   92.35     |  92.91   | LoRA |
|  9 | terrain         |   65.59  |   63.34     |  66.33   | LoRA |
| 10 | sky             |   95.13  |   95.09     |  95.23   | LoRA |
| 11 | person          |   83.70  |   84.78     |  85.01   | LoRA |
| 12 | rider           |   62.86  |   68.59     |  66.31   | partial |
| 13 | car             |   94.58  |   95.20     |  95.42   | LoRA |
| 14 | truck           |   72.66  |   81.54     |  82.49   | LoRA |
| 15 | bus             |   78.12  |   90.63     |  84.09   | partial |
| 16 | train           |   40.92  |   82.88     |  67.15   | partial |
| 17 | motorcycle      |   64.07  |   69.26     |  69.12   | partial |
| 18 | bicycle         |   79.71  |   79.88     |  80.33   | LoRA |

## Key methodology decisions

1. **Adapted starting checkpoint**: COCO weights have incompatible shapes for
   Cityscapes. `pos_embed` interpolated bicubic 40²→64² (Dosovitskiy 2021 standard).
   `q.weight [200,768]`, `class_head.weight [134,768]`, `class_head.bias [134]`
   re-initialized to PyTorch defaults — these would crash on shape mismatch otherwise.

2. **EoMT's own loss untouched**: Used Lightning's `MaskClassificationSemantic`
   module with `MaskClassificationLoss` (Mask2Former-style mask BCE + dice +
   class CE + Hungarian matching). Zero training math was reimplemented.
   Strategies differ only in `FreezingCallback`.

3. **Native-resolution windowed evaluation** (paper Sec. A.3): Each 1024×2048
   image evaluated as overlapping 1024×1024 crops with averaged logits.
   Same pipeline used for Step 4 baselines.

4. **LoRA target layer**: `attn.qkv` (the fused QKV linear) in each of 12 ViT
   blocks. Per Hu et al. 2022 Sec. 4.2, attention QKV is the most effective
   target. Rank 8, alpha 16 (paper defaults, scale = 2).

5. **LoRA merge formula** (`exp_5_3_lora.bin`): `W_eff = W_base + (α/r) · B·A`.
   Final checkpoint has no LoRA structure — loads via vanilla `load_eomt()`.

## Reproducibility

- Configs documented in: `eomt/configs/finetune/README.md`
- Shared eval infrastructure: `p1_infra/` (cityscapes_loader.py, class_mapping.py,
  miou_eval.py, eomt_loader.py)
- Result JSONs (on Drive): `/FAMLDL/results/step5/exp_5_*_eval.json`
- Audit-trail checkpoints (Lightning .ckpt with optimizer state) on Drive:
  `/FAMLDL/checkpoints/exp_5_*-epoch=*.ckpt`
- Wandb runs: `s338784-politecnico-di-torino/FAMLDL`
  - 5.1 head-only: `wl0hf8h8`
  - 5.2 partial: `ghlkafzn`
  - 5.3 LoRA: `mm2zp30q`
