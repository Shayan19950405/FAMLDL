# Step 5 fine-tuning — actual hyperparameters used

The training was NOT launched via Lightning CLI YAML — it was launched
directly from notebook cells (Cells 16, 18, 21 of `FAMLDL.ipynb`).
This README documents the real hyperparameters as a reproducibility reference.
For LoRA, the injection step requires Python code (see notebook Cell 20) and
cannot be expressed in YAML alone.

## Common settings (all three experiments)

| Setting | Value |
|---|---|
| seed | 42 |
| precision | 16-mixed (AMP) |
| devices | 1 (A100 40 GB) |
| gradient_clip | 0.01, norm |
| batch_size | 4 |
| IMG_SIZE | (1024, 1024) |
| num_queries | 100 |
| num_classes | 19 |
| num_blocks | 3 (mask-classification heads, paper default) |
| starting ckpt | `eomt_coco_for_cityscapes.bin` (adapted; see notebook Cell 14) |
| backbone | vit_base_patch14_reg4_dinov2 |
| loss | EoMT's `MaskClassificationLoss` (Mask2Former-style, untouched) |
| logger | WandbLogger (project FAMLDL) |
| best-ckpt monitor | metrics/val_iou_all (max) |

## Experiment 5.1 — head-only

| Setting | Value |
|---|---|
| epochs | 107 |
| FreezingCallback.strategy | head_only |
| trainable | encoder fully frozen; only queries + class_head + mask_head + upscale |
| trainable % | 6.9% |
| ANNEAL_START | [5950, 14875, 23800] |
| ANNEAL_END | [11900, 20825, 29750] |
| wall time | 13.77 h |
| wandb run | wl0hf8h8 |
| best epoch | 97 |
| **mIoU (19 cls)** | **75.52** |
| mIoU (17 cls fair) | 77.09 |

## Experiment 5.2 — partial unfreeze

| Setting | Value |
|---|---|
| epochs | 60 |
| FreezingCallback.strategy | partial |
| FreezingCallback.unfreeze_last_k | 2 |
| trainable | last 2 ViT blocks + backbone.norm + all heads |
| trainable % | 21.8% |
| ANNEAL_START | [3317, 8292, 13268] |
| ANNEAL_END | [6634, 11609, 16585] |
| wall time | 7.18 h |
| wandb run | ghlkafzn |
| best epoch | 53 |
| **mIoU (19 cls)** | **79.75** |
| mIoU (17 cls fair) | 81.41 |

## Experiment 5.3 — LoRA

| Setting | Value |
|---|---|
| epochs | 60 |
| FreezingCallback.strategy | lora |
| LoRA target | attn.qkv (per Hu et al. 2022 Sec. 4.2) |
| LoRA rank (r) | 8 |
| LoRA alpha | 16.0 |
| LoRA scale (α/r) | 2.0 |
| LoRA blocks injected | all 12 ViT blocks |
| trainable | LoRA adapters + all heads |
| trainable params | 6,895,892 |
| trainable % | 7.2% |
| ANNEAL_START | [3317, 8292, 13268] |
| ANNEAL_END | [6634, 11609, 16585] |
| wall time | ~5.8 h |
| wandb run | mm2zp30q |
| best epoch | 56 |
| LoRA merge formula | W_eff = W_base + (α/r) · B · A |
| **mIoU (19 cls)** | **79.34** |
| mIoU (17 cls fair) | 80.83 |

## Summary

| Method | mIoU (19) | Trainable % | mIoU per % trainable |
|---|---|---|---|
| EoMT-COCO zero-shot | 55.00 |  0%  | — |
| 5.1 head-only       | 75.52 | 6.9% | 10.9 |
| 5.2 partial         | 79.75 | 21.8%| 3.7  |
| **5.3 LoRA**        | **79.34** | **7.2%** | **11.0** |
| EoMT-Cityscapes ref | 81.45 | 100% | 0.81 |

LoRA matches partial unfreeze within 0.41 mIoU at 1/3 the parameter budget.
LoRA wins outright on 14 of 19 individual classes.
