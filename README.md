# Comprehensive Road Scene Understanding for Autonomous Driving

Group project (Group 53) for **Fundamentals of AI, Machine and Deep Learning (FAIMDL)**, Politecnico di Torino, 2026.

A self-driving car has to do two things with every camera frame: label each pixel (road, car, person, …) and notice when something on the road does not belong to any class it was trained on (lost cargo, an animal, debris). This project studies both:

1. **Semantic segmentation** on Cityscapes with EoMT (Encoder-only Mask Transformer on a DINOv2-pretrained ViT-Base backbone, 16×16 patches), including parameter-efficient fine-tuning of a COCO-pretrained model.
2. **Post-hoc anomaly (out-of-distribution) detection** on five road-anomaly benchmarks, comparing a pixel-based CNN (ERFNet) with mask-based EoMT models.

## Main results

### Segmentation — Cityscapes val (500 images, native 1024×2048, windowed inference)

| Model | mIoU (19 classes) | mIoU (17 classes)¹ | Trainable params | Training time² |
|---|---|---|---|---|
| EoMT-COCO, zero-shot (class-mapped) | 55.00 | 61.47 | 0% | — |
| Fine-tune 5.1 — head only | 75.52 | 77.09 | 6.9% | 13.77 h (107 ep) |
| Fine-tune 5.2 — last 2 ViT blocks unfrozen | **79.75** | **81.41** | 21.8% | 7.18 h (60 ep) |
| Fine-tune 5.3 — LoRA (r=8, α=16, `attn.qkv`) | 79.34 | 80.83 | 7.2% | 7.04 h (60 ep) |
| EoMT-Cityscapes (fully trained reference) | 81.45 | 82.78 | 100% | — |

¹ Excludes `pole` and `rider`, which have no COCO equivalent, for a fair comparison with the zero-shot COCO model.
² Single A100 40 GB, AMP 16-bit mixed precision, batch size 4.

**Takeaway:** LoRA comes within 0.41 mIoU of partial unfreezing while training about one third as many parameters, and reaches ~97% of the fully trained reference. LoRA is best on 14 of 19 classes; partial unfreezing wins the other 5, by a wide margin on the rare large vehicles (train +15.7 IoU, bus +6.5) and narrowly on wall, rider and motorcycle. Per-class tables are in [`STEP5_RESULTS.md`](STEP5_RESULTS.md).

### Anomaly detection — AuPRC % (↑) / FPR95 % (↓)

Benchmarks: SegmentMeIfYouCan (SMIYC) RoadAnomaly21 (RA-21, 10 images) and RoadObstacle21 (RO-21, 30), Fishyscapes Lost&Found (FS L&F, 100) and Static (FS Static, 30), and Road Anomaly (60). ERFNet scores images at native resolution; the EoMT checkpoints resize each image bilinearly to their training resolution (640² COCO, 1024² otherwise).

| Model | Method | RA-21 | RO-21 | FS L&F | FS Static | Road Anomaly |
|---|---|---|---|---|---|---|
| ERFNet | MSP | 15.8 / 89.4 | 0.6 / 96.4 | 0.3 / 91.5 | 1.1 / 98.1 | 10.8 / 89.3 |
| ERFNet | MaxLogit | 15.4 / 87.7 | 2.5 / 89.8 | 0.3 / 86.1 | 1.2 / 91.9 | 10.9 / 87.3 |
| ERFNet | MaxEntropy | 15.2 / 89.5 | 0.7 / 96.5 | 0.3 / 91.5 | 1.1 / 98.1 | 10.4 / 89.5 |
| EoMT-COCO | MSP | 15.36 / 94.65 | 0.70 / 95.47 | 0.30 / 92.70 | 2.18 / 95.38 | 9.87 / 95.09 |
| EoMT-COCO | MaxLogit | 15.76 / 84.25 | 0.43 / 93.64 | 0.28 / 88.21 | 1.71 / 89.16 | 8.29 / 94.67 |
| EoMT-COCO | MaxEntropy | 16.40 / 92.81 | 0.63 / 95.86 | 0.30 / 89.07 | 2.14 / 91.82 | 9.68 / 93.84 |
| EoMT-COCO | RbA | 15.63 / 84.20 | 0.43 / 93.58 | 0.27 / 87.83 | 1.68 / 89.33 | 8.19 / 94.70 |
| EoMT-Cityscapes | MSP | 12.17 / 92.93 | 0.37 / 99.85 | 0.41 / 65.13 | 2.57 / 87.76 | 8.76 / 93.01 |
| EoMT-Cityscapes | MaxLogit | 12.18 / 92.93 | 0.37 / 99.86 | 0.41 / 65.10 | 2.57 / 87.45 | 8.60 / 92.92 |
| EoMT-Cityscapes | MaxEntropy | 12.79 / 92.92 | 0.38 / 99.87 | 0.43 / 65.10 | 2.98 / 87.41 | 8.70 / 92.86 |
| EoMT-Cityscapes | RbA | 12.32 / 93.14 | 0.37 / 99.89 | 0.41 / 64.59 | 2.61 / 82.05 | 8.65 / 91.39 |
| EoMT-Finetuned (5.2) | MSP | 15.55 / 93.12 | 0.76 / 90.82 | 0.30 / 93.86 | 2.24 / 92.32 | 10.46 / 92.56 |
| EoMT-Finetuned (5.2) | MaxLogit | 15.65 / 92.96 | 0.84 / 90.65 | 0.32 / 93.84 | 2.35 / 92.59 | 10.57 / 92.47 |
| EoMT-Finetuned (5.2) | MaxEntropy | 14.71 / 93.86 | 0.67 / 93.65 | 0.28 / 94.01 | 2.06 / 93.93 | 9.78 / 93.88 |
| EoMT-Finetuned (5.2) | RbA | 15.65 / 92.97 | 0.84 / 90.74 | 0.32 / 93.89 | 2.35 / 92.59 | 10.57 / 92.47 |

**Takeaways:**
- Across the checkpoints we evaluated, the choice of checkpoint affected anomaly performance more than the choice of scoring method; the scoring method moved AuPRC by less than one point in 17 of 20 model × dataset settings.
- Temperature scaling on MSP gave only small gains: at most +0.41 AuPRC for ERFNet (T = 0.5 on RA-21) and +0.23 for EoMT-Finetuned (NLL-optimal T = 0.01, RA-21). T = 1.1 was slightly worse than no scaling on most datasets.
- Post-hoc scores remain weak on these benchmarks (AuPRC below 3% on three of five), so anomaly detection is an open problem here, not a solved one.
- Only the partial-unfreeze checkpoint (5.2) was evaluated for anomaly detection; head-only and LoRA were not.

## The eight project steps

| Step | What | Where |
|---|---|---|
| 1–3 | Background: ERFNet, panoptic segmentation, mask architectures | — |
| 4 | Evaluate EoMT-COCO (zero-shot) and EoMT-Cityscapes on Cityscapes; shared data loader, class mapping, windowed inference, mIoU | [`p1_infra/`](p1_infra), `FAMLDL.ipynb` |
| 5 | Fine-tune EoMT-COCO on Cityscapes: head-only, partial unfreeze, LoRA (AMP, AdamW with layer-wise LR decay, EoMT's Mask2Former-style loss) | `FAMLDL.ipynb`, [`STEP5_RESULTS.md`](STEP5_RESULTS.md), [`eomt/configs/finetune/`](eomt/configs/finetune/README.md) |
| 6 | Background: post-hoc anomaly scoring | — |
| 7 | ERFNet pixel-based anomaly scoring: MSP, MaxLogit, MaxEntropy, temperature scaling | [`eval/evalAnomaly.py`](eval/evalAnomaly.py), [`eval/metrics.py`](eval/metrics.py), [`temperature_scaling.py`](temperature_scaling.py), [`anomaly_detection_v3.ipynb`](anomaly_detection_v3.ipynb), [`step7_erfnet_temperature_scaling.ipynb`](step7_erfnet_temperature_scaling.ipynb) |
| 8 | EoMT mask-based anomaly scoring: MSP, MaxLogit, MaxEntropy, RbA, temperature scaling | `FAMLDL.ipynb`, [`eval/p4_work/step8_mask_baselines.py`](eval/p4_work/step8_mask_baselines.py) |

## Repository layout

```
FAMLDL.ipynb                    end-to-end notebook (Steps 4, 5, 8) with recorded outputs
STEP5_RESULTS.md                Step 5 results and per-class IoU
p1_infra/                       shared modules: Cityscapes loader, COCO→Cityscapes mapping,
                                EoMT loader + windowed inference, mIoU evaluation
anomaly_detection_v3.ipynb      Step 7 (ERFNet anomaly baselines)
step7_erfnet_temperature_scaling.ipynb   Step 7 temperature-scaling sweep
temperature_scaling.py          temperature scaling for MSP
eval/                           ERFNet evaluation code (course starter + Step 7 scoring and metrics)
eval/p4_work/                   Step 8 script version
eomt/                           EoMT source (from tue-mps/eomt, via the course starter)
trained_models/                 pretrained ERFNet weights (course-provided)
```

## Running it

The code was run on Google Colab. Large files are not in the repository: the Cityscapes zips, `Anomaly_Validation_Datasets.zip`, the EoMT checkpoints, and our fine-tuned checkpoints are expected under `MyDrive/FAMLDL/` on Google Drive. Cell 1 of `FAMLDL.ipynb` mounts Drive, clones this repo, installs dependencies, and unpacks the data. The EoMT reference checkpoints are linked from [`eomt/README.md`](eomt/README.md); the anomaly validation sets from [`eval/README.md`](eval/README.md).

## Team

| Member | Branch | Responsibility |
|---|---|---|
| Shayan Bagheri | `p1-infra` | Shared infrastructure, Step 4, integration and final audit |
| Shahrzad Shivaei | `p2-train` | Step 5 — fine-tuning experiments |
| Most Marufatul Jannat Mim | `p3-baselines` | Step 7 — ERFNet anomaly baselines |
| Abdullah Shabab | `p4-eval` | Step 8 — EoMT anomaly evaluation |

## Credits

- Course starter repository: [AlessandroMarinai/MaskArchitectureAnomaly_CourseProject](https://github.com/AlessandroMarinai/MaskArchitectureAnomaly_CourseProject) (ERFNet evaluation code, pretrained ERFNet, EoMT folder).
- EoMT: Kerssies et al., *Your ViT is Secretly an Image Segmentation Model*, CVPR 2025 — [tue-mps/eomt](https://github.com/tue-mps/eomt) (license in [`eomt/LICENSE`](eomt/LICENSE)).
- ERFNet: Romera et al., *ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation*, IEEE T-ITS 2018.
- LoRA: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
- RbA: Nayal et al., *RbA: Segmenting Unknown Regions Rejected by All*, ICCV 2023.
- Datasets: Cityscapes, SegmentMeIfYouCan (RoadAnomaly21, RoadObstacle21), Fishyscapes (Lost&Found, Static), Road Anomaly.
