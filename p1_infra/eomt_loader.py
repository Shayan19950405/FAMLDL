"""EoMT model loading + windowed semantic inference."""
import math, sys, torch
import torch.nn.functional as F
sys.path.insert(0, "/content/project/eomt")
from models.eomt import EoMT
from models.vit import ViT

def build_eomt(num_classes, num_q, img_size, num_blocks=3,
               backbone="vit_base_patch14_reg4_dinov2", patch_size=16):
    enc = ViT(img_size=img_size, patch_size=patch_size, backbone_name=backbone)
    return EoMT(encoder=enc, num_classes=num_classes, num_q=num_q,
                num_blocks=num_blocks, masked_attn_enabled=True)

def load_eomt(ckpt_path, num_classes, num_q, img_size, device="cuda"):
    model = build_eomt(num_classes=num_classes, num_q=num_q, img_size=img_size)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    clean = {k[len("network."):]: v for k, v in sd.items() if k.startswith("network.")}
    model.load_state_dict(clean, strict=False)
    return model.eval().to(device)

@torch.no_grad()
def to_per_pixel_logits(mask_logits, class_logits, drop_no_object=True):
    cls = class_logits.softmax(dim=-1)
    if drop_no_object: cls = cls[..., :-1]
    return torch.einsum("bqhw, bqc -> bchw", mask_logits.sigmoid(), cls)

@torch.no_grad()
def windowed_semantic_inference(model, image, crop_size, device="cuda"):
    """EoMT windowed semantic inference. Image must be [3,H,W] in [0,255]."""
    image = image.float().to(device)
    _, H, W = image.shape
    short = min(H, W)
    factor = crop_size / short
    new_H, new_W = round(H * factor), round(W * factor)
    img_r = F.interpolate(image.unsqueeze(0), size=(new_H, new_W),
                          mode="bilinear", align_corners=False).squeeze(0)
    long_dim = max(new_H, new_W)
    n = max(1, math.ceil(long_dim / crop_size))
    overlap = n * crop_size - long_dim
    step = (crop_size - overlap / (n - 1)) if n > 1 else 0
    crops, origins = [], []
    for j in range(n):
        s = int(round(j * step)); e = s + crop_size
        crops.append(img_r[:, s:e, :] if new_H > new_W else img_r[:, :, s:e])
        origins.append((s, e))
    batch = torch.stack(crops, dim=0)
    mask_list, class_list = model(batch / 255.0)   # CRITICAL: divide by 255
    pp = to_per_pixel_logits(mask_list[-1], class_list[-1])
    pp = F.interpolate(pp, size=(crop_size, crop_size),
                       mode="bilinear", align_corners=False)
    C = pp.shape[1]
    sums   = torch.zeros((C, new_H, new_W), device=device)
    counts = torch.zeros((C, new_H, new_W), device=device)
    for k, (s, e) in enumerate(origins):
        if new_H > new_W:
            sums[:, s:e, :] += pp[k]; counts[:, s:e, :] += 1
        else:
            sums[:, :, s:e] += pp[k]; counts[:, :, s:e] += 1
    avg = sums / counts.clamp(min=1)
    return F.interpolate(avg.unsqueeze(0), size=(H, W),
                         mode="bilinear", align_corners=False).squeeze(0)
