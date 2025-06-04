import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import lpips
import math
from skimage.metrics import structural_similarity as ssim_np
import torch.nn.functional as F
# Set paths
secret_folder = "output/secret"
revealed_folder = "output/revealed"

# LPIPS model
lpips_model = lpips.LPIPS(net='alex').to("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Metric function
def compute_metrics(img1, img2):
    eps = 1e-8
    # Resize img2 to match img1 if needed
    if img1.shape != img2.shape:
        img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
    mse_val = torch.mean((img1 - img2) ** 2).item()
    psnr_val = 20 * math.log10(1.0) - 10 * math.log10(mse_val + eps)

    img1_np = img1.squeeze().cpu().permute(1, 2, 0).numpy()
    img2_np = img2.squeeze().cpu().permute(1, 2, 0).numpy()
    ssim_val = ssim_np(img1_np, img2_np, channel_axis=2, data_range=1.0)

    img1_lp = (img1 * 2 - 1).to(device)
    img2_lp = (img2 * 2 - 1).to(device)
    lpips_val = lpips_model(img1_lp, img2_lp).item()

    img1_gray = np.mean(img1_np, axis=2)
    img2_gray = np.mean(img2_np, axis=2)
    s = img1_gray.flatten()
    r = img2_gray.flatten()

    ncc = np.sum((s - s.mean()) * (r - r.mean())) / (np.sqrt(np.sum((s - s.mean())**2)) * np.sqrt(np.sum((r - r.mean())**2)) + eps)
    ad = np.mean(s - r)
    sc = np.sum(s**2) / (np.sum(r**2) + eps)
    mid = np.mean(s - r)
    nae = np.sum(np.abs(s - r)) / (np.sum(np.abs(s)) + eps)
    rmse = np.sqrt(mse_val)
    cov = np.mean((s - s.mean()) * (r - r.mean()))
    uqi = (4 * cov * s.mean() * r.mean()) / ((np.var(s) + np.var(r) + eps) * (s.mean()**2 + r.mean()**2 + eps))

    s_bin = (img1_gray * 255).astype(np.uint8)
    r_bin = (img2_gray * 255).astype(np.uint8)
    diff = s_bin != r_bin
    npcr = np.sum(diff) / diff.size
    uaci = np.mean(np.abs(s_bin - r_bin)) / 255.0
    baci = np.mean(np.abs(np.unpackbits(s_bin.flatten()) - np.unpackbits(r_bin.flatten())))

    return {
        "mse": mse_val, "psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val,
        "ncc": ncc, "ad": ad, "sc": sc, "mid": mid, "nae": nae, "rmse": rmse,
        "uqi": uqi, "npcr": npcr, "uaci": uaci, "baci": baci
    }

import re

# ===== Gather files and extract indices =====
def extract_index(name):
    match = re.search(r'(\d+)', name)
    return match.group(1) if match else None

secret_files = sorted(os.listdir(secret_folder))
revealed_files = sorted(os.listdir(revealed_folder))

# Create lookup by index
secret_dict = {extract_index(f): f for f in secret_files if extract_index(f)}
revealed_dict = {extract_index(f): f for f in revealed_files if extract_index(f)}

# ===== Evaluate Matching Pairs =====
all_metrics = []

for idx in sorted(secret_dict.keys()):
    if idx in revealed_dict:
        secret_path = os.path.join(secret_folder, secret_dict[idx])
        revealed_path = os.path.join(revealed_folder, revealed_dict[idx])

        secret_img = transform(Image.open(secret_path).convert("RGB")).unsqueeze(0).to(device)
        revealed_img = transform(Image.open(revealed_path).convert("RGB")).unsqueeze(0).to(device)

        metrics = compute_metrics(revealed_img, secret_img)
        all_metrics.append(metrics)

        print(f"[img_{idx}] MSE={metrics['mse']:.4f}, PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, LPIPS={metrics['lpips']:.4f}")
    else:
        print(f"Skipping img_{idx}: No matching revealed image found.")

# ===== Final Average Metrics =====
if not all_metrics:
    print("\n❌ No valid image pairs found. Check your folder or filename pattern.")
    exit()

print("\n=== Average Metrics Across All Images ===")
for key in all_metrics[0]:
    avg = np.mean([m[key] for m in all_metrics])
    print(f"{key.upper():<6}: {avg:.4f}")

