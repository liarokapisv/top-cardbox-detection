#!/usr/bin/env python3
"""Improved trainer: bigger UNet, synth + real-background composites,
rotation-heavy augmentation, resumable in chunks. Uses CUDA automatically
when available (AMP mixed precision + larger batches); falls back to CPU.

usage: python3 train2.py <epochs> [resume]
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import glob
import os
import sys
import time

from train import TinyUNet, dice_loss

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
W, H = 256, 192
CKPT = "topbox_unet2.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = int(os.environ.get("BATCH", "32" if DEVICE.type == "cuda" else "8"))
USE_AMP = DEVICE.type == "cuda"


def load_split(dirs):
    xs, ys = [], []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/images/*.png")):
            name = os.path.basename(f)[:-4]
            img = cv2.resize(cv2.imread(f), (W, H))
            m = cv2.resize(cv2.imread(f"{d}/labels/{name}_topmask.png", 0), (W, H))
            xs.append(img)
            ys.append((m > 127).astype(np.float32))
    return np.stack(xs), np.stack(ys)


def augment(xb, yb, rng):
    for j in range(len(xb)):
        # full random rotation
        ang = rng.uniform(0, 360)
        Mr = cv2.getRotationMatrix2D((W / 2, H / 2), ang, rng.uniform(0.9, 1.1))
        xb[j] = cv2.warpAffine(xb[j], Mr, (W, H), flags=cv2.INTER_LINEAR)
        yb[j] = cv2.warpAffine(yb[j], Mr, (W, H), flags=cv2.INTER_NEAREST)
        if rng.rand() < 0.5:
            xb[j] = xb[j][:, ::-1].copy()
            yb[j] = yb[j][:, ::-1].copy()
        # photometric
        g = rng.uniform(0.75, 1.25)
        xb[j] = (xb[j].astype(np.float32) * g + rng.uniform(-18, 18))
        if rng.rand() < 0.3:
            xb[j] += rng.normal(0, 6, xb[j].shape)
        if rng.rand() < 0.2:
            xb[j] = cv2.GaussianBlur(xb[j], (3, 3), 0)
    return xb, yb


def main():
    n_ep = int(sys.argv[1])
    resume = len(sys.argv) > 2 and sys.argv[2] == "resume"
    xtr, ytr = load_split(["synth_train", "synth_rbg_train"])
    xva, yva = load_split(["synth_val", "synth_rbg_val"])
    print("train", xtr.shape, "val", xva.shape, flush=True)
    net = TinyUNet(ch=(24, 48, 96, 192)).to(DEVICE)
    if resume and os.path.exists(CKPT):
        net.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        print("resumed", flush=True)
    print(f"device {DEVICE} batch {BATCH} amp {USE_AMP} params "
          f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=1.5e-3 if not resume else 6e-4)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    bs = BATCH
    rng = np.random.RandomState(int(time.time()) % 9999)
    for ep in range(n_ep):
        t0 = time.time()
        net.train()
        idx = rng.permutation(len(xtr))
        tot = 0.0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = xtr[b].astype(np.float32).copy()
            yb = ytr[b].copy()
            xb, yb = augment(xb, yb, rng)
            x = torch.from_numpy(xb.clip(0, 255).transpose(0, 3, 1, 2) / 255.0) \
                .float().to(DEVICE, non_blocking=True)
            y = torch.from_numpy(np.ascontiguousarray(yb)).unsqueeze(1).to(DEVICE)
            with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                logit = net(x)
                loss = F.binary_cross_entropy_with_logits(logit, y) + \
                    dice_loss(logit, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(b)
        net.eval()
        ious = []
        with torch.no_grad():
            for i in range(0, len(xva), bs):
                x = torch.from_numpy(
                    xva[i:i + bs].astype(np.float32).transpose(0, 3, 1, 2)
                    / 255.0).float().to(DEVICE)
                with torch.amp.autocast(DEVICE.type, enabled=USE_AMP):
                    p = torch.sigmoid(net(x)).float().cpu().numpy()[:, 0] > 0.5
                for j in range(len(p)):
                    gt = yva[i + j] > 0.5
                    ious.append((p[j] & gt).sum() / max((p[j] | gt).sum(), 1))
        print(f"ep {ep}: loss {tot/len(xtr):.4f} val IoU {np.mean(ious):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(net.state_dict(), CKPT)


if __name__ == "__main__":
    main()
