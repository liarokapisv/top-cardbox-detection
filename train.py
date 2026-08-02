#!/usr/bin/env python3
"""Train a tiny UNet: RGB -> top-box mask, on the synthetic dataset."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import glob
import os
import time

torch.set_num_threads(4)
W, H = 256, 192


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.b1 = nn.BatchNorm2d(co)
        self.b2 = nn.BatchNorm2d(co)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(x)))


class TinyUNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128)):
        super().__init__()
        self.d1 = Block(3, ch[0])
        self.d2 = Block(ch[0], ch[1])
        self.d3 = Block(ch[1], ch[2])
        self.d4 = Block(ch[2], ch[3])
        self.u3 = Block(ch[3] + ch[2], ch[2])
        self.u2 = Block(ch[2] + ch[1], ch[1])
        self.u1 = Block(ch[1] + ch[0], ch[0])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(F.max_pool2d(s1, 2))
        s3 = self.d3(F.max_pool2d(s2, 2))
        x = self.d4(F.max_pool2d(s3, 2))
        x = self.u3(torch.cat([F.interpolate(x, scale_factor=2), s3], 1))
        x = self.u2(torch.cat([F.interpolate(x, scale_factor=2), s2], 1))
        x = self.u1(torch.cat([F.interpolate(x, scale_factor=2), s1], 1))
        return self.out(x)


def load_split(d):
    xs, ys = [], []
    for f in sorted(glob.glob(f"{d}/images/*.png")):
        name = os.path.basename(f)[:-4]
        img = cv2.resize(cv2.imread(f), (W, H))
        m = cv2.resize(cv2.imread(f"{d}/labels/{name}_topmask.png", 0), (W, H))
        xs.append(img)
        ys.append((m > 127).astype(np.float32))
    return np.stack(xs), np.stack(ys)


def dice_loss(logit, target):
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum((1, 2, 3)) + 1
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1
    return (1 - num / den).mean()


def main():
    xtr, ytr = load_split("synth_train")
    xva, yva = load_split("synth_val")
    print("train", xtr.shape, "val", xva.shape)
    net = TinyUNet()
    print("params", sum(p.numel() for p in net.parameters()) / 1e6, "M")
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    bs = 8
    rng = np.random.RandomState(0)
    for ep in range(15):
        t0 = time.time()
        net.train()
        idx = rng.permutation(len(xtr))
        tot = 0.0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = xtr[b].astype(np.float32)
            yb = ytr[b].copy()
            # augmentation: brightness/contrast + horizontal flip
            for j in range(len(b)):
                xb[j] = xb[j] * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
                if rng.rand() < 0.5:
                    xb[j] = xb[j, :, ::-1].copy()
                    yb[j] = yb[j, :, ::-1].copy()
            x = torch.from_numpy(xb.clip(0, 255).transpose(0, 3, 1, 2) / 255.0).float()
            y = torch.from_numpy(yb).unsqueeze(1)
            logit = net(x)
            loss = F.binary_cross_entropy_with_logits(logit, y) + dice_loss(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        # val IoU
        net.eval()
        ious = []
        with torch.no_grad():
            for i in range(0, len(xva), bs):
                x = torch.from_numpy(
                    xva[i:i + bs].astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
                p = torch.sigmoid(net(x)).numpy()[:, 0] > 0.5
                for j in range(len(p)):
                    gt = yva[i + j] > 0.5
                    inter = (p[j] & gt).sum()
                    uni = (p[j] | gt).sum()
                    ious.append(inter / max(uni, 1))
        print(f"ep {ep}: loss {tot/len(xtr):.4f}  val IoU {np.mean(ious):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(net.state_dict(), "topbox_unet.pt")
    print("saved topbox_unet.pt")


if __name__ == "__main__":
    main()
