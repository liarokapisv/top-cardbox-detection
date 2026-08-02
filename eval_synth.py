#!/usr/bin/env python3
"""Standalone synthetic-val evaluator for the mask UNet.

Loads a checkpoint, runs the frozen synth val split (synth_val +
synth_rbg_val), reports per-image IoU with and without 4x flip TTA and
largest-connected-component cleanup, writes results/synth_val_eval.json.

usage: python3 eval_synth.py [checkpoint=topbox_unet2_best.pt]
"""
import numpy as np
import torch
import cv2
import glob
import json
import os
import sys

from train import TinyUNet

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "12")))
W, H = 256, 192
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(dirs):
    xs, ys, names = [], [], []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/images/*.png")):
            name = os.path.basename(f)[:-4]
            img = cv2.resize(cv2.imread(f), (W, H))
            m = cv2.resize(cv2.imread(f"{d}/labels/{name}_topmask.png", 0), (W, H))
            xs.append(img)
            ys.append(m > 127)
            names.append(f"{d}/{name}")
    return np.stack(xs), np.stack(ys), names


def predict_probs(net, xs, tta, bs=16):
    probs = []
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            xb = xs[i:i + bs].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            x = torch.from_numpy(xb).float().to(DEVICE)
            p = torch.sigmoid(net(x))
            if tta:
                p = p + torch.sigmoid(net(torch.flip(x, [3]))).flip([3])
                p = p + torch.sigmoid(net(torch.flip(x, [2]))).flip([2])
                p = p + torch.sigmoid(net(torch.flip(x, [2, 3]))).flip([2, 3])
                p = p / 4
            probs.append(p.cpu().numpy()[:, 0])
    return np.concatenate(probs)


def largest_cc(mask):
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 2:
        return mask
    sizes = [(lab == k).sum() for k in range(1, n)]
    return lab == (1 + int(np.argmax(sizes)))


def iou(p, gt):
    return (p & gt).sum() / max((p | gt).sum(), 1)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "topbox_unet2_best.pt"
    xs, ys, names = load_split(["synth_val", "synth_rbg_val"])
    print(f"val {xs.shape} ckpt {ckpt}", flush=True)
    net = TinyUNet(ch=(24, 48, 96, 192)).to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    net.eval()

    report = {"checkpoint": ckpt, "n": len(xs), "variants": {}}
    for tta in (False, True):
        probs = predict_probs(net, xs, tta)
        for lcc in (False, True):
            ious = []
            for j in range(len(xs)):
                p = probs[j] > 0.5
                if lcc:
                    p = largest_cc(p)
                ious.append(float(iou(p, ys[j])))
            key = f"tta={int(tta)},lcc={int(lcc)}"
            report["variants"][key] = {
                "mean": float(np.mean(ious)),
                "median": float(np.median(ious)),
                "min": float(np.min(ious)),
                "per_image": {n: round(v, 4) for n, v in zip(names, ious)},
            }
            print(f"{key}: mean {np.mean(ious):.4f} median "
                  f"{np.median(ious):.4f} min {np.min(ious):.4f}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/synth_val_eval.json", "w") as f:
        json.dump(report, f, indent=1)
    print("wrote results/synth_val_eval.json", flush=True)


if __name__ == "__main__":
    main()
