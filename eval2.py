#!/usr/bin/env python3
"""Real-frame evaluation with test-time augmentation (4x flip ensemble).

usage: python3 eval2.py <checkpoint> [ch24]
"""
import numpy as np
import torch
import cv2
import json
import glob
import os
import sys

from train import TinyUNet
from train2 import W, H

BW, BH = 640, 480


def predict_tta(net, rgb):
    x0 = cv2.resize(rgb, (W, H)).astype(np.float32) / 255.0
    outs = []
    with torch.no_grad():
        for fx in (False, True):
            for fy in (False, True):
                xi = x0[::-1] if fy else x0
                xi = xi[:, ::-1] if fx else xi
                t = torch.from_numpy(np.ascontiguousarray(
                    xi.transpose(2, 0, 1))[None])
                p = torch.sigmoid(net(t))[0, 0].numpy()
                if fx:
                    p = p[:, ::-1]
                if fy:
                    p = p[::-1]
                outs.append(p)
    return np.mean(outs, axis=0)


def main():
    ckpt = sys.argv[1]
    ch = (24, 48, 96, 192) if (len(sys.argv) > 2 and sys.argv[2] == "ch24") \
        else (16, 32, 64, 128)
    net = TinyUNet(ch=ch)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    lref = json.load(open("overlays_lfit/lfit_results.json"))
    HOLDOUT = ['003', '009', '013', '020', '025', '031', '034', '015']
    frames = sorted(os.path.basename(f)[:3]
                    for f in glob.glob("dataset/two_lights/*_rgb.png"))
    ious, hold_ious, tiles = [], [], []
    for f in frames:
        rgb = cv2.imread(f"dataset/two_lights/{f}_rgb.png")
        p = predict_tta(net, rgb)
        pm = cv2.resize(p, (BW, BH)) > 0.5
        pm8 = (pm * 255).astype(np.uint8)
        n, cc = cv2.connectedComponents((pm8 > 0).astype(np.uint8))
        if n > 2:
            sizes = np.bincount(cc.ravel()); sizes[0] = 0
            pm8 = ((cc == sizes.argmax()) * 255).astype(np.uint8)
        pm8 = cv2.morphologyEx(pm8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        vis = rgb.copy()
        ov = pm8 > 0
        vis[ov] = (0.5 * vis[ov] + np.array([0, 140, 0])).clip(0, 255).astype(np.uint8)
        txt = f
        if lref.get(f, {}).get("found"):
            gt = np.zeros((BH, BW), np.uint8)
            cv2.fillPoly(gt, [np.array(lref[f]["polygon_px"], np.int32)], 255)
            inter = ((gt > 0) & (pm8 > 0)).sum()
            uni = ((gt > 0) | (pm8 > 0)).sum()
            iou = inter / max(uni, 1)
            ious.append(iou)
            if f in HOLDOUT:
                hold_ious.append(iou)
            txt += f" IoU={iou:.2f}" + (" *" if f in HOLDOUT else "")
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        tiles.append(cv2.resize(vis, (320, 240)))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, 36, 6)]
    cv2.imwrite("preview/contact_sheet_net2.jpg", np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"ALL: mean {np.mean(ious):.3f} median {np.median(ious):.3f}")
    print(f"HOLDOUT(8): mean {np.mean(hold_ious):.3f}")


if __name__ == "__main__":
    main()
