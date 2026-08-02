#!/usr/bin/env python3
"""Evaluate the trained UNet on the REAL frames: predict the top-box mask,
fit the canonical L-polygon to the prediction (2D chamfer over rotation/
mirror/scale on the mask raster), compare with l_fit reference polygons."""
import numpy as np
import torch
import cv2
import json
import glob
import os

from train import TinyUNet, W, H

BW, BH = 640, 480


def fit_L_to_mask(mask):
    """2D image-space L fit to a binary mask (approx: no plane projection)."""
    import l_fit as LF
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    cnt = max(cs, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 3000:
        return None
    # scale template from metres to px via mask extent
    rect = cv2.minAreaRect(cnt)
    long_px = max(rect[1])
    best = None
    V = np.full(mask.shape, -0.4, np.float32)
    V[mask > 0] = 1.0
    for sc_m in (0.93, 1.0, 1.07):
        px_per_m = long_px / (LF.L_W * sc_m)
        for mirror in (False, True):
            base = LF.l_polygon(1.0, mirror) * px_per_m * sc_m
            for deg in range(0, 360, 4):
                th = np.deg2rad(deg)
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                poly = base @ R.T
                mn = poly.min(0)
                pp = (poly - mn).astype(np.int32)
                tw, thh = pp.max(0) + 1
                if tw >= mask.shape[1] or thh >= mask.shape[0]:
                    continue
                tmpl = np.zeros((thh, tw), np.uint8)
                cv2.fillPoly(tmpl, [pp], 1)
                r = cv2.matchTemplate(V, tmpl.astype(np.float32), cv2.TM_CCORR)
                _, mx, _, loc = cv2.minMaxLoc(r)
                score = mx / tmpl.sum()
                if best is None or score > best[0]:
                    best = (score, poly - mn + loc)
    return best


def main():
    net = TinyUNet()
    net.load_state_dict(torch.load("topbox_unet.pt", map_location="cpu"))
    net.eval()
    lref = json.load(open("overlays_lfit/lfit_results.json"))
    os.makedirs("overlays_net", exist_ok=True)
    frames = sorted(os.path.basename(f)[:3]
                    for f in glob.glob("dataset/two_lights/*_rgb.png"))
    ious, tiles = [], []
    import time
    t0 = time.time()
    for f in frames:
        rgb = cv2.imread(f"dataset/two_lights/{f}_rgb.png")
        x = torch.from_numpy(cv2.resize(rgb, (W, H)).astype(np.float32)
                             .transpose(2, 0, 1)[None] / 255.0)
        with torch.no_grad():
            p = torch.sigmoid(net(x))[0, 0].numpy()
        pm = cv2.resize(p, (BW, BH)) > 0.5
        pm8 = (pm * 255).astype(np.uint8)
        n, cc = cv2.connectedComponents((pm8 > 0).astype(np.uint8))
        if n > 2:
            sizes = np.bincount(cc.ravel()); sizes[0] = 0
            pm8 = ((cc == sizes.argmax()) * 255).astype(np.uint8)
        vis = rgb.copy()
        ov = pm8 > 0
        vis[ov] = (0.5 * vis[ov] + np.array([0, 140, 0])).clip(0, 255).astype(np.uint8)
        fit = fit_L_to_mask(pm8)
        iou_txt = ""
        if fit is not None:
            score, poly = fit
            cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 255, 0), 3)
        if lref.get(f, {}).get("found"):
            gt = np.zeros((BH, BW), np.uint8)
            cv2.fillPoly(gt, [np.array(lref[f]["polygon_px"], np.int32)], 255)
            inter = ((gt > 0) & (pm8 > 0)).sum()
            uni = ((gt > 0) | (pm8 > 0)).sum()
            iou = inter / max(uni, 1)
            ious.append(iou)
            iou_txt = f" IoU={iou:.2f}"
        cv2.putText(vis, f"{f}{iou_txt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(vis, f"{f}{iou_txt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(f"overlays_net/{f}_net.png", vis)
        tiles.append(cv2.resize(vis, (320, 240)))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, 36, 6)]
    cv2.imwrite("preview/contact_sheet_net.jpg", np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"mean IoU vs L-fit reference: {np.mean(ious):.3f} on {len(ious)} frames")
    print(f"median IoU: {np.median(ious):.3f}  min: {np.min(ious):.3f}")
    print(f"inference+fit avg {(time.time()-t0)/len(frames)*1000:.0f} ms/frame")


if __name__ == "__main__":
    main()
