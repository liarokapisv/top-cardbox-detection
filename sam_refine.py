#!/usr/bin/env python3
"""Hybrid top-blank masker: MobileSAM proposals + depth-based selection.

The classical pipeline proposes candidate top groups; interior points of
those groups prompt MobileSAM, whose masks have robust visual boundaries.
Each SAM mask is then scored with DEPTH evidence (border topness voting),
the grey-bin colour prior and a rectangularity prior; the best mask wins.
"""
import numpy as np
import cv2
import json
import os

import pipeline as P

ABOVE_MARGIN = 0.0025
MIN_PHYS, MAX_PHYS = 0.010, 0.12
MAX_GREY = 0.45
IOU_NMS = 0.60
SAM_WEIGHTS = "sam2.1_t.pt"
GRID_STEP = 64
# a complete blank (all 4 faces visible — always true for the top box)
BLANK_LONG = (0.26, 0.375)     # m, long side of a full blank
BLANK_SHORT = (0.18, 0.30)     # m, short side of a full blank


def interior_points(mask, n=3, min_dist=45):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    pts = []
    for _ in range(n):
        idx = int(np.argmax(dt))
        y, x = divmod(idx, dt.shape[1])
        if dt[y, x] < 6:
            break
        pts.append([int(x), int(y)])
        cv2.circle(dt, (x, y), min_dist, 0, -1)
    return pts


def mask_topness(mask, d, valid, box_col):
    """Fraction of the mask boundary along which the mask is the nearer
    (upper) surface, judged in local windows."""
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    # local mean depths inside / outside the mask
    dm = d.copy(); dm[~(valid & mask)] = 0
    win = 11
    num_in = cv2.blur((valid & mask).astype(np.float32), (win, win))
    din = cv2.blur(dm, (win, win)) / np.maximum(num_in, 1e-6)
    out = valid & ~mask
    do_ = d.copy(); do_[~out] = 0
    num_out = cv2.blur(out.astype(np.float32), (win, win))
    dout = cv2.blur(do_, (win, win)) / np.maximum(num_out, 1e-6)

    above = below = 0
    for c in cnts:
        for p in c[::4, 0, :]:
            x, y = int(p[0]), int(p[1])
            if num_in[y, x] < 0.15 or num_out[y, x] < 0.15:
                continue
            if din[y, x] < dout[y, x] - ABOVE_MARGIN:
                above += 1
            elif dout[y, x] < din[y, x] - ABOVE_MARGIN:
                below += 1
    # only decisive segments vote: a border flush with a level neighbour
    # says nothing about stacking order and must not dilute the score
    tot = above + below
    return above / tot if tot >= 8 else 0.5


def score_mask(mask, d, valid, pts3, box_col, intr):
    mask = mask & valid
    area = int(mask.sum())
    if area < 2500:
        return None
    grey = 1.0 - float(box_col[mask].mean())
    if grey > MAX_GREY:
        return None
    phys = float((d[mask] ** 2).sum() / (intr["fx"] * intr["fy"]))
    if not (MIN_PHYS <= phys <= MAX_PHYS):
        return None
    p = pts3[mask]
    sub = p[np.random.RandomState(0).choice(len(p), min(len(p), 4000), replace=False)]
    c, n, rms = P.fit_plane(sub)
    tilt = np.degrees(np.arccos(min(1.0, abs(n[2]))))
    if tilt > P.MAX_TILT_DEG or rms > P.MAX_PLANE_RMS:
        return None
    # physical extent sanity: blanks are <= ~0.35 m per side; an oversized
    # or over-elongated plane rect means the mask spans several blanks
    # (measured with the same robust trimmed fit as the final tight box)
    (sa, sb), _, _ = tight_box(mask, d, valid, pts3, intr)
    side_a, side_b = sorted([sa, sb])
    if side_b > 0.38 or (side_a > 0.01 and side_b / side_a > 2.0):
        return None
    af = mask_topness(mask, d, valid, box_col)
    fill = 0.0
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        (rw, rh) = cv2.minAreaRect(np.vstack([q_.reshape(-1, 2) for q_ in cnts]))[1]
        fill = area / max(rw * rh, 1.0)
    score = phys * (af ** 3) * (0.5 + 0.5 * fill)
    return dict(mask=mask, area=area, phys=phys, grey=grey, tilt=tilt,
                rms=rms, above_frac=af, fill=fill, score=score,
                centroid=c, normal=n)


def tight_box(mask, d, valid, pts3, intr):
    mask0 = mask & valid
    p_all = pts3[mask0]
    rs = np.random.RandomState(0)
    sub = p_all[rs.choice(len(p_all), min(len(p_all), 6000), replace=False)]
    c, n, _ = P.fit_plane(sub)
    for _ in range(2):
        dist_ = np.abs((sub - c) @ n)
        keep = dist_ < P.REFINE_DIST
        if keep.sum() < 100:
            break
        c, n, _ = P.fit_plane(sub[keep])
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    q = p_all - c
    dist_ = np.abs(q @ n)
    q = q[dist_ < P.REFINE_DIST]
    uv = np.stack([q @ e1, q @ e2], axis=1)
    rect = cv2.minAreaRect((uv * 1000).astype(np.int32).reshape(-1, 1, 2))
    box_uv = cv2.boxPoints(rect) / 1000.0
    corners3d = c + box_uv[:, :1] * e1 + box_uv[:, 1:2] * e2
    size = (rect[1][0] / 1000.0, rect[1][1] / 1000.0)
    return size, P.project(corners3d, intr), n


def run(frames=None):
    from ultralytics import SAM
    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, "dataset", "two_lights")
    out_dir = os.path.join(base, "overlays_sam")
    mask_dir = os.path.join(base, "masks")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    intr = json.load(open(os.path.join(data, "intrinsics.json")))
    sam = SAM(os.path.join(base, SAM_WEIGHTS))

    import glob as g, time
    frames = frames or sorted(os.path.basename(f)[:3]
                              for f in g.glob(os.path.join(data, "*_rgb.png")))
    results = {}
    t0 = time.time()
    for f in frames:
        res = P.analyze(os.path.join(data, f + "_depth.npy"),
                        os.path.join(data, f + "_rgb.png"), intr)
        d, valid, pts3, box_col = res["d"], res["valid"], res["pts"], res["box_col"]
        rgb = res["rgb"]

        # prompts: interior points of the best-scoring classical groups
        groups = sorted([c for c in res["comps"].values() if c["score"] > 0],
                        key=lambda c: -c["score"])[:5]
        grp_prompts = []
        for c_ in groups:
            ip = interior_points(c_["mask"], n=3)
            if ip:
                grp_prompts.append((ip + [ip[-1]] * 3)[:3])
        pts_prompt = [p for gp in grp_prompts for p in gp]
        if not pts_prompt:                       # empty bin
            results[f] = {"found": False}
            ov = rgb.copy()
            cv2.putText(ov, "NO CANDIDATE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(out_dir, f + "_overlay.png"), ov)
            cv2.imwrite(os.path.join(mask_dir, f + "_mask.png"),
                        np.zeros(d.shape, np.uint8))
            print(f"{f}: no prompts (empty bin?)")
            continue

        img_path = os.path.join(data, f + "_rgb.png")
        # grouped prompts: SAM segments the object containing all 3 points
        # of a group -> whole-blank masks
        sr = sam(img_path, points=grp_prompts,
                 labels=[[1, 1, 1]] * len(grp_prompts),
                 device="cpu", verbose=False)
        sam_masks = list(sr[0].masks.data.cpu().numpy().astype(bool))
        # individual points + a coarse grid over box-coloured areas ->
        # per-panel masks and coverage of blanks the classical stage missed
        grid = []
        core = cv2.erode(box_col.astype(np.uint8), np.ones((9, 9), np.uint8))
        for gy in range(GRID_STEP // 2, core.shape[0], GRID_STEP):
            for gx in range(GRID_STEP // 2, core.shape[1], GRID_STEP):
                if core[gy, gx] and valid[gy, gx] and \
                   all((gx - p[0]) ** 2 + (gy - p[1]) ** 2 > 30 ** 2
                       for p in pts_prompt):
                    grid.append([gx, gy])
        singles = pts_prompt + grid[:36]
        sr2 = sam(img_path, points=singles, labels=[1] * len(singles),
                  device="cpu", verbose=False)
        sam_masks += list(sr2[0].masks.data.cpu().numpy().astype(bool))

        cands = []
        for sm in sam_masks:
            sc = score_mask(sm, d, valid, pts3, box_col, intr)
            if sc is not None:
                cands.append(sc)
        # classical winner joins only when SAM produced nothing usable —
        # SAM boundaries are strictly better delineated
        if not cands and res["best"] is not None:
            m0 = res["comps"][res["best"]].get("mask_refined",
                                               res["comps"][res["best"]]["mask"])
            sc = score_mask(m0, d, valid, pts3, box_col, intr)
            if sc is not None:
                sc["classical"] = True
                cands.append(sc)

        # NMS by IoU
        cands.sort(key=lambda c: -c["score"])
        kept = []
        for c_ in cands:
            dup = False
            for k in kept:
                inter = (c_["mask"] & k["mask"]).sum()
                union = (c_["mask"] | k["mask"]).sum()
                if union and inter / union > IOU_NMS:
                    dup = True
                    break
            if not dup:
                kept.append(c_)

        if not kept:
            results[f] = {"found": False}
            print(f"{f}: no valid candidate")
            continue
        # ---- the top box ALWAYS has all 4 faces visible, so the correct
        # answer is ALWAYS a complete blank. Try seeds in topness order and
        # grow each to blank size from connected, level, non-stacked pieces
        # (SAM candidates + classical groups); a seed that cannot reach
        # blank dimensions is the wrong object — move to the next. ----
        def rect_fill(mask_):
            mm_ = mask_.astype(np.uint8)
            cs, _ = cv2.findContours(mm_, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                return 0.0
            (rw, rh) = cv2.minAreaRect(
                np.vstack([q.reshape(-1, 2) for q in cs]))[1]
            return float(mask_.sum()) / max(rw * rh, 1.0)

        def blank_sized(sa, sb):
            lo, hi = sorted([sa, sb])
            return (BLANK_LONG[0] <= hi <= BLANK_LONG[1] and
                    BLANK_SHORT[0] <= lo <= BLANK_SHORT[1])

        # growable pieces: every scored candidate plus classical groups
        pieces = [k["mask"] for k in kept]
        for c_ in res["comps"].values():
            if c_["score"] > 0 and c_["grey_frac"] < 0.5:
                pieces.append(c_["mask"])

        def try_grow(seed_mask):
            wmask = seed_mask.copy()
            for _ in range(4):                     # growth rounds
                (sa, sb), _, _ = tight_box(wmask, d, valid, pts3, intr)
                if blank_sized(sa, sb):
                    return wmask, True
                grew = False
                for pm in pieces:
                    ov = (pm & wmask).sum()
                    if ov / max(pm.sum(), 1) > 0.5:
                        continue                   # duplicate
                    band = cv2.dilate(wmask.astype(np.uint8),
                                      np.ones((11, 11), np.uint8)).astype(bool) & pm
                    band2 = cv2.dilate(pm.astype(np.uint8),
                                       np.ones((11, 11), np.uint8)).astype(bool) & wmask
                    if band.sum() < 40 or band2.sum() < 40:
                        continue                   # not adjacent
                    if abs(float(np.median(d[band])) -
                           float(np.median(d[band2]))) > 0.004:
                        continue                   # depth step: other blank
                    pmv = pm & valid
                    if pmv.sum() > 400:
                        pa = pts3[pmv]
                        rs2 = np.random.RandomState(1)
                        ca, na, _ = P.fit_plane(pa[rs2.choice(
                            len(pa), min(len(pa), 2000), replace=False)])
                        wv = wmask & valid
                        pw = pts3[wv]
                        cw, nw, _ = P.fit_plane(pw[rs2.choice(
                            len(pw), min(len(pw), 3000), replace=False)])
                        ang = np.degrees(np.arccos(min(1.0, abs(float(na @ nw)))))
                        off = 0.5 * (abs(float((ca - cw) @ nw)) +
                                     abs(float((cw - ca) @ na)))
                        if ang < 10.0 and off > 0.008:
                            continue               # stacked parallel blank
                    trial = wmask | pm
                    (ta, tb), _, _ = tight_box(trial, d, valid, pts3, intr)
                    lo, hi = sorted([ta, tb])
                    if hi > BLANK_LONG[1] or lo > BLANK_SHORT[1]:
                        continue                   # would exceed a blank
                    if rect_fill(trial) < 0.50:
                        continue
                    wmask = trial
                    grew = True
                if not grew:
                    break
            (sa, sb), _, _ = tight_box(wmask, d, valid, pts3, intr)
            return wmask, blank_sized(sa, sb)

        kept.sort(key=lambda k: (-k["above_frac"], -k["phys"]))
        win, wmask, complete = kept[0], kept[0]["mask"], False
        for k in kept:
            gm, ok_ = try_grow(k["mask"])
            if ok_:
                win, wmask, complete = k, gm, True
                break
        if not complete:                            # best effort
            wmask, _ = try_grow(kept[0]["mask"])
            win = kept[0]
        win["complete"] = complete
        wm = wmask.astype(np.uint8)
        wm = cv2.morphologyEx(wm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n_cc, cc = cv2.connectedComponents(wm)
        if n_cc > 2:
            sizes = np.bincount(cc.ravel()); sizes[0] = 0
            wm = (cc == sizes.argmax()).astype(np.uint8)
        size, corners, nrm = tight_box(wm.astype(bool), d, valid, pts3, intr)

        cv2.imwrite(os.path.join(mask_dir, f + "_mask.png"), wm * 255)
        results[f] = {
            "found": True,
            "plane_rect_corners_px": corners.tolist(),
            "plane_rect_size_m": list(size),
            "plane_normal": nrm.tolist(),
            "above_frac": win["above_frac"],
            "grey_frac": win["grey"],
            "fill": win["fill"],
            "source": "classical" if win.get("classical") else "sam",
        }

        # ---- overlay ----
        h, w = rgb.shape[:2]
        p1 = rgb.copy()
        cnts, _ = cv2.findContours(wm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p1, cnts, -1, (0, 255, 255), 2)
        cv2.polylines(p1, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        txt = (f"{size[0]*100:.1f} x {size[1]*100:.1f} cm  top={win['above_frac']:.2f}"
               f"  [{'cls' if win.get('classical') else 'sam'}]"
               f"{'' if win.get('complete') else '  PARTIAL'}")
        cv2.putText(p1, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(p1, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        p2 = P.colorize_depth(d, valid)
        for x, y in pts_prompt:
            cv2.drawMarker(p2, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 12, 2)

        rng = np.random.RandomState(3)
        p3 = np.zeros_like(rgb)
        for k in kept[::-1]:
            p3[k["mask"]] = tuple(int(v) for v in rng.randint(60, 255, 3))
        for k in kept:
            ys, xs = np.nonzero(k["mask"])
            cv2.putText(p3, f"{k['above_frac']:.2f}", (int(xs.mean()) - 22, int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.drawContours(p3, cnts, -1, (255, 255, 255), 2)

        p4 = (rgb * 0.35).astype(np.uint8)
        mm = wm.astype(bool)
        p4[mm] = (0.35 * rgb[mm] + np.array([0, 160, 0])).clip(0, 255).astype(np.uint8)
        cv2.polylines(p4, [corners.astype(np.int32)], True, (0, 255, 0), 2)

        for p, t in [(p1, "result"), (p2, "depth + prompts"),
                     (p3, "candidates/topness"), (p4, "winner mask")]:
            cv2.putText(p, t, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
        grid = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
        cv2.imwrite(os.path.join(out_dir, f + "_overlay.png"), grid)
        print(f"{f}: cands={len(kept)} top={win['above_frac']:.2f} "
              f"box={size[0]*100:.1f}x{size[1]*100:.1f}cm "
              f"{'COMPLETE' if win.get('complete') else 'PARTIAL'}")
    with open(os.path.join(out_dir, "results_sam.json"), "w") as fp:
        json.dump(results, fp, indent=1)
    print(f"avg {(time.time()-t0)/len(frames):.1f} s/frame")


if __name__ == "__main__":
    import sys
    run(sys.argv[1:] or None)
