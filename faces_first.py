#!/usr/bin/env python3
"""Face-first top-blank detector (CPU-only, no learned model at runtime).

A blank is 4 connected, relatively flat faces; every face of the TOP blank
is fully visible, while lower blanks always have faces truncated by
occlusion. So instead of segmenting whole blanks directly:

  1. detect FACE candidates: colour-class regions (white / green / brown —
     the bin is grey) split by depth-jump edges, each validated as a face
     (planar, rectangular, face-sized)
  2. link faces that are adjacent, depth-continuous at the border and NOT
     parallel-offset (stacked) -> face graph
  3. grow box hypotheses inside the graph, bounded by the physical extent
     of one blank; a hypothesis is COMPLETE when its union is blank-sized
     and rectangular
  4. the top blank = the complete hypothesis whose border topness is best
"""
import numpy as np
import cv2
import json
import os

import pipeline as P

# face validation
MIN_FACE_PX = 350
MAX_FACE_SIDE = 0.36        # m
MAX_FACE_RMS = 0.016        # m
MIN_FACE_FILL = 0.40
# face linking
LINK_GAP = 0.0045           # m, border depth continuity
STACK_ANG = 12.0            # deg
STACK_OFF = 0.008           # m
MIN_BORDER = 25             # px
# box hypothesis
MAX_BOX_SIDE = 0.38         # m
MIN_BOX_PHYS = 0.030        # m^2 (a lone face is ~0.015-0.03)
MIN_BOX_FILL = 0.52
ABOVE_MARGIN = 0.0025


def color_classes(rgb):
    hsv = cv2.cvtColor(cv2.medianBlur(rgb, 5), cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    green = (S > 60) & (H >= 35) & (H <= 95)
    brown = (S > 60) & (H < 35)
    white = (V > 180) & (S < 90) & ~green & ~brown
    cls = np.zeros(H.shape, np.uint8)          # 0 = grey/bin
    cls[white] = 1
    cls[green] = 2
    cls[brown] = 3
    cls = cv2.medianBlur(cls, 5)
    return cls


def plane_rect(mask, pts3, valid, trim=True):
    m = mask & valid
    if m.sum() < 200:
        return None
    p = pts3[m]
    rs = np.random.RandomState(0)
    sub = p[rs.choice(len(p), min(len(p), 4000), replace=False)]
    c, n, rms = P.fit_plane(sub)
    if trim:
        for _ in range(2):
            keep = np.abs((sub - c) @ n) < 0.012
            if keep.sum() < 100:
                break
            c, n, rms = P.fit_plane(sub[keep])
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    q = p - c
    q = q[np.abs(q @ n) < 0.015]
    if len(q) < 100:
        return None
    uv = np.stack([q @ e1, q @ e2], axis=1)
    rect = cv2.minAreaRect((uv * 1000).astype(np.int32).reshape(-1, 1, 2))
    sa, sb = sorted([rect[1][0] / 1000.0, rect[1][1] / 1000.0])
    area_uv = len(q)                       # px count ~ area proxy for fill
    box_uv = cv2.boxPoints(rect) / 1000.0
    corners3d = c + box_uv[:, :1] * e1 + box_uv[:, 1:2] * e2
    # fill in plane coords: fraction of rect covered by points
    cell = max(sa * sb, 1e-6)
    px_area = None
    return dict(c=c, n=n, rms=rms, side_a=sa, side_b=sb,
                corners3d=corners3d, e1=e1, e2=e2, rect=rect)


def mask_phys(mask, d, intr):
    return float((d[mask] ** 2).sum() / (intr["fx"] * intr["fy"]))


def rect_fill_img(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 0.0
    (rw, rh) = cv2.minAreaRect(np.vstack([q.reshape(-1, 2) for q in cs]))[1]
    return float(mask.sum()) / max(rw * rh, 1.0)


def border_topness(mask, d, valid):
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.5
    win = 11
    dm = d.copy(); dm[~(valid & mask)] = 0
    num_in = cv2.blur((valid & mask).astype(np.float32), (win, win))
    din = cv2.blur(dm, (win, win)) / np.maximum(num_in, 1e-6)
    out = valid & ~mask
    do_ = d.copy(); do_[~out] = 0
    num_out = cv2.blur(out.astype(np.float32), (win, win))
    dout = cv2.blur(do_, (win, win)) / np.maximum(num_out, 1e-6)
    above = below = 0
    for c in cnts:
        for pxy in c[::4, 0, :]:
            x, y = int(pxy[0]), int(pxy[1])
            if num_in[y, x] < 0.15 or num_out[y, x] < 0.15:
                continue
            if din[y, x] < dout[y, x] - ABOVE_MARGIN:
                above += 1
            elif dout[y, x] < din[y, x] - ABOVE_MARGIN:
                below += 1
    tot = above + below
    return above / tot if tot >= 8 else 0.5


def analyze_frame(f, data, intr):
    depth = np.load(os.path.join(data, f + "_depth.npy"))
    rgb = cv2.imread(os.path.join(data, f + "_rgb.png"))
    valid = (depth > P.VALID_MIN) & (depth < P.VALID_MAX)
    d = depth.copy(); d[~valid] = 0
    d = cv2.medianBlur(d, 5); d = cv2.medianBlur(d, 5)
    valid = valid & (d > 0)
    pts3 = P.deproject(d, intr)

    # depth-jump edges split same-colour faces of different boxes; normal
    # creases split faces meeting at an angle even when the step is tiny
    # (e.g. a blank resting on glare-bright floor)
    k = np.ones((5, 5), np.uint8)
    dmax = cv2.dilate(d, k)
    dinf = d.copy(); dinf[~valid] = 10.0
    dmin = cv2.erode(dinf, k)
    jump = ((dmax - dmin) > P.JUMP_THR * np.clip(d / 0.45, 1.0, 4.0) ** 2) & valid
    nrm = P.normals_from_points(pts3)
    nrm_s = cv2.blur(nrm, (11, 11))
    nl = np.linalg.norm(nrm_s, axis=2, keepdims=True); nl[nl == 0] = 1
    nrm_s /= nl
    dot = np.clip((nrm * nrm_s).sum(axis=2), -1, 1)
    crease = (np.degrees(np.arccos(dot)) > P.CREASE_DEG) & valid
    jump = jump | crease

    cls = color_classes(rgb)

    # RGB edges: the only cue separating same-colour flush contacts between
    # different boxes is the printed boundary line
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 30, 5)
    canny = cv2.dilate(cv2.Canny(gray, 40, 110),
                       np.ones((3, 3), np.uint8)).astype(bool)
    jump = jump | (canny & (cls > 0))

    # ---- face candidates ----
    faces = []
    for ci in (1, 2, 3):
        cm = (cls == ci) & valid & ~jump
        cm = cv2.morphologyEx(cm.astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
        cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n_lbl, lbl = cv2.connectedComponents(cm, connectivity=4)
        for l in range(1, n_lbl):
            m = lbl == l
            if m.sum() < MIN_FACE_PX:
                continue
            pr = plane_rect(m, pts3, valid)
            if pr is None or pr["rms"] > MAX_FACE_RMS:
                continue
            if pr["side_b"] > MAX_FACE_SIDE:
                continue
            fill = rect_fill_img(m)
            if fill < MIN_FACE_FILL:
                continue
            tilt = np.degrees(np.arccos(min(1.0, abs(pr["n"][2]))))
            if tilt > P.MAX_TILT_DEG:
                continue
            faces.append(dict(mask=m, cls=ci, pr=pr, fill=fill,
                              phys=mask_phys(m, d, intr)))

    result = dict(rgb=rgb, d=d, valid=valid, pts3=pts3, faces=faces,
                  cls=cls, jump=jump)
    if not faces:
        result["box"] = None
        return result

    # ---- face graph: adjacency + depth continuity + not stacked ----
    nf = len(faces)
    dil = [cv2.dilate(fc["mask"].astype(np.uint8),
                      np.ones((11, 11), np.uint8)).astype(bool) for fc in faces]
    bboxes = []
    for fc in faces:
        ys, xs = np.nonzero(fc["mask"])
        bboxes.append((xs.min() - 8, ys.min() - 8, xs.max() + 8, ys.max() + 8))
    link = np.zeros((nf, nf), bool)
    for i in range(nf):
        for j in range(i + 1, nf):
            bi, bj = bboxes[i], bboxes[j]
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue
            band_ij = dil[i] & faces[j]["mask"]
            band_ji = dil[j] & faces[i]["mask"]
            if band_ij.sum() < MIN_BORDER or band_ji.sum() < MIN_BORDER:
                continue
            gap = abs(float(np.median(d[band_ji])) - float(np.median(d[band_ij])))
            if gap > LINK_GAP:
                continue
            ni, nj = faces[i]["pr"]["n"], faces[j]["pr"]["n"]
            ci_, cj_ = faces[i]["pr"]["c"], faces[j]["pr"]["c"]
            ang = np.degrees(np.arccos(min(1.0, abs(float(ni @ nj)))))
            off = 0.5 * (abs(float((cj_ - ci_) @ ni)) + abs(float((ci_ - cj_) @ nj)))
            if ang < STACK_ANG and off > STACK_OFF:
                continue
            link[i, j] = link[j, i] = True

    # ---- box hypotheses: grow unions inside the graph, extent-bounded ----
    tops = [border_topness(fc["mask"], d, valid) for fc in faces]
    hyps = []
    order = np.argsort([-(t * faces[i]["phys"]) for i, t in enumerate(tops)])
    used_seeds = set()
    for seed in order:
        if seed in used_seeds:
            continue
        members = {int(seed)}
        mask = faces[seed]["mask"].copy()
        grown = True
        while grown:
            grown = False
            for j in range(nf):
                if j in members or not any(link[i, j] for i in members):
                    continue
                trial = mask | faces[j]["mask"]
                pr = plane_rect(trial, pts3, valid)
                if pr is None or pr["side_b"] > MAX_BOX_SIDE:
                    continue
                if rect_fill_img(trial) < MIN_BOX_FILL - 0.08:
                    continue
                mask = trial
                members.add(j)
                grown = True
        used_seeds |= members
        pr = plane_rect(mask, pts3, valid)
        if pr is None:
            continue
        phys = mask_phys(mask, d, intr)
        fill = rect_fill_img(mask)
        top = border_topness(mask, d, valid)
        complete = (phys >= MIN_BOX_PHYS and fill >= MIN_BOX_FILL and
                    pr["side_b"] <= MAX_BOX_SIDE)
        hyps.append(dict(members=sorted(members), mask=mask, pr=pr, phys=phys,
                         fill=fill, top=top, complete=complete,
                         n_faces=len(members)))

    result["hyps"] = hyps
    pool = [h for h in hyps if h["complete"]]
    if pool:
        t_max = max(h["top"] for h in pool)
        pool2 = [h for h in pool if h["top"] >= t_max - 0.03]
        result["box"] = max(pool2, key=lambda h: h["phys"])
    elif hyps:
        result["box"] = max(hyps, key=lambda h: h["top"] * h["phys"])
    else:
        result["box"] = None
    return result


FACE_COLS = [(80, 200, 80), (80, 130, 255), (255, 170, 60), (210, 80, 210),
             (80, 220, 220), (170, 170, 255), (140, 220, 140), (255, 120, 120)]


def overlay(f, res, intr):
    rgb = res["rgb"]
    h, w = rgb.shape[:2]
    box = res["box"]

    p1 = rgb.copy()
    if box is not None:
        cor = P.project(box["pr"]["corners3d"], intr).astype(np.int32)
        cs, _ = cv2.findContours(box["mask"].astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p1, cs, -1, (0, 255, 255), 2)
        cv2.polylines(p1, [cor], True, (0, 255, 0), 3)
        txt = (f"{box['pr']['side_b']*100:.1f} x {box['pr']['side_a']*100:.1f} cm "
               f"faces={box['n_faces']} top={box['top']:.2f}"
               f"{'' if box['complete'] else ' PARTIAL'}")
        cv2.putText(p1, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(p1, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    else:
        cv2.putText(p1, "NO BOX", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # panel 2: all face candidates
    p2 = (rgb * 0.3).astype(np.uint8)
    for i, fc in enumerate(res["faces"]):
        col = np.array(FACE_COLS[i % len(FACE_COLS)], np.float32)
        mm = fc["mask"]
        p2[mm] = (0.4 * p2[mm] + 0.6 * col).astype(np.uint8)
    for i, fc in enumerate(res["faces"]):
        ys, xs = np.nonzero(fc["mask"])
        cv2.putText(p2, f"{i}", (int(xs.mean()) - 5, int(ys.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # panel 3: winner's faces coloured individually
    p3 = (rgb * 0.3).astype(np.uint8)
    if box is not None:
        for k, i in enumerate(box["members"]):
            col = np.array(FACE_COLS[k % len(FACE_COLS)], np.float32)
            mm = res["faces"][i]["mask"]
            p3[mm] = (0.35 * p3[mm] + 0.65 * col).astype(np.uint8)
            cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p3, cs, -1, (255, 255, 255), 1)
            pr = res["faces"][i]["pr"]
            ys, xs = np.nonzero(mm)
            cv2.putText(p3, f"F{k+1} {pr['side_b']*100:.0f}x{pr['side_a']*100:.0f}",
                        (int(xs.mean()) - 35, int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # panel 4: colour classes + jump edges
    p4 = np.zeros_like(rgb)
    p4[res["cls"] == 1] = (230, 230, 230)
    p4[res["cls"] == 2] = (60, 170, 60)
    p4[res["cls"] == 3] = (60, 110, 180)
    p4[res["jump"]] = (0, 0, 255)

    for p, t in [(p1, "result"), (p2, "face candidates"),
                 (p3, "winner faces"), (p4, "colour classes + depth jumps")]:
        cv2.putText(p, t, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
    return np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])


if __name__ == "__main__":
    import sys, glob, time
    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, "dataset", "two_lights")
    out = os.path.join(base, "overlays_faces")
    os.makedirs(out, exist_ok=True)
    intr = json.load(open(os.path.join(data, "intrinsics.json")))
    frames = sys.argv[1:] or sorted(
        os.path.basename(x)[:3] for x in glob.glob(os.path.join(data, "*_rgb.png")))
    t0 = time.time()
    stats = {}
    for f in frames:
        res = analyze_frame(f, data, intr)
        cv2.imwrite(os.path.join(out, f + "_overlay.png"), overlay(f, res, intr))
        b = res["box"]
        if b is None:
            stats[f] = {"found": False}
            print(f"{f}: NO BOX ({len(res['faces'])} faces)")
        else:
            stats[f] = {"found": True, "complete": bool(b["complete"]),
                        "n_faces": b["n_faces"], "top": round(b["top"], 2),
                        "size_cm": [round(b["pr"]["side_b"] * 100, 1),
                                    round(b["pr"]["side_a"] * 100, 1)]}
            print(f"{f}: faces={len(res['faces'])} box_faces={b['n_faces']} "
                  f"top={b['top']:.2f} fill={b['fill']:.2f} "
                  f"box={b['pr']['side_b']*100:.1f}x{b['pr']['side_a']*100:.1f}cm"
                  f"{'' if b['complete'] else ' PARTIAL'}")
    with open(os.path.join(out, "faces_results.json"), "w") as fp:
        json.dump(stats, fp, indent=1)
    print(f"avg {(time.time()-t0)/len(frames)*1000:.0f} ms/frame")
