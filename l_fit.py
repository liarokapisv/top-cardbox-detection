#!/usr/bin/env python3
"""Top blank detection by fitting the canonical 6-point L-polygon.

Every blank is the SAME specific shape: a 6-vertex L-polygon (outer
~31 x 24 cm rectangle with a ~15 x 10.5 cm notch at one corner; measured
from the clean frames). The top blank is the pose (position, rotation,
mirror, slight scale) where the COMPLETE L is covered by on-plane
box-coloured pixels and nothing lies above it. Fragments cannot fake a
full L and over-merged masks cannot fit inside one.

CPU-only, no learned model: ~2-4 s/frame.
"""
import numpy as np
import cv2
import json
import os

import pipeline as P

# canonical 8-gon (metres), calibrated on the single-box frames:
# 3 faces, 2 edge-revolute hinges. Top flap F1 (flush left), full-width
# middle panel M, bottom flap F3 (flush right). The top flap's visible
# height H1 is a FOLD STATE (often folded under -> 0, degenerating to
# the 6-point L).
L_W, L_H = 0.292, 0.220        # outer footprint (H at h1=0)
F1_W = 0.120                   # top flap width
F3_W, F3_H = 0.136, 0.100      # bottom flap
H1_STATES = (0.0, 0.022, 0.045)
# kept for backward compat with older callers
L_NW, L_NH = L_W - F3_W, F3_H
SCALES = (0.93, 1.0, 1.07)
ROT_STEP = 3                   # degrees
RES = 0.004                    # m per raster pixel
PLANE_TOL = 0.013              # m, "on plane"
ABOVE_TOL = 0.018              # m, occluder threshold
ACCEPT = 0.55                  # normalized fit score to accept
W_ABOVE = -2.0                 # penalty weights
W_NEUTRAL = -0.30


def l_polygon(scale=1.0, mirror=False, h1=0.0):
    """8-point boundary of the 3-face blank (h1=0 degenerates to 6-pt L)."""
    W = L_W * scale
    H = (L_H + h1) * scale
    w1, w3, h3 = F1_W * scale, F3_W * scale, F3_H * scale
    h1s = h1 * scale
    pts = np.array([(0, 0), (w1, 0), (w1, h1s), (W, h1s),
                    (W, H), (W - w3, H), (W - w3, H - h3), (0, H - h3)],
                   np.float64)
    pts -= [W / 2, H / 2]
    if mirror:
        pts[:, 0] *= -1
        pts = pts[::-1]
    return pts


def l_faces(scale=1.0, mirror=False, h1=0.0):
    """The 3 faces as corner quads (canonical, centred): [F1, M(middle), F3].
    Hinges: F1-M at y=h1, M-F3 at y=H-h3 — 2 edge revolute joints."""
    W = L_W * scale
    H = (L_H + h1) * scale
    w1, w3, h3 = F1_W * scale, F3_W * scale, F3_H * scale
    h1s = h1 * scale
    F1 = np.array([(0, 0), (w1, 0), (w1, h1s), (0, h1s)], np.float64)
    M = np.array([(0, h1s), (W, h1s), (W, H - h3), (0, H - h3)], np.float64)
    F3 = np.array([(W - w3, H - h3), (W, H - h3), (W, H), (W - w3, H)],
                  np.float64)
    out = []
    for q in (F1, M, F3):
        q = q - [W / 2, H / 2]
        if mirror:
            q[:, 0] *= -1
            q = q[::-1]
        out.append(q)
    return out            # [F1, M(middle), F3]


def make_templates():
    tmpls = []
    for scale in SCALES:
        for mirror in (False, True):
            for h1 in H1_STATES:
                base = l_polygon(scale, mirror, h1)
                for deg in range(0, 360, ROT_STEP):
                    th = np.deg2rad(deg)
                    R = np.array([[np.cos(th), -np.sin(th)],
                                  [np.sin(th), np.cos(th)]])
                    poly = base @ R.T
                    mn = poly.min(0)
                    px = ((poly - mn) / RES).astype(np.int32)
                    w, h = px.max(0) + 1
                    img = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(img, [px], 1)
                    tmpls.append(dict(img=img, poly=poly, deg=deg, scale=scale,
                                      mirror=mirror, h1=h1,
                                      area=int(img.sum()), off=mn))
    return tmpls


def fit_frame(res, intr, tmpls):
    d, valid, pts3, box_col = res["d"], res["valid"], res["pts"], res["box_col"]
    seeds = sorted([c for c in res["comps"].values() if c["score"] > 0],
                   key=lambda c: -(c["above_frac"] ** 2) * c["area"])[:3]
    best = None
    for sd in seeds:
        m = sd["mask"] & valid
        if m.sum() < 1500:
            continue
        p = pts3[m]
        rs = np.random.RandomState(0)
        c, n, _ = P.fit_plane(p[rs.choice(len(p), min(len(p), 5000), replace=False)])
        for _ in range(2):
            keep = np.abs((p - c) @ n) < 0.012
            if keep.sum() < 200:
                break
            c, n, _ = P.fit_plane(p[keep][rs.choice(
                int(keep.sum()), min(int(keep.sum()), 5000), replace=False)])
        ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(n, ref); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)

        # project every valid pixel into plane coords
        pv = pts3[valid]
        q = pv - c
        s = q @ n                       # signed height above plane
        u = q @ e1
        v = q @ e2
        roi = 0.34
        cs_u, cs_v = np.median(u[np.abs(s) < PLANE_TOL]), np.median(v[np.abs(s) < PLANE_TOL])
        sel = (np.abs(u - cs_u) < roi) & (np.abs(v - cs_v) < roi)
        u_, v_, s_ = u[sel] - cs_u + roi, v[sel] - cs_v + roi, s[sel]
        bc = box_col[valid][sel]
        N = int(2 * roi / RES) + 1
        iu = np.clip((u_ / RES).astype(int), 0, N - 1)
        iv = np.clip((v_ / RES).astype(int), 0, N - 1)
        okm = np.zeros((N, N), np.float32)
        abv = np.zeros((N, N), np.float32)
        onp = (np.abs(s_) < PLANE_TOL) & bc
        np.maximum.at(okm, (iv[onp], iu[onp]), 1.0)
        hi = s_ > ABOVE_TOL
        np.maximum.at(abv, (iv[hi], iu[hi]), 1.0)
        okm = cv2.morphologyEx(okm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        V = np.full((N, N), W_NEUTRAL, np.float32)
        V[okm > 0] = 1.0
        V[abv > 0] = W_ABOVE

        for t in tmpls:
            th, tw = t["img"].shape
            if th >= N or tw >= N:
                continue
            r = cv2.matchTemplate(V, t["img"].astype(np.float32), cv2.TM_CCORR)
            _, mx, _, loc = cv2.minMaxLoc(r)
            score = mx / t["area"]
            better = (best is None or score > best["score"] + 0.015 or
                      (score > best["score"] - 0.015 and t["area"] > best["area"]))
            if better:
                # polygon in plane coords -> 3D -> image
                ox = loc[0] * RES - roi + cs_u
                oy = loc[1] * RES - roi + cs_v
                poly_uv = t["poly"] - t["off"] + [ox, oy]
                poly3d = c + poly_uv[:, :1] * e1 + poly_uv[:, 1:2] * e2
                th_ = np.deg2rad(t["deg"])
                Rt = np.array([[np.cos(th_), -np.sin(th_)],
                               [np.sin(th_), np.cos(th_)]])
                faces_px, faces3d = [], []
                for q in l_faces(t["scale"], t["mirror"], t["h1"]):
                    q_uv = q @ Rt.T - t["off"] + [ox, oy]
                    q3d = c + q_uv[:, :1] * e1 + q_uv[:, 1:2] * e2
                    faces3d.append(q3d)
                    faces_px.append(P.project(q3d, intr))
                best = dict(score=float(score), deg=t["deg"], scale=t["scale"],
                            mirror=t["mirror"], h1=t["h1"], poly3d=poly3d,
                            area=t["area"],
                            poly_px=P.project(poly3d, intr), n=n, c=c,
                            faces_px=faces_px, faces3d=faces3d)
    return best


def run(frames=None):
    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, "dataset", "two_lights")
    out = os.path.join(base, "overlays_lfit")
    os.makedirs(out, exist_ok=True)
    intr = json.load(open(os.path.join(data, "intrinsics.json")))
    tmpls = make_templates()

    import glob as g, time
    frames = frames or sorted(os.path.basename(x)[:3]
                              for x in g.glob(os.path.join(data, "*_rgb.png")))
    results = {}
    t0 = time.time()
    for f in frames:
        res = P.analyze(os.path.join(data, f + "_depth.npy"),
                        os.path.join(data, f + "_rgb.png"), intr)
        rgb = res["rgb"]
        fit = fit_frame(res, intr, tmpls) if res["comps"] else None
        p1 = rgb.copy()
        if fit is None:
            results[f] = {"found": False}
            cv2.putText(p1, "NO CANDIDATE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(out, f + "_overlay.png"), p1)
            print(f"{f}: NO CANDIDATE")
            continue
        ok_fit = fit["score"] >= ACCEPT
        poly = fit["poly_px"].astype(np.int32)
        mask = np.zeros(rgb.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        col = (0, 255, 0) if ok_fit else (0, 165, 255)
        cv2.polylines(p1, [poly], True, col, 3)
        for v in poly:
            cv2.circle(p1, tuple(v), 5, (0, 0, 255), -1)
        sc = fit["scale"]
        txt = (f"L-fit {fit['score']:.2f}  {L_W*sc*100:.0f}x{L_H*sc*100:.0f}cm "
               f"rot={fit['deg']}deg{' MIRROR' if fit['mirror'] else ''}"
               f"{'' if ok_fit else ' LOW-CONF'}")
        cv2.putText(p1, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(p1, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        # side panel: the 3 FACES (label map 1/2/3) + middle-face 4-corner box
        FACE_COLS = [(80, 200, 80), (60, 120, 255), (255, 170, 60)]
        FACE_NAMES = ["F1", "M(mid)", "F3"]
        face_map = np.zeros(rgb.shape[:2], np.uint8)
        p2 = (rgb * 0.35).astype(np.uint8)
        for i, q in enumerate(fit["faces_px"]):
            qi = q.astype(np.int32)
            cv2.fillPoly(face_map, [qi], i + 1)
            fm = np.zeros(rgb.shape[:2], np.uint8)
            cv2.fillPoly(fm, [qi], 255)
            mm = fm > 0
            colf = np.array(FACE_COLS[i], np.float32)
            p2[mm] = (0.35 * rgb[mm] + 0.6 * colf).clip(0, 255).astype(np.uint8)
            ys, xs = np.nonzero(mm)
            if len(xs):
                cv2.putText(p2, FACE_NAMES[i], (int(xs.mean()) - 20, int(ys.mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        mid = fit["faces_px"][1].astype(np.int32)
        cv2.polylines(p2, [mid], True, (255, 255, 255), 3)
        cv2.polylines(p1, [mid], True, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(out, f + "_overlay.png"), np.hstack([p1, p2]))
        cv2.imwrite(os.path.join(out, f + "_mask.png"), mask)
        cv2.imwrite(os.path.join(out, f + "_faces.png"), face_map * 80)
        results[f] = {"found": True, "score": round(fit["score"], 3),
                      "polygon_px": poly.tolist(),
                      "polygon_3d": fit["poly3d"].tolist(),
                      "rot_deg": fit["deg"], "mirror": fit["mirror"],
                      "scale": fit["scale"], "top_flap_h1_m": fit["h1"],
                      "faces_px": {n_: fit["faces_px"][i].round(1).tolist()
                                   for i, n_ in enumerate(("F1_flap", "M_middle", "F3_flap"))},
                      "faces_3d": {n_: fit["faces3d"][i].tolist()
                                   for i, n_ in enumerate(("F1_flap", "M_middle", "F3_flap"))},
                      "middle_face_box_px": fit["faces_px"][1].round(1).tolist()}
        print(f"{f}: score={fit['score']:.2f} rot={fit['deg']} "
              f"scale={fit['scale']} mirror={fit['mirror']}"
              f"{'' if ok_fit else ' LOW-CONF'}")
    with open(os.path.join(out, "lfit_results.json"), "w") as fp:
        json.dump(results, fp, indent=1)
    print(f"avg {(time.time()-t0)/len(frames):.1f} s/frame")


if __name__ == "__main__":
    import sys
    run(sys.argv[1:] or None)
