#!/usr/bin/env python3
"""Articulated synthetic generator v3.

Each cutout is rectified to the canonical L frame and split into its 3
face rectangles:
    A = [0, W-nw] x [0, nh]     (beside the notch)
    B = [0, W-nw] x [nh, H]
    C = [W-nw, W] x [nh, H]
Seams: A-B at y=nh, B-C at x=W-nw.

Per generated box:
  * per-face compliance: each face is perspective-warped with its SEAM
    corners pinned and only free outer corners jittered slightly (+ per-
    face brightness) -> faces flex but coincide exactly at the seams
  * whole object: slight out-of-plane tilt (small perspective), full
    in-plane rotation, translation, slight scale
The composed outline stays an exact 6-point polygon -> the top-box label.
"""
import numpy as np
import cv2
import json
import os
import sys

import l_fit as LF

PXM = 1000.0          # canonical raster: 1 px per mm
MARGIN = 40           # px margin around canonical canvas
FACE_JIT = 5.0        # mm, per-face free-corner jitter (compliance)
TILT_JIT = 0.05       # fraction, whole-object perspective corner jitter


def canonical_vertices(scale, mirror, h1=0.0):
    return LF.l_polygon(scale, mirror, h1) * PXM   # mm


def load_cutouts():
    meta = json.load(open("cutouts/meta.json"))
    lref = json.load(open("overlays_lfit/lfit_results.json"))
    cuts = []
    for k, m in meta.items():
        f = m["frame"]
        r = lref[f]
        img = cv2.imread(f"cutouts/{k}.png", cv2.IMREAD_UNCHANGED)
        poly_img = np.array(m["polygon"], np.float64)          # cutout-local px
        verts = canonical_vertices(r["scale"], r["mirror"],
                                   r.get("top_flap_h1_m", 0.0))  # mm, centred
        vmin = verts.min(0)
        verts = verts - vmin + MARGIN                          # canvas px
        Hm, _ = cv2.findHomography(poly_img.reshape(-1, 1, 2),
                                   verts.reshape(-1, 1, 2))
        cw = int(verts[:, 0].max() + MARGIN)
        ch = int(verts[:, 1].max() + MARGIN)
        rgb = cv2.warpPerspective(img[:, :, :3], Hm, (cw, ch))
        a = cv2.warpPerspective(img[:, :, 3], Hm, (cw, ch))
        h1 = r.get("top_flap_h1_m", 0.0)
        faces = [q * PXM - vmin + MARGIN
                 for q in LF.l_faces(r["scale"], r["mirror"], h1)]
        cuts.append(dict(key=k, side=m["side"], mirror=r["mirror"],
                         scale=r["scale"], rgb=rgb, a=a, verts=verts,
                         faces=faces))
    return cuts


def hinge_fold(quad, hinge_p0, hinge_p1, theta, view_vec):
    """Revolute joint: rotate a face by theta about the hinge LINE through
    hinge_p0-hinge_p1. In projection: points compress toward the hinge by
    cos(theta) and shift laterally by view_vec * (out-of-plane lift)."""
    h0 = np.asarray(hinge_p0, np.float64)
    hv = np.asarray(hinge_p1, np.float64) - h0
    hv /= np.linalg.norm(hv)
    nvec = np.array([-hv[1], hv[0]])          # in-plane normal to hinge
    out = quad.copy().astype(np.float64)
    for i, p in enumerate(out):
        d = float((p - h0) @ nvec)            # signed distance to hinge
        foot = p - d * nvec
        lift = abs(d) * np.sin(theta)         # out-of-plane height
        out[i] = foot + nvec * d * np.cos(theta) + view_vec * lift
    return out


def articulate(cut, rng):
    """Compose the box from its 3 rigid faces (F1 flap, M middle, F3 flap)
    sharing 2 edge revolute joints. M is the base; each flap folds about
    its hinge with M. Face brightness follows its tilt."""
    verts = cut["verts"].copy()
    ch, cw = cut["a"].shape
    canvas_rgb = np.zeros((ch, cw, 3), np.float32)
    canvas_a = np.zeros((ch, cw), np.float32)
    F1, M, F3 = [q.copy() for q in cut["faces"]]

    def hinge_edge(flap, base):
        """The flap edge shared with the base face (the 2 flap corners
        closest to the base quad)."""
        d = [min(np.linalg.norm(p - b) for b in base) for p in flap]
        i2 = np.argsort(d)[:2]
        return flap[i2[0]], flap[i2[1]]

    view = rng.uniform(-0.35, 0.35, 2)
    light = rng.choice([-1.0, 1.0])
    warps = []
    areaF1 = abs(cv2.contourArea(F1.astype(np.float32)))
    for flap in (F1, F3):
        if abs(cv2.contourArea(flap.astype(np.float32))) < 100:  # folded away
            warps.append((flap, flap.copy(), 1.0))
            continue
        th = np.deg2rad(rng.uniform(-14, 14))
        h0, h1_ = hinge_edge(flap, M)
        warps.append((flap, hinge_fold(flap, h0, h1_, th, view),
                      1.0 + 0.22 * np.sin(th) * light))
    (srcF1, dstF1, gF1), (srcF3, dstF3, gF3) = warps
    dstM, gM = M.copy(), rng.uniform(0.96, 1.04)

    for src, dst, g in ((srcF1, dstF1, gF1), (M, dstM, gM), (srcF3, dstF3, gF3)):
        if abs(cv2.contourArea(src.astype(np.float32))) < 100:
            continue
        Mx = cv2.getPerspectiveTransform(src.astype(np.float32),
                                         dst.astype(np.float32))
        fm = np.zeros((ch, cw), np.uint8)
        cv2.fillPoly(fm, [src.astype(np.int32)], 255)
        frgb = cut["rgb"].copy(); frgb[fm == 0] = 0
        fa = cut["a"].copy(); fa[fm == 0] = 0
        wrgb = cv2.warpPerspective(frgb, Mx, (cw, ch)).astype(np.float32) * g
        wa = cv2.warpPerspective(fa, Mx, (cw, ch)).astype(np.float32)
        keep = wa > canvas_a
        canvas_rgb[keep] = wrgb[keep]
        canvas_a[keep] = wa[keep]

    # articulated outline: map each vertex through the warp of the face
    # that owns it (nearest quad); hinge points agree from either side
    quads = [(srcF1, dstF1), (M, dstM), (srcF3, dstF3)]
    new_verts = verts.copy()
    for i, v in enumerate(verts):
        dmin, own = None, None
        for src, dst in quads:
            if abs(cv2.contourArea(src.astype(np.float32))) < 100:
                continue
            d = cv2.pointPolygonTest(src.astype(np.float32),
                                     (float(v[0]), float(v[1])), True)
            if dmin is None or d > dmin:
                dmin, own = d, (src, dst)
        Mx = cv2.getPerspectiveTransform(own[0].astype(np.float32),
                                         own[1].astype(np.float32))
        p = Mx @ np.array([v[0], v[1], 1.0])
        new_verts[i] = p[:2] / p[2]
    return canvas_rgb.clip(0, 255).astype(np.uint8), \
        canvas_a.clip(0, 255).astype(np.uint8), new_verts


def gen(out, N, seed):
    rng = np.random.RandomState(seed)
    for sub in ("images", "labels", "annotated"):
        os.makedirs(f"{out}/{sub}", exist_ok=True)
    cuts = load_cutouts()
    bin_img = cv2.imread("dataset/two_lights/028_rgb.png")
    BH, BW = bin_img.shape[:2]
    FLOOR = np.array([[95, 130], [560, 130], [600, 420], [55, 420]], np.int32)
    floor_mask = np.zeros((BH, BW), np.uint8)
    cv2.fillPoly(floor_mask, [FLOOR], 255)
    # canonical mm -> scene px: blank long side 310 mm appears ~300 px
    base_s = 300.0 / 310.0 / (PXM / 1000.0)   # px per canvas-px

    recs = {}
    for idx in range(N):
        img = bin_img.copy().astype(np.float32)
        n_boxes = rng.randint(1, 9)
        top = None
        for b in range(n_boxes):
            c = cuts[rng.randint(len(cuts))]
            rgbc, ac, verts = articulate(c, rng)
            ch, cw = ac.shape
            # whole-object: slight out-of-plane tilt via corner-jittered
            # perspective, then rotation+scale+translation
            src4 = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32)
            tj = TILT_JIT * max(cw, ch)
            dst4 = src4 + rng.uniform(-tj, tj, (4, 2)).astype(np.float32)
            Hp = cv2.getPerspectiveTransform(src4, dst4)
            ang = rng.uniform(0, 360)
            sc = base_s * rng.uniform(0.93, 1.07)
            th = np.deg2rad(ang)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * sc
            for _ in range(60):
                # transformed bbox
                corners = np.array([[0, 0, 1], [cw, 0, 1], [cw, ch, 1], [0, ch, 1]]).T
                pc = Hp @ corners
                pc = (pc[:2] / pc[2]).T @ R.T
                mn, mx = pc.min(0), pc.max(0)
                nw, nh = int(mx[0] - mn[0]) + 2, int(mx[1] - mn[1]) + 2
                if nw >= BW or nh >= BH:
                    sc *= 0.95
                    R = np.array([[np.cos(th), -np.sin(th)],
                                  [np.sin(th), np.cos(th)]]) * sc
                    continue
                px = rng.randint(0, BW - nw)
                py = rng.randint(0, BH - nh)
                if floor_mask[py + nh // 2, px + nw // 2]:
                    break
            else:
                continue
            Aff = np.zeros((3, 3))
            Aff[:2, :2] = R
            Aff[:2, 2] = [px - mn[0], py - mn[1]]
            Aff[2, 2] = 1
            Hfull = Aff @ Hp
            wrgb = cv2.warpPerspective(rgbc, Hfull, (BW, BH))
            wa = cv2.warpPerspective(ac, Hfull, (BW, BH)).astype(np.float32) / 255.0
            # shadow
            sh = cv2.GaussianBlur(np.roll(np.roll(wa, 8, 0), 6, 1), (21, 21), 0) * 0.45
            img *= (1.0 - sh[..., None] * 0.6)
            g = rng.uniform(0.9, 1.08)
            img = img * (1 - wa[..., None]) + \
                (wrgb.astype(np.float32) * g).clip(0, 255) * wa[..., None]
            vh = np.hstack([verts, np.ones((len(verts), 1))]).T
            pv = Hfull @ vh
            pverts = (pv[:2] / pv[2]).T
            mfull = (wa * 255).astype(np.uint8)
            top = dict(poly=pverts, mask=mfull, side=c["side"], key=c["key"])
        img = (img * rng.uniform(0.92, 1.06)).clip(0, 255).astype(np.uint8)
        name = f"synth_{idx:04d}"
        cv2.imwrite(f"{out}/images/{name}.png", img)
        cv2.imwrite(f"{out}/labels/{name}_topmask.png", top["mask"])
        recs[name] = {"n_boxes": int(n_boxes), "top_side": top["side"],
                      "top_source": top["key"],
                      "top_polygon_px": top["poly"].round(1).tolist()}
        if idx < 30:
            ann = img.copy()
            cv2.polylines(ann, [top["poly"].astype(np.int32)], True, (0, 255, 0), 3)
            for v in top["poly"].astype(np.int32):
                cv2.circle(ann, tuple(v), 5, (0, 0, 255), -1)
            lbl = f"{name} boxes={n_boxes} top={top['side']}"
            cv2.putText(ann, lbl, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(ann, lbl, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(f"{out}/annotated/{name}.png", ann)
    json.dump(recs, open(f"{out}/labels/labels.json", "w"), indent=1)
    print(out, N, "done")


if __name__ == "__main__":
    gen(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
