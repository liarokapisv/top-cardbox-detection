#!/usr/bin/env python3
"""Articulated synthetic generator v3.

Each cutout is rectified to the canonical frame and split into the L's
3 intrinsic faces: arm A and tab B (one per extension) joined to the
CORNER face C at the junction, with perpendicular hinges (A-C vertical,
C-B horizontal) meeting at the inner corner. Both seams are refined per
cutout: the actual fold crease (dark line) is searched near the
canonical position, so the split follows the physical crease even when
the source blank was photographed slightly folded.

Per generated box:
  * C is the base; A and B each fold about their hinge with C (edge
    revolute joints; the hinge line is pinned so the seam stays closed
    by construction); face brightness follows its tilt
  * content is split into quadrants at the seam lines with a 2 px
    overlap, and residual cracks inside the outline are inpainted
  * whole object: slight out-of-plane tilt (small perspective), full
    in-plane rotation, translation, slight scale
The composed outline stays an exact 6/8-point polygon -> top-box label.
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
SEAM_SEARCH = 14      # px(=mm) each side of the canonical seam row
SEAM_MIN_DEPTH = 5.0  # gray levels below baseline to accept a crease
SIDES = ("A",)        # only the printed side; side B (brown back) is unused


def canonical_vertices(scale, mirror, h1=0.0):
    return LF.l_polygon(scale, mirror, h1) * PXM   # mm


def find_crease(gray, a, y_can, x0, x1, ylo, yhi):
    """Refine one seam row: darkest smoothed row (the fold line) within
    +/-SEAM_SEARCH of the canonical position; canonical kept if no clear
    crease shows in the texture."""
    lo = int(max(ylo, y_can - SEAM_SEARCH))
    hi = int(min(yhi, y_can + SEAM_SEARCH + 1))
    xs0, xs1 = int(max(0, x0 + 3)), int(min(gray.shape[1], x1 - 3))
    if hi - lo < 5 or xs1 - xs0 < 20:
        return float(y_can)
    band = gray[lo:hi, xs0:xs1]
    ok = a[lo:hi, xs0:xs1] > 128
    cnt = ok.sum(axis=1)
    rows = (band * ok).sum(axis=1) / np.maximum(cnt, 1)
    valid = cnt > 0.6 * (xs1 - xs0)
    if valid.sum() < 5:
        return float(y_can)
    base = float(np.median(rows[valid]))
    rows[~valid] = base
    k = np.array([1, 4, 6, 4, 1], np.float32) / 16
    sm = np.convolve(rows, k, mode="same")
    i = int(np.argmin(sm))
    if base - sm[i] < SEAM_MIN_DEPTH or not valid[i]:
        return float(y_can)
    return float(lo + i)


def _rect(x0, x1, ya, yb):
    return np.array([(x0, ya), (x1, ya), (x1, yb), (x0, yb)], np.float64)


def refine_seams(rgb, a, faces):
    """Per-cutout seam positions (vertical A-C, horizontal C-B): the
    canonical hinge lines nudged onto the actual crease found in the
    rectified texture."""
    A, C, B = faces
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a_left = A[:, 0].mean() < C[:, 0].mean()
    if a_left:
        xc = 0.5 * (A[:, 0].max() + C[:, 0].min())
        x_far = max(C[:, 0].max(), B[:, 0].max())
        x_out = A[:, 0].min()
    else:
        xc = 0.5 * (A[:, 0].min() + C[:, 0].max())
        x_far = min(C[:, 0].min(), B[:, 0].min())
        x_out = A[:, 0].max()
    yc = 0.5 * (C[:, 1].max() + B[:, 1].min())
    y_topC = float(C[:, 1].min())
    y_bot = float(B[:, 1].max())
    # horizontal C-B seam: dark row near yc over the tab's x-span
    yc = find_crease(gray, a, yc, min(B[:, 0]), max(B[:, 0]),
                     y_topC + 10, y_bot - 3)
    # vertical A-C seam: dark column near xc over the corner's y-span
    # (transposed arrays swap row/column roles)
    lo, hi = (x_out, x_far) if a_left else (x_far, x_out)
    xc = find_crease(np.ascontiguousarray(gray.T), np.ascontiguousarray(a.T),
                     xc, y_topC + 3, yc - 3, lo + 10, hi - 10)
    return float(xc), float(yc), a_left


def load_cutouts():
    meta = json.load(open("cutouts/meta.json"))
    lref = json.load(open("overlays_lfit/lfit_results.json"))
    cuts = []
    for k, m in meta.items():
        if m["side"] not in SIDES:
            continue
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
        xc, yc, a_left = refine_seams(rgb, a, faces)
        A, C, B = faces
        if a_left:
            faces = [_rect(A[:, 0].min(), xc, A[:, 1].min(), yc),
                     _rect(xc, C[:, 0].max(), C[:, 1].min(), yc),
                     _rect(xc, B[:, 0].max(), yc, B[:, 1].max())]
        else:
            faces = [_rect(xc, A[:, 0].max(), A[:, 1].min(), yc),
                     _rect(C[:, 0].min(), xc, C[:, 1].min(), yc),
                     _rect(B[:, 0].min(), xc, yc, B[:, 1].max())]
        cuts.append(dict(key=k, side=m["side"], mirror=r["mirror"],
                         scale=r["scale"], rgb=rgb, a=a, verts=verts,
                         faces=faces, seams=(xc, yc), a_left=a_left))
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


def articulate(cut, rng, dbg=False):
    """Compose the box from the L's 3 rigid faces (arm A, corner C, tab B)
    sharing 2 perpendicular edge revolute joints. C is the base; A and B
    each fold about their hinge with C. Face brightness follows its tilt."""
    verts = cut["verts"].copy()
    ch, cw = cut["a"].shape
    canvas_rgb = np.zeros((ch, cw, 3), np.float32)
    canvas_a = np.zeros((ch, cw), np.float32)
    A, C, B = [q.copy() for q in cut["faces"]]
    xc, yc = cut["seams"]
    a_left = cut["a_left"]

    def hinge_edge(flap, base):
        """The flap edge shared with the base face (the 2 flap corners
        closest to the base QUAD, not its corners — the seam corners lie
        on the base's edge, so quad distance is 0 there)."""
        bq = base.astype(np.float32)
        d = [abs(cv2.pointPolygonTest(bq, (float(p[0]), float(p[1])), True))
             for p in flap]
        i2 = np.argsort(d)[:2]
        return flap[i2[0]], flap[i2[1]]

    view = rng.uniform(-0.35, 0.35, 2)
    light = rng.choice([-1.0, 1.0])
    warps, thetas = [], []
    for flap in (A, B):
        if abs(cv2.contourArea(flap.astype(np.float32))) < 100:  # folded away
            warps.append((flap, flap.copy(), 1.0))
            thetas.append(0.0)
            continue
        th = np.deg2rad(rng.uniform(-14, 14))
        thetas.append(float(np.degrees(th)))
        h0, h1_ = hinge_edge(flap, C)
        warps.append((flap, hinge_fold(flap, h0, h1_, th, view),
                      1.0 + 0.22 * np.sin(th) * light))
    (srcA, dstA, gA), (srcB, dstB, gB) = warps
    dstC, gC = C.copy(), rng.uniform(0.96, 1.04)

    alive = [abs(cv2.contourArea(q.astype(np.float32))) >= 100
             for q in (srcA, C, srcB)]
    Mxs = [cv2.getPerspectiveTransform(s.astype(np.float32),
                                       d.astype(np.float32)) if alive[i]
           else np.eye(3)
           for i, (s, d) in enumerate(((srcA, dstA), (C, dstC),
                                       (srcB, dstB)))]
    # content split: quadrants at the (refined) seam lines tile every
    # alpha>0 pixel exactly once (A gets its full column band; C above /
    # B below the horizontal seam on the other side); +2 px overlap
    # kills aliasing cracks (overlaps resolved by the max-alpha keep)
    c1, r2 = int(round(xc)), int(round(yc))
    fms = [np.zeros((ch, cw), np.uint8) for _ in range(3)]
    if a_left:
        fms[0][:, :min(c1 + 2, cw)] = 255
        fms[1][:min(r2 + 2, ch), max(c1 - 2, 0):] = 255
        fms[2][max(r2 - 2, 0):, max(c1 - 2, 0):] = 255
    else:
        fms[0][:, max(c1 - 2, 0):] = 255
        fms[1][:min(r2 + 2, ch), :min(c1 + 2, cw)] = 255
        fms[2][max(r2 - 2, 0):, :min(c1 + 2, cw)] = 255
    for i, g in enumerate((gA, gC, gB)):
        if not alive[i]:
            continue
        fm = fms[i]
        frgb = cut["rgb"].copy(); frgb[fm == 0] = 0
        fa = cut["a"].copy(); fa[fm == 0] = 0
        wrgb = cv2.warpPerspective(frgb, Mxs[i], (cw, ch)).astype(np.float32) * g
        wa = cv2.warpPerspective(fa, Mxs[i], (cw, ch)).astype(np.float32)
        keep = wa > canvas_a
        canvas_rgb[keep] = wrgb[keep]
        canvas_a[keep] = wa[keep]

    # articulated outline: each vertex moves with the face that owns it
    # (by seam quadrant); hinge points agree from either side
    new_verts = verts.copy()
    for i, v in enumerate(verts):
        on_a = (v[0] < xc) if a_left else (v[0] >= xc)
        j = 0 if on_a else (1 if v[1] < yc else 2)
        if not alive[j]:
            j = 1
        p = Mxs[j] @ np.array([v[0], v[1], 1.0])
        new_verts[i] = p[:2] / p[2]

    rgb8 = canvas_rgb.clip(0, 255).astype(np.uint8)
    a8 = canvas_a.clip(0, 255).astype(np.uint8)
    # seal any residual crack strictly inside the articulated outline
    interior = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(interior, [np.round(new_verts).astype(np.int32)], 255)
    interior = cv2.erode(interior, np.ones((5, 5), np.uint8))
    holes = ((interior > 0) & (a8 < 100)).astype(np.uint8) * 255
    if holes.any():
        rgb8 = cv2.inpaint(rgb8, holes, 3, cv2.INPAINT_TELEA)
        a8[holes > 0] = 255
    if dbg:
        return rgb8, a8, new_verts, [dstA, dstC, dstB], alive, thetas
    return rgb8, a8, new_verts


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
