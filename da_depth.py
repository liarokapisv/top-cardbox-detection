#!/usr/bin/env python3
"""Depth-Anything-V2 (ONNX, CPU) -> top-box mask by edge + blob fill,
then the existing l_fit engine for the 8-gon / faces / pose.

The captured D435 depth cannot delineate a flat-lying blank (bin floor is
bowed ~2 cm, noise floor ~3 mm, blank ~3 mm). DA V2 predicts RGB-only
relative depth whose boundaries are appearance-aligned and crisp; we use it
to find WHERE the top blank is, then reuse the sensor depth (via l_fit) to
recover the metric 8-gon.

usage:
  python3 da_depth.py 009            # one frame -> overlays_da/009_*.png
  python3 da_depth.py                # all frames
  python3 da_depth.py --compare      # + A/B contact sheet vs pipeline seed

Model: onnx-community/depth-anything-v2-small (ViT-S). Auto-fetched to
models/da2/ on first use. CPU only.
"""
import numpy as np
import cv2
import json
import os
import sys
import glob
import time

import onnxruntime as ort

import pipeline as P
import l_fit as LF

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "dataset", "two_lights")
OUT = os.path.join(BASE, "overlays_da")
MODEL_DIR = os.path.join(BASE, "models", "da2")
MODEL = os.path.join(MODEL_DIR, "da2_small.onnx")
HF_REPO = "onnx-community/depth-anything-v2-small"
HF_FILE = "onnx/model.onnx"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
LONG_SIDE = 518                       # DA input; kept a multiple of 14
# bin floor polygon (shared with the synth generators / heatmaps)
FLOOR = np.array([[95, 130], [560, 130], [600, 420], [55, 420]], np.int32)

MIN_BLOB = 1800                       # px, drop specks
_session = None


# ---------------- DA V2 inference ----------------
def ensure_model():
    if os.path.exists(MODEL):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download
    print(f"fetching {HF_REPO}/{HF_FILE} ...", flush=True)
    src = hf_hub_download(HF_REPO, HF_FILE)
    import shutil
    shutil.copy(src, MODEL)


def session():
    global _session
    if _session is None:
        ensure_model()
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "4"))
        _session = ort.InferenceSession(MODEL, so,
                                        providers=["CPUExecutionProvider"])
    return _session


def _mult14(x):
    return max(14, int(round(x / 14.0)) * 14)


def infer_depth(rgb):
    """RGB (BGR uint8, HxW) -> relative inverse-depth map, same HxW.
    Larger value = nearer the camera."""
    h, w = rgb.shape[:2]
    s = LONG_SIDE / max(h, w)
    nh, nw = _mult14(h * s), _mult14(w * s)
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), (nw, nh),
                     interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    inp = img.transpose(2, 0, 1)[None].astype(np.float32)
    out = session().run(None, {"pixel_values": inp})[0][0]     # (nh',nw')
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)


# ---------------- floor detrend ----------------
def _quad_basis(shape):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx - w / 2) / (w / 2)
    v = (yy - h / 2) / (h / 2)
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1)


def detrend_floor(depth):
    """Subtract a robust quadratic surface fit to the bin floor, so a raised
    blank stands out. Returns (rel_height, robust_sigma)."""
    A = _quad_basis(depth.shape)
    fl = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(fl, [FLOOR], 255)
    m = fl > 0
    Am, dm = A[m], depth[m]
    w = np.ones(len(dm), bool)
    coef = np.zeros(A.shape[-1], np.float32)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(Am[w], dm[w], rcond=None)
        r = dm - Am @ coef
        sig = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = np.abs(r - np.median(r)) < 2.5 * sig
    rel = depth - (A @ coef).astype(np.float32)
    resid = rel[m]
    sig = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
    return rel, float(sig)


# ---------------- top-box mask (edge + blob fill) ----------------
def top_mask_from_depth(depth, res):
    """The blank is bright/printed vs the grey bin (box colour prior); DA
    relief RANKS overlapping blanks (nearer = top) and SPLITS a stack at the
    depth step. So: box-coloured components inside the bin floor, cut apart
    at DA edges, ranked by DA relief -> the top blank.
    Returns (top_mask bool, edges uint8, rel float, blobs list)."""
    rel, sig = detrend_floor(depth)

    fl = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(fl, [FLOOR], 255)
    fl = fl > 0

    # boundary evidence from DA (sign-agnostic depth step): only a genuine
    # gradient ridge, well above the per-frame relief noise, cuts blanks
    # apart. On a near-flat single blank gmag ~ noise, so nothing is cut and
    # its box-colour region stays whole.
    gmag = np.hypot(cv2.Sobel(rel, cv2.CV_32F, 1, 0, ksize=5),
                    cv2.Sobel(rel, cv2.CV_32F, 0, 1, ksize=5))
    edges = (gmag > 6 * sig).astype(np.uint8) * 255

    # box-coloured region inside the floor, de-speckled; then carve at the
    # DA depth steps so a top blank overlapping a lower one splits off.
    bc = (res["box_col"] & fl).astype(np.uint8)
    bc = cv2.morphologyEx(bc, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cut = bc & (cv2.dilate(edges, np.ones((3, 3), np.uint8)) == 0)
    n, lab = cv2.connectedComponents(cut)

    blobs = []
    for i in range(1, n):
        m0 = lab == i
        if m0.sum() < MIN_BLOB:
            continue
        # regrow to the box-colour region it was carved from, fill holes
        m = cv2.morphologyEx(m0.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((11, 11), np.uint8)) & bc
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        mb = m > 0
        blobs.append((mb, float(np.median(rel[m0])), int(mb.sum())))

    # the top blank is fully visible, so its box-colour region is the LARGEST
    # connected piece; lower blanks are occluded -> smaller fragments. (The DA
    # depth-step cut above separates a top blank from a same-colour one it
    # overlaps.) We order candidates largest-first; l_fit then arbitrates
    # (a merged coplanar-adjacent pair fails the L completeness check, so the
    # next single blob wins). Selection happens in process().
    blobs.sort(key=lambda b: -b[2])
    return edges, rel, blobs


# ---------------- l_fit on the DA mask ----------------
CAND_K = 4                            # blobs to try through l_fit


def fit_on_mask(res, mask, intr, tmpls):
    """Reuse l_fit.fit_frame unchanged by presenting one DA blob as the
    single seed component."""
    m = mask & res["valid"]
    if int(m.sum()) < 1500:
        return None
    res2 = dict(res)
    res2["comps"] = {0: {"mask": m, "score": 1.0, "above_frac": 1.0,
                         "area": int(m.sum())}}
    return LF.fit_frame(res2, intr, tmpls)


def best_fit(res, blobs, intr, tmpls):
    """Try the largest CAND_K blobs; keep the highest-scoring l_fit. Returns
    (fit, chosen_mask). l_fit is the completeness authority that rejects
    merged/occluded blobs so the true top blank wins."""
    best, chosen = None, None
    for mb, _relief, _area in blobs[:CAND_K]:
        fit = fit_on_mask(res, mb, intr, tmpls)
        if fit is not None and (best is None or fit["score"] > best["score"]):
            best, chosen = fit, mb
    return best, chosen


# ---------------- viz ----------------
def turbo(vals, lo, hi):
    n = np.clip((vals - lo) / max(hi - lo, 1e-9), 0, 1)
    return cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _label(img, txt, org, col, sc=0.6):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 4)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)


def render(f, res, depth, top, edges, rel, blobs, chosen, fit):
    rgb = res["rgb"]
    da = turbo(depth, *np.percentile(depth, [2, 98]))
    rl = turbo(rel, *np.percentile(rel, [2, 98]))

    # ---- relief panel: annotate candidate blobs (already area-sorted); the
    # l_fit-arbitrated pick is green, other candidates orange, rest grey
    for rank, (mb, relief, area) in enumerate(blobs):
        cnts, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        is_pick = chosen is not None and mb is chosen
        col = (0, 255, 0) if is_pick else (
            (0, 200, 255) if rank < CAND_K else (160, 160, 160))
        cv2.drawContours(rl, cnts, -1, col, 2)
        ys, xs = np.nonzero(mb)
        _label(rl, f"#{rank} {area//1000}k r={relief:.2f}",
               (int(xs.mean()) - 40, int(ys.mean())), col, 0.45)

    # ---- overlay panel: chosen mask + edges + 8-gon + faces + debug block
    ov = rgb.copy()
    if top.any():
        ov[top] = (0.45 * ov[top] + np.array([0, 140, 0])).clip(0, 255).astype(np.uint8)
    ov[edges > 0] = (0, 0, 255)
    lines = [f"{f}  blobs={len(blobs)}  top=#0"]
    if fit is not None:
        acc = fit["score"] >= LF.ACCEPT
        col = (0, 255, 0) if acc else (0, 165, 255)
        poly = fit["poly_px"].astype(np.int32)
        cv2.polylines(ov, [poly], True, col, 2)
        for v in poly:
            cv2.circle(ov, tuple(v), 4, (0, 0, 255), -1)
        cv2.polylines(ov, [fit["faces_px"][1].astype(np.int32)], True,
                      (255, 255, 255), 2)
        lines += [f"score={fit['score']:.2f} {'ACCEPT' if acc else 'LOW-CONF'}",
                  f"rot={fit['deg']} scale={fit['scale']} "
                  f"mir={int(fit['mirror'])} h1={fit['h1']*1000:.0f}mm"]
    else:
        col = (0, 0, 255)
        lines += ["NO FIT"]
    for k, ln in enumerate(lines):
        _label(ov, ln, (10, 24 + 22 * k), col if k else (255, 255, 255), 0.6)

    for panel, name in ((da, "DA depth"), (rl, "floor relief"),
                        (ov, "mask+8gon")):
        _label(panel, name, (10, panel.shape[0] - 12), (255, 255, 255), 0.5)
    grid = np.hstack([rgb, da, rl, ov])
    cv2.imwrite(os.path.join(OUT, f"{f}_da.png"), grid)


# ---------------- driver ----------------
def process(frames, intr, tmpls, results):
    for f in frames:
        t0 = time.time()
        res = P.analyze(os.path.join(DATA, f + "_depth.npy"),
                        os.path.join(DATA, f + "_rgb.png"), intr)
        depth = infer_depth(res["rgb"])
        edges, rel, blobs = top_mask_from_depth(depth, res)
        fit, chosen = best_fit(res, blobs, intr, tmpls)
        top = chosen if chosen is not None else (
            blobs[0][0] if blobs else np.zeros(depth.shape, bool))
        render(f, res, depth, top, edges, rel, blobs, chosen, fit)
        if fit is None:
            results[f] = {"found": False}
            print(f"{f}: no fit  ({time.time()-t0:.1f}s)", flush=True)
            continue
        poly = fit["poly_px"].round(1).tolist()
        results[f] = {
            "found": True, "score": round(fit["score"], 3),
            "polygon_px": np.asarray(fit["poly_px"]).round().astype(int).tolist(),
            "polygon_3d": fit["poly3d"].tolist(),
            "rot_deg": fit["deg"], "mirror": fit["mirror"],
            "scale": fit["scale"], "top_flap_h1_m": fit["h1"],
            "faces_px": {n_: fit["faces_px"][i].round(1).tolist()
                         for i, n_ in enumerate(("F1_flap", "M_middle", "F3_flap"))},
            "middle_face_box_px": fit["faces_px"][1].round(1).tolist(),
            "n_blobs": len(blobs)}
        print(f"{f}: DA-fit score={fit['score']:.2f} rot={fit['deg']} "
              f"blobs={len(blobs)} ({time.time()-t0:.1f}s)", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    intr = json.load(open(os.path.join(DATA, "intrinsics.json")))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    compare = "--compare" in sys.argv
    frames = args or sorted(os.path.basename(x)[:3]
                            for x in glob.glob(os.path.join(DATA, "*_rgb.png")))
    tmpls = LF.make_templates()
    results = {}
    process(frames, intr, tmpls, results)
    json.dump(results, open(os.path.join(OUT, "da_results.json"), "w"), indent=1)
    n_ok = sum(1 for v in results.values() if v.get("found"))
    print(f"\n{n_ok}/{len(frames)} fitted -> {OUT}/da_results.json")
    if compare:
        compare_sheet(frames)


def compare_sheet(frames):
    """Contact sheet: RGB | DA relief+mask+8gon per frame."""
    tiles = []
    for f in frames:
        p = os.path.join(OUT, f + "_da.png")
        if os.path.exists(p):
            g = cv2.imread(p)
            tiles.append(cv2.resize(g, (960, 180)))
    if not tiles:
        return
    sheet = np.vstack(tiles)
    cv2.imwrite(os.path.join(OUT, "contact_sheet_da.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"contact sheet -> {OUT}/contact_sheet_da.jpg")


if __name__ == "__main__":
    main()
