#!/usr/bin/env python3
"""Top-level cardboard blank detector — depth-based, CPU-only.

Pipeline:
  1. depth cleanup (valid range, median smoothing)
  2. edge map = depth jumps (morph. range) + normal creases + invalid px
  3. connected components on smooth regions  -> over-segmentation
  4. merge components whose shared border has continuous depth
     (fold creases inside one blank) but keep true steps apart
  5. per-group plane fit + tilt/flatness filters (drop bin walls)
  6. "topness": a group is top-level when it is the nearer side along
     its boundaries with every neighbour
  7. tight box: min-area rect in the blank's own plane (physical size)
     + image-space rotated rect
"""
import numpy as np
import cv2
import json
import os

# ---------------- parameters ----------------
VALID_MIN, VALID_MAX = 0.15, 1.30      # m
JUMP_THR = 0.010                       # m at 0.45 m, scales with z^2
CREASE_DEG = 16.0                      # normal-variation threshold
MIN_SEG_AREA = 400                     # px, keep small panels for merging
MIN_GROUP_AREA = 3500                  # px, min final candidate size
MERGE_GAP = 0.0035                     # m, border depth gap for same object
MERGE_ANGLE = 32.0                     # deg, max normal angle to merge
MERGE_PLANE_DIST = 0.0045              # m, border points vs other plane
MAX_TILT_DEG = 55.0                    # reject steep surfaces (bin walls)
MAX_PLANE_RMS = 0.025                  # m, reject non-flat groups
ABOVE_MARGIN = 0.0025                  # m, margin to call A above B (~thickness)
REFINE_DIST = 0.012                    # m, plane-outlier cut when refining mask
MIN_PHYS_AREA = 0.012                  # m^2, lone fragments below this can't win
BOX_SAT_MIN = 65                       # HSV S above this -> box (green/brown)
BOX_VAL_MIN = 185                      # HSV V above this -> box (white)
MAX_GREY_FRAC = 0.60                   # group more grey than this = bin, not box
MIN_SHARED_BORDER = 15                 # px of shared boundary


def deproject(depth, intr):
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    v = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)
    x = (u - intr["cx"]) / intr["fx"] * depth
    y = (v - intr["cy"]) / intr["fy"] * depth
    return np.dstack([x, y, depth])


def project(pts3d, intr):
    z = pts3d[:, 2]
    u = pts3d[:, 0] / z * intr["fx"] + intr["cx"]
    v = pts3d[:, 1] / z * intr["fy"] + intr["cy"]
    return np.stack([u, v], axis=1)


def fit_plane(pts):
    c = pts.mean(axis=0)
    q = pts - c
    cov = q.T @ q / len(q)
    w_, v_ = np.linalg.eigh(cov)
    n = v_[:, 0]
    rms = float(np.sqrt(max(w_[0], 0.0)))
    if n[2] > 0:
        n = -n
    return c, n, rms


def normals_from_points(pts):
    dx = cv2.Sobel(pts, cv2.CV_32F, 1, 0, ksize=7)
    dy = cv2.Sobel(pts, cv2.CV_32F, 0, 1, ksize=7)
    n = np.cross(dy.reshape(-1, 3), dx.reshape(-1, 3)).reshape(pts.shape)
    nn = np.linalg.norm(n, axis=2, keepdims=True)
    nn[nn == 0] = 1
    n = n / nn
    n[n[:, :, 2] > 0] *= -1
    return n


def segment(depth, intr, rgb=None):
    valid = (depth > VALID_MIN) & (depth < VALID_MAX)
    d = depth.copy()
    d[~valid] = 0
    d = cv2.medianBlur(d, 5)
    d = cv2.medianBlur(d, 5)          # ~median9, kills stereo ripple
    valid_s = valid & (d > 0)
    pts = deproject(d, intr)

    # jump edges: local depth range over 5x5
    k = np.ones((5, 5), np.uint8)
    dmax = cv2.dilate(d, k)
    dinf = d.copy(); dinf[~valid_s] = 10.0
    dmin = cv2.erode(dinf, k)
    rng = dmax - dmin
    thr = JUMP_THR * np.clip(d / 0.45, 1.0, 4.0) ** 2
    jump = (rng > thr) & valid_s

    # crease edges: normal deviates from local mean normal
    nrm = normals_from_points(pts)
    nrm_s = cv2.blur(nrm, (11, 11))
    nl = np.linalg.norm(nrm_s, axis=2, keepdims=True); nl[nl == 0] = 1
    nrm_s /= nl
    dot = np.clip((nrm * nrm_s).sum(axis=2), -1, 1)
    crease = (np.degrees(np.arccos(dot)) > CREASE_DEG) & valid_s

    ang_dev = np.degrees(np.arccos(dot))
    edges = jump | crease

    # colour prior: the bin is grey metal (low saturation, dim); boxes are
    # saturated (green/brown print) or very bright (white panels)
    box_col = np.zeros_like(jump)
    rgb_e = np.zeros_like(jump)
    grey_edge = np.zeros_like(jump)
    if rgb is not None:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        raw = (hsv[:, :, 1] > BOX_SAT_MIN) | (hsv[:, :, 2] > BOX_VAL_MIN)
        raw = cv2.medianBlur(raw.astype(np.uint8) * 255, 5)
        box_col = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                                   np.ones((5, 5), np.uint8)).astype(bool)
        # grey<->box transitions are near-certain blank outlines
        bc = box_col.astype(np.uint8)
        grey_edge = (cv2.dilate(bc, np.ones((3, 3), np.uint8)) -
                     cv2.erode(bc, np.ones((3, 3), np.uint8))).astype(bool)
        # generic RGB edges only within box-coloured areas (print seams);
        # kept low-weight: depth cues dominate
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 30, 5)
        rgb_e = cv2.dilate(cv2.Canny(gray, 40, 110),
                           np.ones((3, 3), np.uint8)).astype(bool) & box_col

    # --- marker-based watershed: edge chains rarely close into loops, so
    # plain flood fill leaks; watershed partitions everything instead ---
    free = (valid_s & ~(edges | rgb_e | grey_edge)).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    seeds = (dist > 4).astype(np.uint8)
    n_seed, mk = cv2.connectedComponents(seeds)
    # drop tiny seeds
    cnts = np.bincount(mk.ravel(), minlength=n_seed)
    kill = np.where(cnts < 150)[0]
    mk[np.isin(mk, kill)] = 0
    mk = mk + 1                       # background seed label becomes 1
    mk[(dist <= 4) & valid_s] = 0     # unknown -> to be flooded
    mk[~valid_s] = 1                  # invalid px stay "background"

    # depth terms weighted highest: depth-gradient structure is the most
    # reliable signature of the blanks; colour only assists
    elev = (np.clip(rng / np.maximum(thr, 1e-6), 0, 3) * 80 +
            np.clip(ang_dev / CREASE_DEG, 0, 3) * 70 +
            grey_edge * 90 + rgb_e * 25).clip(0, 255).astype(np.uint8)
    elev3 = cv2.merge([elev, elev, elev])
    mk32 = mk.astype(np.int32)
    cv2.watershed(elev3, mk32)
    labels = mk32.copy()
    labels[labels <= 1] = 0           # background/watershed lines -> 0
    labels[~valid_s] = 0
    n_lbl = labels.max() + 1
    return d, valid_s, jump, crease, labels, n_lbl, pts, nrm, box_col


class DSU:
    def __init__(self, items):
        self.p = {i: i for i in items}
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def analyze(depth_path, rgb_path, intr):
    depth = np.load(depth_path)
    rgb = cv2.imread(rgb_path)
    d, valid, jump, crease, labels, n_lbl, pts, nrm, box_col = segment(depth, intr, rgb)

    # ---- initial segments ----
    areas = np.bincount(labels.ravel(), minlength=n_lbl)
    segs = [l for l in range(1, n_lbl) if areas[l] >= MIN_SEG_AREA]
    masks = {l: labels == l for l in segs}
    # narrow bands: keeps the fold-crease plane distance small (sin(angle) *
    # band width) while a 4 mm stacked-blank offset stays fully visible;
    # 9 px also bridges thin occlusion-shadow stripes at folds
    dil = {l: cv2.dilate(masks[l].astype(np.uint8),
                         np.ones((9, 9), np.uint8)).astype(bool) for l in segs}

    # per-segment plane fits (for coplanarity-checked merging)
    seg_plane = {}
    rs = np.random.RandomState(0)
    for l in segs:
        p = pts[masks[l]]
        sub = p[rs.choice(len(p), min(len(p), 2500), replace=False)]
        seg_plane[l] = fit_plane(sub)

    # ---- pairwise border stats ----
    pair_stats = {}
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            band_ab = dil[a] & masks[b]
            n_ab = int(band_ab.sum())
            if n_ab < MIN_SHARED_BORDER:
                continue
            band_ba = dil[b] & masks[a]
            n_ba = int(band_ba.sum())
            if n_ba < MIN_SHARED_BORDER:
                continue
            da = float(np.median(d[band_ba]))
            db = float(np.median(d[band_ab]))
            pair_stats[(a, b)] = (da, db, min(n_ab, n_ba), band_ab, band_ba)

    # ---- merge only truly continuous, coplanar-at-the-border pairs ----
    # (a fold crease inside one blank passes; a flush contact between two
    #  different blanks usually fails the normal/plane-distance tests)
    def stacked(i, j):
        """near-parallel planes with a clear offset = two stacked blanks"""
        ci, ni, _ = seg_plane[i]
        cj, nj, _ = seg_plane[j]
        ang = np.degrees(np.arccos(min(1.0, abs(float(ni @ nj)))))
        if ang > 10.0:
            return False
        off = 0.5 * (abs(float((cj - ci) @ ni)) + abs(float((ci - cj) @ nj)))
        return off > 0.008

    dsu = DSU(segs)
    members = {l: [l] for l in segs}
    cand = []
    for (a, b), (da, db, w, band_ab, band_ba) in pair_stats.items():
        if abs(da - db) >= MERGE_GAP:
            continue
        ca, na, _ = seg_plane[a]
        cb, nb, _ = seg_plane[b]
        ang = np.degrees(np.arccos(min(1.0, abs(float(na @ nb)))))
        if ang > MERGE_ANGLE:
            continue
        # border points of B must lie on A's plane and vice versa
        pb = pts[band_ab]
        pa = pts[band_ba]
        dist_b = float(np.median(np.abs((pb - ca) @ na)))
        dist_a = float(np.median(np.abs((pa - cb) @ nb)))
        pd = max(dist_a, dist_b)
        if pd > MERGE_PLANE_DIST:
            continue
        cand.append((pd, a, b))
    # confident merges first; veto a union that would chain two segments
    # lying in parallel planes with a clear offset (stacked blanks)
    for pd, a, b in sorted(cand):
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            continue
        if any(stacked(i, j) for i in members[ra] for j in members[rb]):
            continue
        dsu.union(a, b)
        r = dsu.find(a)
        merged = members[ra] + members[rb]
        members[ra] = members[rb] = []
        members[r] = merged
    groups = {}
    for l in segs:
        groups.setdefault(dsu.find(l), []).append(l)

    comps = {}
    for g, members in groups.items():
        mask = np.zeros_like(masks[members[0]])
        for m_ in members:
            mask |= masks[m_]
        area = int(mask.sum())
        if area < MIN_GROUP_AREA:
            continue
        p = pts[mask]
        sub = p[np.random.RandomState(0).choice(len(p), min(len(p), 5000), replace=False)]
        c, n, rms = fit_plane(sub)
        tilt = np.degrees(np.arccos(min(1.0, abs(n[2]))))
        phys = float((d[mask] ** 2).sum() / (intr["fx"] * intr["fy"]))
        grey_frac = 1.0 - float(box_col[mask].mean())
        comps[g] = dict(mask=mask, members=members, area=area,
                        med_d=float(np.median(d[mask])), phys_area=phys,
                        centroid=c, normal=n, rms=rms, tilt=tilt,
                        grey_frac=grey_frac,
                        ok=(tilt < MAX_TILT_DEG and rms < MAX_PLANE_RMS
                            and grey_frac < MAX_GREY_FRAC))

    # ---- topness between groups ----
    gdil = {g: cv2.dilate(c_["mask"].astype(np.uint8),
                          np.ones((11, 11), np.uint8)).astype(bool)
            for g, c_ in comps.items()}
    for g in comps:
        comps[g].update(above_w=0.0, total_w=0.0, nbrs={})
    glist = list(comps)
    for i, a in enumerate(glist):
        for b in glist[i + 1:]:
            band_ab = gdil[a] & comps[b]["mask"]
            n_ab = int(band_ab.sum())
            if n_ab < MIN_SHARED_BORDER:
                continue
            band_ba = gdil[b] & comps[a]["mask"]
            n_ba = int(band_ba.sum())
            if n_ba < MIN_SHARED_BORDER:
                continue
            da = float(np.median(d[band_ba]))
            db = float(np.median(d[band_ab]))
            w = float(min(n_ab, n_ba))
            if da < db - ABOVE_MARGIN:
                res_a = 1.0
            elif db < da - ABOVE_MARGIN:
                res_a = 0.0
            else:
                res_a = 0.5
            comps[a]["above_w"] += w * res_a
            comps[a]["total_w"] += w
            comps[a]["nbrs"][b] = res_a
            comps[b]["above_w"] += w * (1.0 - res_a)
            comps[b]["total_w"] += w
            comps[b]["nbrs"][a] = 1.0 - res_a

    best, best_score = None, -1.0
    fallback, fb_score = None, -1.0
    for g, c_ in comps.items():
        af = c_["above_w"] / c_["total_w"] if c_["total_w"] > 0 else 0.5
        c_["above_frac"] = af
        score = c_["area"] * (af ** 3) if c_["ok"] else -1.0
        c_["score"] = score
        if score > fb_score:
            fallback, fb_score = g, score
        if c_["phys_area"] >= MIN_PHYS_AREA and score > best_score:
            best, best_score = g, score
    if best is None:
        best = fallback

    result = dict(comps=comps, labels=labels, jump=jump, crease=crease,
                  d=d, valid=valid, best=best, rgb=rgb, pts=pts,
                  box_col=box_col)
    if best is None:
        return result

    # ---- absorb split-off panels of the same blank: adjacent groups that
    # are box-coloured, level with the winner (neutral topness), depth-
    # continuous at the border and not a parallel-offset (stacked) plane.
    # A rectangularity prior arbitrates: blanks are rectangles, so a true
    # panel keeps the mask's min-area-rect fill ratio, a neighbouring
    # blank sticking out sideways lowers it. ----
    def rect_fill(mask_):
        mm = mask_.astype(np.uint8)
        cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return 0.0
        cnt_ = max(cs, key=cv2.contourArea)
        (rw, rh) = cv2.minAreaRect(np.vstack([c.reshape(-1, 2) for c in cs]))[1]
        return float(mask_.sum()) / max(rw * rh, 1.0)

    win_mask = comps[best]["mask"].copy()
    win_fill = rect_fill(win_mask)
    cb_, nb_ = comps[best]["centroid"], comps[best]["normal"]
    cand_abs = sorted(comps[best]["nbrs"].items(),
                      key=lambda kv: -comps[kv[0]]["area"])
    for g, res in cand_abs:
        cg = comps[g]
        if not (0.3 <= res <= 0.7):
            continue
        if cg["grey_frac"] > 0.5 or cg["tilt"] > MAX_TILT_DEG:
            continue
        ang = np.degrees(np.arccos(min(1.0, abs(float(nb_ @ cg["normal"])))))
        off = 0.5 * (abs(float((cg["centroid"] - cb_) @ nb_)) +
                     abs(float((cb_ - cg["centroid"]) @ cg["normal"])))
        if ang < 10.0 and off > 0.008:
            continue                      # stacked parallel blank, keep out
        band_ab = gdil[best] & cg["mask"]
        band_ba = gdil[g] & comps[best]["mask"]
        if band_ab.sum() < MIN_SHARED_BORDER or band_ba.sum() < MIN_SHARED_BORDER:
            continue
        if abs(float(np.median(d[band_ba])) -
               float(np.median(d[band_ab]))) > 0.004:
            continue
        trial = win_mask | cg["mask"]
        trial_fill = rect_fill(trial)
        if trial_fill >= win_fill - 0.04 or trial_fill > 0.75:
            win_mask, win_fill = trial, trial_fill

    # ---- refine winner mask: drop pixels far from its (bent-tolerant)
    # plane and grey (bin-coloured) leak pixels ----
    mask0 = win_mask & valid & box_col
    p_all = pts[mask0]
    rs = np.random.RandomState(0)
    sub = p_all[rs.choice(len(p_all), min(len(p_all), 6000), replace=False)]
    c, n, _ = fit_plane(sub)
    for _ in range(2):                       # robust re-fit
        dist_ = np.abs((sub - c) @ n)
        keep = dist_ < REFINE_DIST
        if keep.sum() < 100:
            break
        c, n, _ = fit_plane(sub[keep])
    dist_map = np.abs((pts.reshape(-1, 3) - c) @ n).reshape(mask0.shape)
    m = (mask0 & (dist_map < REFINE_DIST)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n_cc, cc = cv2.connectedComponents(m)
    if n_cc > 2:
        sizes = np.bincount(cc.ravel()); sizes[0] = 0
        m = (cc == sizes.argmax()).astype(np.uint8)
    comps[best]["mask_refined"] = m.astype(bool)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    result["contour"] = cnt
    result["img_rect"] = cv2.minAreaRect(cnt)

    mask_b = m.astype(bool) & valid
    p = pts[mask_b]
    sub = p[np.random.RandomState(0).choice(len(p), min(len(p), 6000), replace=False)]
    c, n, _ = fit_plane(sub)
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    q = p - c
    uv = np.stack([q @ e1, q @ e2], axis=1)
    uv_i = (uv * 1000).astype(np.int32)
    rect = cv2.minAreaRect(uv_i.reshape(-1, 1, 2))
    box_uv = cv2.boxPoints(rect) / 1000.0
    corners3d = c + box_uv[:, :1] * e1 + box_uv[:, 1:2] * e2
    result["plane_rect_size"] = (rect[1][0] / 1000.0, rect[1][1] / 1000.0)
    result["plane_corners_img"] = project(corners3d, intr)
    result["plane_normal"] = n
    return result


# ---------------- visualisation ----------------
def colorize_depth(d, valid):
    dv = d.copy(); dv[~valid] = 0
    lo, hi = 0.25, 1.0
    dn = np.clip((dv - lo) / (hi - lo), 0, 1)
    dc = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dc[~valid] = 0
    return dc


def overlay(res, name):
    rgb = res["rgb"]
    h, w = rgb.shape[:2]
    comps, best = res["comps"], res["best"]

    p1 = rgb.copy()
    if best is not None:
        cv2.drawContours(p1, [res["contour"]], -1, (0, 255, 255), 2)
        box = cv2.boxPoints(res["img_rect"]).astype(np.int32)
        cv2.polylines(p1, [box], True, (255, 255, 0), 2)
        pc = res["plane_corners_img"].astype(np.int32)
        cv2.polylines(p1, [pc], True, (0, 255, 0), 3)
        sw, sh = res["plane_rect_size"]
        c_ = comps[best]
        txt = f"{sw*100:.1f} x {sh*100:.1f} cm  z={c_['med_d']:.3f}m  top={c_['above_frac']:.2f}"
        cv2.putText(p1, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(p1, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(p1, "NO CANDIDATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    p2 = colorize_depth(res["d"], res["valid"])
    p2[res["crease"]] = (200, 200, 200)
    p2[res["jump"]] = (255, 255, 255)

    rng = np.random.RandomState(42)
    p3 = np.zeros_like(rgb)
    for g, c_ in comps.items():
        col = tuple(int(x) for x in rng.randint(60, 255, 3))
        p3[c_["mask"]] = col
    for g, c_ in comps.items():
        ys, xs = np.nonzero(c_["mask"])
        cx, cy = int(xs.mean()), int(ys.mean())
        tag = f"{c_['above_frac']:.2f}" + ("" if c_["ok"] else
              (" G" if c_["grey_frac"] >= MAX_GREY_FRAC else " X"))
        cv2.putText(p3, tag, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(p3, tag, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if best is not None:
        cm = comps[best]["mask"].astype(np.uint8)
        cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p3, cnts, -1, (255, 255, 255), 2)

    p4 = (rgb * 0.35).astype(np.uint8)
    if best is not None:
        mm = comps[best]["mask"]
        p4[mm] = (0.35 * rgb[mm] + np.array([0, 160, 0])).clip(0, 255).astype(np.uint8)
        pc = res["plane_corners_img"].astype(np.int32)
        cv2.polylines(p4, [pc], True, (0, 255, 0), 2)

    for p, t in [(p1, "result"), (p2, "depth+edges"), (p3, "groups/topness"), (p4, "chosen mask")]:
        cv2.putText(p, t, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])


if __name__ == "__main__":
    import sys, glob, time
    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, "dataset", "two_lights")
    out = os.path.join(base, "overlays")
    os.makedirs(out, exist_ok=True)
    intr = json.load(open(os.path.join(data, "intrinsics.json")))
    frames = sys.argv[1:] or sorted(
        os.path.basename(f)[:3] for f in glob.glob(os.path.join(data, "*_rgb.png")))
    t0 = time.time()
    all_results = {}
    for f in frames:
        res = analyze(os.path.join(data, f + "_depth.npy"),
                      os.path.join(data, f + "_rgb.png"), intr)
        cv2.imwrite(os.path.join(out, f + "_overlay.png"), overlay(res, f))
        b = res["best"]
        if b is not None:
            c_ = res["comps"][b]
            sw, sh = res["plane_rect_size"]
            (cx, cy), (rw, rh), rang = res["img_rect"]
            all_results[f] = {
                "found": True,
                "img_rect": {"center": [cx, cy], "size": [rw, rh], "angle_deg": rang},
                "plane_rect_corners_px": res["plane_corners_img"].tolist(),
                "plane_rect_size_m": [sw, sh],
                "plane_normal": res["plane_normal"].tolist(),
                "median_depth_m": c_["med_d"],
                "above_frac": c_["above_frac"],
                "mask_area_px": c_["area"],
            }
            print(f"{f}: groups={len(res['comps'])} best area={c_['area']} top={c_['above_frac']:.2f} "
                  f"tilt={c_['tilt']:.0f} rms={c_['rms']*1000:.1f}mm box={sw*100:.1f}x{sh*100:.1f}cm")
        else:
            all_results[f] = {"found": False}
            print(f"{f}: NO CANDIDATE ({len(res['comps'])} groups)")
    with open(os.path.join(out, "results.json"), "w") as fp:
        json.dump(all_results, fp, indent=1)
    print(f"avg {(time.time()-t0)/len(frames)*1000:.0f} ms/frame")
