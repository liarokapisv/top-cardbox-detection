#!/usr/bin/env python3
"""Score a results JSON (da_results.json / lfit_results.json) against the
hand ground truth in dataset/two_lights/gt.json: is the ground-truth top-box
point inside the chosen 8-gon?  A miss where an `alternate` box point lands
inside is flagged separately (a plausible but non-top pick).

usage: python3 eval_gt.py overlays_da/da_results.json
"""
import json
import sys
import numpy as np
import cv2

GT = "dataset/two_lights/gt.json"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "overlays_da/da_results.json"
    gt = json.load(open(GT))
    res = json.load(open(path))
    hits = miss = nofit = 0
    bad = []
    for f, g in gt.items():
        if not g or g.get("primary") is None:      # empty bin
            continue
        prim = tuple(map(float, g["primary"]))
        r = res.get(f, {})
        if not r.get("found"):
            nofit += 1
            bad.append(f + "(nofit)")
            continue
        poly = np.array(r["polygon_px"], np.int32)
        if cv2.pointPolygonTest(poly, prim, False) >= 0:
            hits += 1
            continue
        alts = [tuple(map(float, a)) for a in (g.get("alternates") or [])]
        alt_in = any(cv2.pointPolygonTest(poly, a, False) >= 0 for a in alts)
        miss += 1
        bad.append(f + ("(alt)" if alt_in else "(MISS)"))
    tot = hits + miss + nofit
    print(f"{path}: {hits}/{tot} correct top box  ({miss} wrong, {nofit} nofit)")
    if bad:
        print("  wrong:", " ".join(bad), "   [alt = picked a valid non-top box]")


if __name__ == "__main__":
    main()
