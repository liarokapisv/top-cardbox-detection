# top-cardbox-detection

Detects the **topmost flattened cardbox blank** in a bin from RealSense
D435 RGB-D frames. The blank is modelled as its true physical structure:
**3 rigid faces joined by 2 edge-revolute hinges**, whose outline is an
**8-point polygon** (full-width middle panel ~29x12 cm + two narrower
flaps; one flap's visible height is a per-frame fold state).

## Components

| file | role |
|---|---|
| `pipeline.py` | classical RGB-D proposal stage (depth-gradient edges, watershed, coplanar merge, topness voting), ~0.35 s/frame CPU |
| `l_fit.py` | **8-gon template fit**: chamfer-matches the canonical 3-face polygon over rotation/mirror/scale/fold-state against plane-space compatibility maps. Outputs the 8-point boundary, per-face quads (px + 3D), middle-face box, pose. The labelling engine and geometric authority. ~2.4 s/frame CPU |
| `sam_refine.py` | optional SAM 2.1-tiny hybrid masker (prompted by classical proposals, depth-scored) |
| `da_depth.py` | **Depth-Anything-V2 (ONNX, CPU) top-box masker**: RGB->relative depth, floor-detrend, colour-prior box regions split at DA depth steps + ranked by relief for topness, then `l_fit` for the 8-gon. ~2 s/frame CPU, no GPU/torch. See below. |
| `synth3.py` | hinge-articulated synthetic scene generator (empty-bin background) |
| `synth_realbg.py` | real-background composites: articulated boxes pasted onto real frames — top box by construction, perfect labels, minimal domain gap |
| `train.py` / `train2.py` | UNet definition + CUDA-ready trainer (AMP, auto device, rotation-heavy augmentation, resumable: `python3 train2.py <epochs> [resume]`) |
| `finetune.py` | fine-tune on real frames using `l_fit` auto-labels |
| `eval_real.py` / `eval2.py` | real-frame evaluation (eval2: 4x flip TTA) |
| `faces_first.py` | face-first classical experiment (documented negative result) |
| `models/` | trained checkpoints |
| `results/` | per-frame 8-gon + face quads (`lfit_results.json`), masks, contact sheets |
| `cutouts/` | rectified box cutouts (both print sides) used by the generators |

## Quick start (Docker, CUDA)

```bash
docker build -t topcardbox .
# generate synthetic data + train on GPU:
docker run --gpus all -v /path/to/two_lights:/app/dataset/two_lights -it topcardbox bash
  python3 synth3.py synth_train 2000 11 && python3 synth3.py synth_val 100 99
  python3 synth_realbg.py synth_rbg_train 1000 21 && python3 synth_realbg.py synth_rbg_val 60 77
  python3 train2.py 60            # CUDA + AMP auto-enabled, BATCH=32 default
  python3 finetune.py
  python3 eval2.py topbox_unet2.pt ch24
# label new frames with the geometric engine (no model needed):
  python3 l_fit.py
```

Works on CPU unchanged (auto-fallback, BATCH=8).

## Depth-Anything-V2 top-box masking (CPU, no GPU)

The captured D435 depth cannot delineate a *flat-lying* blank: the bin floor
is bowed ~2 cm (stereo distortion) and per-pixel noise is ~3 mm, while a
blank is ~3 mm thick. Depth-Anything-V2 predicts RGB-only relative depth
with crisp, appearance-aligned boundaries, and its ViT-S runs ~2 s/frame on
CPU. `da_depth.py` uses DA relief to (a) rank overlapping blanks by nearness
= topness and (b) split a stack at the depth step, gated by the existing HSV
box/bin colour prior, then reuses `l_fit.fit_frame` for the metric 8-gon.

```bash
pip install -r requirements-da.txt      # onnxruntime, no torch
python3 da_depth.py 009                 # one frame -> overlays_da/009_da.png
python3 da_depth.py --compare           # all frames + contact_sheet_da.jpg
```

Weights (`onnx-community/depth-anything-v2-small`, ~99 MB) auto-download to
`models/da2/` on first run. Each `overlays_da/NNN_da.png` panel is
RGB | DA depth | floor-relief | mask+8-gon; results in
`overlays_da/da_results.json` (same schema as `lfit_results.json`). All 35
blanks fit (028 is the empty bin).

`eval_gt.py` scores the *chosen* top box against the hand ground truth
(`dataset/two_lights/gt.json`, the top-box point per frame):

```bash
python3 eval_gt.py overlays_da/da_results.json   # 24/35 correct top box
```

**Two-light occlusion cut (`DA_SHADOW=1`).** Flush-stacked identical blanks
have no DA depth step, but the two lamps cast a thin shadow at the upper
blank's cut edge. Setting `DA_SHADOW=1` adds those shadow/shading ridges to
the split step, raising correct top-box picks to **26/35** (hard misses
6->4). It can over-cut a heavily *folded* single blank (its fold creases
mimic occlusion shadows), so it is opt-in. Distinguishing intra-blank folds
from inter-blank occlusion at the cut stage is the open problem (`l_fit`
resolves it at the fit stage via the L template).

## Recommended production recipe

1. `l_fit.py` auto-labels every new bin image (no manual annotation).
2. Retrain `train2.py` + `finetune.py` as real data accumulates.
3. Runtime: UNet mask (~100 ms CPU, ~ms GPU) -> `l_fit`-style pose fit on
   the mask -> 8-gon boundary, 3 face quads, middle-face box.
