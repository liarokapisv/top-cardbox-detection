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

## Recommended production recipe

1. `l_fit.py` auto-labels every new bin image (no manual annotation).
2. Retrain `train2.py` + `finetune.py` as real data accumulates.
3. Runtime: UNet mask (~100 ms CPU, ~ms GPU) -> `l_fit`-style pose fit on
   the mask -> 8-gon boundary, 3 face quads, middle-face box.
