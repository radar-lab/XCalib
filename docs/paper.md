# Paper validation and evidence

Lab-side validation for _Position Encoding in Detection-Based LiDAR–Camera Matching: A Diagnostic
Study at Infrastructure Sites_ (accepted at IEEE Sensors Letters): how to reproduce the public A9
Table I column, verify UTC reference numbers when institutional data are available, and capture the
Thor latency numbers. Frozen evaluation summaries and how they map to supplemental tables live in
[the evidence index](evidence/index.md).

The pairwise baseline `calibrefine` is ported from
[CalibRefine](https://github.com/radar-lab/Lidar_Camera_Automatic_Calibration) (Cheng et al., IEEE
TIM 2026, [arXiv:2502.17648](https://arxiv.org/abs/2502.17648)); citation entries for both papers
are in the repo README.

## 1. Paper validation

```bash
pixi run validate-a9      # public A9 r02_s01 release

# UTC reference verification; requires local institutional HDF5 caches.
pixi run validate-utc4
pixi run validate-utc3
```

The A9 command is the public reproducibility path for the release. UTC4/UTC3 commands verify the
same Table I reference numbers only for partners who already hold compatible institutional HDF5
caches and weights; those raw UTC artefacts are not distributed in this repository. All commands
evaluate the held-out **test** split used by the paper experiments. The A9 r02_s01 caches ship inside
this repo at `datasets/a9_dataset_r02_s01/hdf5_cache/a9_r02_s01_{train,val,test}.h5` (and on the public HF
dataset repo). The training pipeline reserves the `_val.h5` split for early-stopping /
best-checkpoint selection, so `_test.h5` is the only file in each cache that is never seen during
training.

**A9 note (multi-camera frames).** A9 r02_s01 frames carry detections from two cameras
(`s110_camera_basler_south1_8mm`, `s110_camera_basler_south2_8mm`); the loader returns every
referenced camera image plus a per-detection source-camera array, and the cropping path picks each
2-D box's own camera (`prepare_frame(..., images=, camera_per_det=)`) exactly like the paper-side
`consistent_loader`. The A9 configs also use `bbox_expansion: 1.25` (UTC uses 1.05), mirroring the
per-site lab dataset configs.

By default the script evaluates the 5 paper-claimed methods only (`calibrefine`, `crlite`,
`crlite_2dpe`, `crlite_vit_exp1`, `crlite_vit_exp3`). To also include the delivery-only
`crlite_vit_exp4` or to customize the H5 path / write a JSON report:

```bash
pixi run python scripts/paper/validate_paper.py \
    --site utc4 \
    --hdf5 datasets/utc_dataset4/hdf5_cache/utc_test.h5 \
    --models crlite crlite_2dpe crlite_vit_exp1 crlite_vit_exp3 crlite_vit_exp4 calibrefine \
    --output docs/evidence/standalone_paper_validation_utc4.json
```

`--output` writes a JSON file containing `site`, `hdf5` (absolute path of the H5 actually used),
`device`, and a `results` dict keyed by model name — each value has `top1`, `top3`, `mrr`,
`latency_ms_mean`, `latency_ms_p50`, `latency_ms_p95`, `latency_ms_p99`, `latency_ms_std`,
`latency_ms_min`, `latency_ms_max`, `throughput_fps_mean`, `n_iters`, `n_frames`, and
`wall_seconds`. Parent directories are created automatically.

### 1.1 UTC reference numbers (RTX 5090, `utc_test.h5`, 50 frames per site)

Canonical reference numbers captured in `docs/evidence/standalone_paper_validation_utc{3,4}.json` on
**2026-05-16** with the retrained calibrefine UTC4 weights. They are frozen for manuscript
reconciliation; only partners with local UTC data should expect to rerun the same commands. Latency
is end-to-end model forward (no I/O); fps is
`throughput_fps_mean` from the JSON. Top-1 / Top-3 / MRR are stable within ±0.3 pp across runs (CUDA
non-determinism); latency drifts ±20 % depending on GPU thermal state, which is why we also report
p50 / p95 / p99 in the JSON.

**UTC4**

| Model             | Top-1 |  Top-3 |   MRR |  mean lat |    p95 |   fps |   Paper Table I |
| ----------------- | ----: | -----: | ----: | --------: | -----: | ----: | --------------: |
| `crlite_2dpe`     | 99.75 | 100.00 | 0.999 |    7.6 ms |   10.3 | 132.1 | ≈97.9 (PE abl.) |
| `crlite`          | 99.02 |  99.75 | 0.994 |   10.9 ms |   17.2 |  91.7 |            99.0 |
| `crlite_vit_exp3` | 97.54 |  99.26 | 0.984 |    8.3 ms |   13.0 | 119.9 |            98.0 |
| `crlite_vit_exp1` | 96.81 |  99.26 | 0.980 |    7.8 ms |   13.6 | 127.7 |            96.8 |
| `calibrefine`     | 97.54 | 100.00 | 0.988 | 1093.8 ms | 1689.4 |   0.9 |            97.5 |

**UTC3**

| Model             | Top-1 |  Top-3 |   MRR |  mean lat |    p95 |   fps | Paper Table I |
| ----------------- | ----: | -----: | ----: | --------: | -----: | ----: | ------------: |
| `crlite_2dpe`     | 97.84 | 100.00 | 0.989 |    9.6 ms |   14.3 | 103.8 |           n/a |
| `crlite`          | 96.46 |  99.41 | 0.979 |   11.5 ms |   22.6 |  86.7 |          94.6 |
| `crlite_vit_exp1` | 92.53 |  98.43 | 0.956 |    5.6 ms |    7.6 | 178.5 |          90.5 |
| `calibrefine`     | 92.14 |  99.02 | 0.956 | 1196.2 ms | 1892.0 |   0.8 |          91.9 |
| `crlite_vit_exp3` | 91.75 |  99.61 | 0.956 |    5.2 ms |    7.3 | 191.6 |          89.9 |

### 1.1a Public A9 numbers (RTX 5090, `a9_r02_s01_test.h5`, 29 frames, captured 2026-06-11)

Canonical public A9 numbers in `docs/evidence/standalone_paper_validation_a9_dataset_r02_s01.json` from
`pixi run validate-a9`. "Paper Table I" is the manuscript's per-frame Top-1, while the standalone
accumulator micro-averages over detections (same definition gap that explains the UTC3 offsets
above), so agreement within ~2 pp is expected and observed.

| Model             | Top-1 |  Top-3 |   MRR |  mean lat |  fps |  Paper Table I |
| ----------------- | ----: | -----: | ----: | --------: | ---: | -------------: |
| `calibrefine`     | 98.07 |  99.68 | 0.989 | 1956.8 ms |  0.5 |           98.4 |
| `crlite_2dpe`     | 97.43 | 100.00 | 0.987 |   25.4 ms | 39.3 | 97.5 (PE abl.) |
| `crlite`          | 91.96 | 100.00 | 0.959 |   32.6 ms | 30.6 |           92.7 |
| `crlite_vit_exp1` | 90.35 |  97.43 | 0.942 |   13.3 ms | 75.1 |           92.4 |
| `crlite_vit_exp3` | 90.03 |  99.04 | 0.945 |   11.6 ms | 86.2 |           89.8 |

The paper's A9 ordering is preserved: pairwise > 2D-only > ResNet + 3D PE, and ViT (no PE) ≥ ViT +
3D PE. A9 latencies are higher than UTC because the A9 gantry frames contain more detections per
frame (latency is per frame, not per pair).

### 1.2 ONNX accuracy validation (lossless-export check)

Section 1.1 reproduces the paper numbers with the **PyTorch** model. For the partner deployment we
need to confirm that the **ONNX** artifact they consume on Thor still computes the same matching
quality on the same `utc_test.h5` split — not just per-tensor parity on a synthetic batch (which the
export-time check at `max|torch − onnx| ≈ 1e-6` already covers).

```bash
pixi run validate-onnx-utc4
pixi run validate-onnx-utc3
```

Each task loads the graphs from `onnx/<model>_<site>/` (per-site dir introduced 2026-05-17 — see
§2.1 for the silent-mismatch bug it fixes), drives them with `onnxruntime` (GPU on dev, CPU on CI /
Thor), and feeds them the same `utc_test.h5` frames `validate-utc{3,4}` use. Output:

- `docs/evidence/standalone_onnx_validation_utc{3,4}.json` — same schema as the PyTorch validation JSON
  (Top-1 / Top-3 / MRR / latency stats per model, plus `providers`, `hdf5`, and `onnx_dirs`
  metadata; the last field records which on-disk ONNX directory each row was loaded from, so a
  `_utc3` validator that silently picked up `_utc4` weights is immediately visible in the JSON).
- A pretty-printed per-model row table on stdout.

Direct comparison against `docs/evidence/standalone_paper_validation_utc{3,4}.json` should show
**agreement to within ±0.3 pp on Top-1 / Top-3** (FP32 vs FP32, sole sources of drift are ORT graph
fusions and CUDA non- determinism). Anything larger means the ONNX export silently changed behaviour
— the script is the regression-guard for that.

The script knows about all four control flows we ship — two-stage hybrid (`crlite`, `crlite_2dpe`),
single-graph cosine (`crlite_vit_exp1`, `crlite_vit_exp3`), GNN cosine (`crlite_vit_exp4`), and
pairwise (`calibrefine`) — and re-implements each in numpy + ORT calls. No PyTorch op runs on the
eval path. To override the provider, model set, or H5:

```bash
pixi run python scripts/paper/validate_onnx.py --site utc4 \
    --provider cpu --models crlite calibrefine \
    --output reports/onnx_smoke.json
```

CPU is fine for `crlite` / `crlite_2dpe` / ViT variants (~0.2 s/frame). `calibrefine` is pairwise so
it has to issue N×M ORT calls (up to 1024 per frame); CPU is OK for a smoke run but use
`--provider cuda` or `auto` for the full 50-frame split unless you don't mind a ~10 minute wall.

**Reference numbers (ORT-CPU, FP32 graph, captured 2026-05-17 in
`docs/evidence/standalone_onnx_validation_utc{3,4}.json`):**

| Model             | UTC3 ONNX | UTC3 PyTorch | UTC4 ONNX | UTC4 PyTorch |
| ----------------- | --------: | -----------: | --------: | -----------: |
| `crlite_2dpe`     |   97.84 % |      97.84 % |   99.75 % |      99.75 % |
| `crlite`          |   96.46 % |      96.46 % |   99.02 % |      99.02 % |
| `crlite_vit_exp1` |   92.53 % |      92.53 % |   96.81 % |      96.81 % |
| `crlite_vit_exp3` |   91.75 % |      91.75 % |   97.54 % |      97.54 % |
| `calibrefine`     |   91.94 % |      92.14 % |   97.54 % |      97.54 % |

Every row matches PyTorch FP32 to the rounding floor on both sites (the single 0.20 pp `calibrefine`
UTC3 row is within CUDA non-determinism). The ONNX export is matching-quality-preserving on both the
deployed UTC3 and UTC4 weights — this is the headline claim cited in supplemental §S-7.

### 1.3 What this does _not_ check

- **TRT FP16 quantisation effect.** Both §1.1 and §1.2 are FP32 evaluations (PyTorch FP32 and
  ONNX-Runtime FP32 respectively). The shipped engine on Thor is FP16. We have not yet wired a Top-1
  / Top-3 / MRR check on top of the TRT FP16 engine itself, in part because doing it cleanly on Thor
  needs `utc_test.h5` to live there too. If the partner ever sees Top-1 drop more than ~1 pp on real
  data we should add that script (sketch: `scripts/validate_trt.py`, mirror `validate_onnx.py` but
  use `tensorrt.Runtime` instead of `onnxruntime.InferenceSession`).
- **Throughput numbers in §1.1 / §1.2** are wall-time PyTorch / ORT numbers and should _not_ be
  cited as edge latency. The Thor numbers for the paper / supplemental come from §3
  (`benchmark_thor.py`, PyTorch eager) and §2.3(a) (`bench-trt`, TRT FP16) only.

## 2. ONNX export and TensorRT engines for Thor

### 2.1 Export ONNX

All ONNX / engine / report artifacts are **site-tagged** so the UTC3 and UTC4 weight sets cannot
silently overwrite each other (an earlier issue: a UTC4-only export was unintentionally evaluated on
UTC3 frames and produced 13–28 % Top-1 noise; per-site subdirs are the structural fix):

```
onnx/<model>_<site>/{stage1,stage2,model}.onnx
engines/<model>_<site>/{stage1,stage2,model}.<precision>.engine
docs/evidence/standalone_onnx_validation_<site>.json
docs/evidence/thor_trt_latency_<site>.json
```

```bash
# Single model, default site (utc4):
pixi run export-onnx                       # crlite -> onnx/crlite_utc4/

# Both sites, all 6 models in one go (cleanest entry point):
pixi run export-onnx-all                    # = utc4-all + utc3-all

# Per-site batch:
pixi run export-onnx-utc4-all              # writes onnx/<model>_utc4/
pixi run export-onnx-utc3-all              # writes onnx/<model>_utc3/

# Explicit override:
pixi run python scripts/paper/export_onnx.py \
    --model crlite --site utc3 \
    --weights checkpoints/crlite_utc3_best.pth \
    --output onnx/crlite_utc3
```

The same invocation works on both the lab workstation and Thor itself — the `pixi install` step
picks the right torch wheel for the host platform, so the command line stays identical.

**Back-compat:** the resolver in `validate_onnx.py` / `build_trt.py` / `bench_trt.py` still falls
back to the legacy site-less layout (`onnx/<model>/`, `engines/<model>/`) **only** when
`--site=utc4` is requested and the per-site dir doesn't exist. Any UTC3 lookup requires the per-site
dir, so the silent-mismatch bug cannot recur.

For `crlite` / `crlite_2dpe` this writes two graphs:

- `stage1.onnx` — backbones + position encoding + cosine retrieval matrix.
- `stage2.onnx` — embed-fusion + similarity-head MLP for Top-K refinement.

For the ViT variants (`crlite_vit_exp1`, `crlite_vit_exp3`, `crlite_vit_exp4`) a single `model.onnx`
is written that directly returns the N×M cosine similarity matrix. `calibrefine` exports as one
pairwise graph indexed by batch B (one logit per (i, j) pair).

### 2.2 Build TensorRT engines

All graphs have dynamic axes on detection counts (N, M, K, B). The helper task auto-locates
`trtexec` (it ships at `/usr/src/tensorrt/bin/trtexec` on JetPack 7 / Thor and is **not** on `$PATH`
by default) and passes the right `min/opt/max` shape profile for each model:

```bash
# Both sites, paper-validated models only, FP16 (default precision):
pixi run build-trt-all                     # = utc4-all + utc3-all

# Per-site batch (writes engines/<model>_<site>/):
pixi run build-trt-utc4-all
pixi run build-trt-utc3-all

# Single model + site:
pixi run build-trt --model crlite --site utc4 --precision fp16

# Partner deliverable that also wants the ViT+GNN engine:
pixi run python scripts/paper/build_trt.py --all-incl-delivery --site utc4 \
    --log-dir reports/trt_logs

# Override trtexec location explicitly if it lives somewhere else:
pixi run python scripts/paper/build_trt.py --model crlite --site utc4 \
    --trtexec /opt/nvidia/tensorrt/bin/trtexec
```

**Scope note (`--all` semantics):** `--all` builds only the 5 paper-validated models (`crlite`,
`crlite_2dpe`, `crlite_vit_exp1`, `crlite_vit_exp3`, `calibrefine`). The 6th shipped model
`crlite_vit_exp4` (ViT + Edge-Aware GAT, delivery-only) is **not** included because its
TorchScript-traced ONNX uses ScatterND with dynamic axes which TensorRT 10.x on Blackwell does not
yet support; empirically the build fails on Thor under JetPack 7 for both UTC3 and UTC4 weights,
with no impact on the paper. If you need the vit_exp4 engine for partner delivery, request it
explicitly with `--all-incl-delivery` (or `--model crlite_vit_exp4`); the failure mode is the same
regardless of site.

Engines land at `engines/<model>_<site>/<stage>.<precision>.engine`. trtexec's own latency summary
is printed at the end of each build; it's the canonical TensorRT-FP16 number we cite in the
supplemental.

INT8 calibration is left to the partner — pass extra trtexec args through after `--`, e.g.

```bash
pixi run build-trt --model crlite --precision fp16 -- \
    --int8 --calib=/path/to/crops.cache --workspace=8192
```

If `trtexec` cannot be located the script prints an actionable error listing the directories it
searched.

### 2.3 Use the engine

`pixi run build-trt-all` writes `engines/<model>_<site>/{stage1,stage2,model}.fp16.engine`. There
are three ways to put those engines to work, depending on what you need.

**(a) Canonical Thor-TRT-FP16 latency for the paper.** Re-runs trtexec on every built engine with
the opt-profile shapes (N=8, M=12, B=80 for crlite stage2, B=96 for calibrefine pairs) and parses
the `GPU Compute Time` block out of stdout into a single JSON. This is what we cite in the
supplemental table.

```bash
pixi run bench-trt                    # = bench-trt-utc4 (default site)
pixi run bench-trt-utc4               # explicit; engines/<model>_utc4/
pixi run bench-trt-utc3               # explicit; engines/<model>_utc3/
pixi run bench-trt --model crlite     # one model on utc4
pixi run bench-trt --iterations 1000  # tighter percentiles
pixi run bench-trt -- --useCudaGraph  # forward extra flags to trtexec
```

Outputs:

- `docs/evidence/thor_trt_latency_<site>.json` —
  `{model: [{stage, precision, latency_ms_{mean,p50,p95,p99,min,max}, throughput_fps_mean}, ...]}`.
  For `--site utc4` we also keep the legacy site-less alias `docs/evidence/thor_trt_latency.json` so the
  existing supplemental table doesn't have to be re-pathed.
- `reports/trt_logs/<model>_<site>_<stage>.fp16.bench.log` — full trtexec stdout per engine.

**Reference Thor numbers (JetPack 7, FP16, N=8 / M=12 dets, warmup=50, iters=200; captured
2026-05-16/17 in `docs/evidence/thor_trt_latency.json` (UTC4) and `docs/evidence/thor_trt_latency_utc3.json`
(UTC3)):**

| Model             | Stage(s)      | UTC3 mean | UTC3 p99 | UTC3 thr. | UTC4 mean | UTC4 p99 | UTC4 thr. |
| ----------------- | ------------- | --------: | -------: | --------: | --------: | -------: | --------: |
| `crlite_2dpe`     | stage1+stage2 | **0.645** |     0.72 |  1550 fps | **0.640** |     0.74 |  1565 fps |
| `crlite`          | stage1+stage2 | **0.685** |     0.79 |  1459 fps | **0.700** |     0.81 |  1430 fps |
| `crlite_vit_exp1` | model         | **0.924** |     1.03 |  1082 fps | **0.925** |     1.03 |  1081 fps |
| `crlite_vit_exp3` | model         | **0.934** |     1.04 |  1071 fps | **0.926** |     1.02 |  1080 fps |
| `calibrefine`     | model (B=96)  |     334.7 |    345.0 |  2.99 fps |     342.6 |    370.9 |  2.92 fps |

The `crlite` two-stage column is `stage1 + stage2`; on either site stage 1 alone is ~0.64 ms
(~1530–1566 fps) and stage 2 alone is ~0.046 ms (~21 700–22 300 fps), so end-to-end is dominated by
stage 1. Both sites agree to within 2.3 % — engines share topology so latency is essentially
weight-independent. These are the canonical FP16 Thor numbers cited in supplemental §S-7 (Table
S-VII).

If you also want a sanity-check **without** building the engine first, trtexec can do it all in one
shot:

```bash
trtexec --loadEngine=engines/crlite_utc4/stage1.fp16.engine \
        --shapes=image_crops:8x3x32x32,lidar_crops:12x1024x3,img_centers:8x2,lid_centers:12x3 \
        --iterations=200 --warmUp=50 --useSpinWait --noDataTransfers
```

**(b) ONNX-Runtime smoke test (any host, no TRT needed).** The exported ONNX graphs are validated
against PyTorch at export time (`max|torch-onnx| ≈ 1e-6`) and again at the matching-quality level by
**§1.2** (`pixi run validate-onnx-utc{3,4}`, full Top-1 / Top-3 / MRR on `utc_test.h5`). If you just
want a programmatic smoke — for example to sanity check a checkpoint update on a partner box without
rebuilding the engine — call `onnxruntime` directly. It's already in the pixi environment.

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession(
    "onnx/crlite_utc4/stage1.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
out = sess.run(
    None,
    {
        "image_crops": np.random.randn(8, 3, 32, 32).astype(np.float32),
        "lidar_crops": np.random.randn(12, 1024, 3).astype(np.float32),
        "img_centers": np.random.randn(8, 2).astype(np.float32),
        "lid_centers": np.random.randn(12, 3).astype(np.float32),
    },
)
img_embed, lid_embed = out                              # [8, 256], [12, 256]
similarity = (img_embed / np.linalg.norm(img_embed, axis=1, keepdims=True)) @ \
             (lid_embed / np.linalg.norm(lid_embed, axis=1, keepdims=True)).T
```

For the cosine-only and pairwise models (`crlite_vit_exp1`, `crlite_vit_exp3`, `crlite_vit_exp4`,
`calibrefine`) there is just one graph (`onnx/<model>_<site>/model.onnx`) and the output is the
similarity matrix or per-pair logit directly.

**(c) Production deployment via Python TensorRT.** This is what the deployment pipeline actually
consumes. We don't ship a TRT runtime (it is partner-side glue), but the integration is small — load
the engine, allocate buffers, set dynamic shapes, run, copy back:

```python
import numpy as np, tensorrt as trt
import pycuda.driver as cuda, pycuda.autoinit  # or cudart-py

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open("engines/crlite_utc4/stage1.fp16.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()

# Set the actual N / M for this frame (must lie in min..max of the build profile)
ctx.set_input_shape("image_crops", (8, 3, 32, 32))
ctx.set_input_shape("lidar_crops", (12, 1024, 3))
ctx.set_input_shape("img_centers", (8, 2))
ctx.set_input_shape("lid_centers", (12, 3))

# ... allocate device buffers, copy inputs in, run ctx.execute_v3(stream),
# copy outputs out.  See NVIDIA's `samples/python/onnx_resnet50` for the
# full buffer-management boilerplate; the contract here is identical, only
# the input names differ.
```

For `crlite` / `crlite_2dpe` you run **stage1** to get the `(N, D)` / `(M, D)` embeddings, do an N×M
cosine + top-K on the host (milliseconds, not worth a kernel), and feed the K selected pairs into
**stage2**'s `(B, D) × (B, D) → (B, 1)` engine. The pairwise wrapper in
`xcalib.engine.wrappers.HybridTwoStageWrapper` is the torch-side reference for that orchestration;
mirror its control flow 1-to-1 in C++/Python around the two engines.

## 3. Thor latency benchmark

Matching-quality validation (Top-1 / Top-3 / MRR) is done on the lab workstation, where we keep the
full UTC HDF5 caches. The Jetson AGX Thor's role in the deliverable is exclusively to **back up the
real-time-feasibility claim** — i.e. that these lightweight models can keep up with a 10 Hz
fused-perception pipeline on the partner's iGPU.

```bash
pixi run benchmark            # all 6 shipped models, warmup=20, iters=200
                              # writes reports/thor_latency.json + .png

pixi run benchmark-quick      # warmup=5, iters=30 -- fast sanity run

# Single model with explicit weights:
pixi run python scripts/paper/benchmark_thor.py --model crlite_vit_exp4 \
    --weights checkpoints/crlite_vit_exp4_utc4_best.pth
```

The script uses synthetic frames (random crops + point clouds at the right shapes), so no HDF5 cache
is required on the Thor. It reports per-model FP32 + FP16 mean / p50 / p95 / p99 / std / min / max /
fps across the timed iterations, and the matplotlib bar chart annotates each model with its FPS and
overlays the 100 ms (10 Hz) real-time budget line. The pairwise baseline `calibrefine` is shown
alongside the lightweight models so the figure tells the speed story end-to-end.

### 3.1 Reference numbers (RTX 5090, N=8 / M=12 dets, warmup=20, iters=200)

Captured **2026-05-16** in `reports/thor_latency.json` with the matching `reports/thor_latency.png`.
These are dev-box reference numbers — the partner reports the analogous Thor numbers from running
the same `pixi run benchmark` on the iGPU.

**FP32 (all 6 shipped models)**

| Model             |     mean |   p50 |   p95 |   p99 |  std |   fps |
| ----------------- | -------: | ----: | ----: | ----: | ---: | ----: |
| `crlite_vit_exp3` |   3.0 ms |   2.9 |   3.8 |   4.5 |  0.5 | 331.4 |
| `crlite_vit_exp1` |   4.4 ms |   4.1 |   5.7 |   6.7 |  3.9 | 227.5 |
| `crlite_2dpe`     |   5.4 ms |   5.2 |   6.7 |   7.5 |  0.7 | 184.2 |
| `crlite`          |   5.5 ms |   5.4 |   6.5 |   7.5 |  0.6 | 183.1 |
| `crlite_vit_exp4` |   7.6 ms |   7.3 |   9.1 |  10.2 |  0.8 | 132.3 |
| `calibrefine`     | 547.5 ms | 534.9 | 601.0 | 718.5 | 44.7 |   1.8 |

**FP16 (PyTorch `model.half()`, all 6 shipped models)**

| Model             |     mean |   p50 |   p95 |   p99 |  std |   fps |
| ----------------- | -------: | ----: | ----: | ----: | ---: | ----: |
| `crlite_vit_exp3` |   2.8 ms |   2.7 |   3.7 |   4.0 |  0.4 | 351.8 |
| `crlite_vit_exp1` |   3.2 ms |   3.0 |   4.3 |   4.8 |  0.7 | 310.4 |
| `crlite`          |   5.5 ms |   5.5 |   6.6 |   7.5 |  0.6 | 180.5 |
| `crlite_2dpe`     |   5.5 ms |   5.3 |   6.8 |   7.8 |  0.6 | 182.1 |
| `crlite_vit_exp4` |   7.5 ms |   7.3 |   9.0 |  10.0 |  0.8 | 133.5 |
| `calibrefine`     | 532.3 ms | 528.6 | 575.5 | 595.9 | 24.8 |   1.9 |

**FP16 note.** All six models run in PyTorch eager-mode FP16 after `model.half()`. An earlier
revision tripped `Index put requires the source and destination dtypes match` for `crlite`,
`crlite_2dpe`, and `calibrefine` because three pre-allocated buffers (`final_sim` in
`CRLiteModel.forward_inference`, and `scores`

- `img/lid_centers` zero-pads in `PairwiseWrapper`, and the FPS distance buffer in
  `_pointnet2_ops.farthest_point_sample`) defaulted to `float32` while the model output was
  `float16`. Those allocations now honour the model's parameter dtype, so the FP16 column is fully
  populated. TensorRT FP16 engines on Thor are still the recommended deployment path — they do
  better mixed-precision scheduling than eager-mode `model.half()` — but the eager-mode FP16 numbers
  above are a useful sanity check on the iGPU before `trtexec` runs.

### 3.2 What to expect on Thor

Thor's Blackwell iGPU is roughly 0.4–0.6× the throughput of an RTX 5090 on these workloads, so
absolute FP32 latencies will be ~2× larger but the **~250× spread** between the lightweight models
and the pairwise baseline carries over. TensorRT FP16 engines on Thor are expected to give another
2–3× over PyTorch FP32 inside this directory, putting every lightweight model comfortably below the
100 ms (10 Hz) real-time budget. The pairwise baseline (`calibrefine`) is reported for reference
only — it is the slow ceiling that motivates the lightweight design and is **not** the recommended
deployment model.
