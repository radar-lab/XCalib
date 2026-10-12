# Continual refine

XCalib learns from a human in **two tiers**. Both start from the same signal — verified
camera↔LiDAR pairs — but differ in how long the result lives and whether a held-out check
stands between the update and the model.

| Tier | Entry point | State | Held-out check | Use it to |
| --- | --- | --- | --- | --- |
| **One-shot** | `matcher.oneshot(intr).bank_pairs(...)` → `.adapt()` | in memory, ephemeral | no | nudge the model *right now*, this session |
| **Continual** | `record_verified_pairs(...)` → corpus → `run_continual_refine(...)` | durable sqlite, accumulates | **yes** | keep improving over many sessions, safely |

The continual tier is the one this page covers. Verified pairs accumulate in a **corrections
corpus**. A bounded refine trains a candidate from that corpus, compares it with the serving
model on held-out samples, and publishes a new **revision** only when the candidate passes.
Otherwise the serving model stays as it is.

!!! tip "See it end-to-end"
    [`demo/refine_unification.ipynb`](https://github.com/radar-lab/XCalib/blob/main/demo/refine_unification.ipynb)
    runs this flow on the committed A9 sample: build a scene, match, record, inspect the
    corpus, refine with a gate relaxed for the demo, and load the published revision.

!!! note "Stability"
    `xcalib.continual` and `xcalib.gui` are not part of the curated [API reference](api.md).
    Their names and signatures can change between minor versions.

## Scene and model directory

The corpus, the frame loader, and the review window read the SUSTechPOINTS on-disk layout:

```text
<data_root>/<scene>/
    camera/<camera>/<frame>.jpg|png            images
    lidar/<frame>.pcd|bin|npy                  point clouds
    label/<frame>.json                         3D boxes: {"objs": [{"obj_id", "obj_type", "psr"}]}
    label_fusion/camera/<camera>/<frame>.json  2D rects: {"objs": [{"rect": {"x1", "y1", "x2", "y2"}, "obj_attr": "t<N>"}]}
    calib/xcalib/<camera>_model/               the model directory
```

The model directory holds the base model, the corpus, and everything refine publishes:

```text
<camera>_model/
    crlite.pth        base model, named <model>.pth where <model> is a model name
    crlite.yaml       its config (optional: without it XCalib uses the packaged A9 config)
    refine.sqlite     the corpus and the job state
    current.json      written by the first publish; points at the serving revision
    revisions/        one directory per published revision
```

Save a matcher there with `save_pretrained`, which writes both model files. Refine raises
`ContinualRefineUnavailable` when the directory holds no model.

```python
from xcalib import Matcher
from xcalib.continual import model_dir, open_store

matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")
matcher.save_pretrained(model_dir(data_root, scene, camera))

store = open_store(model_dir(data_root, scene, camera))
```

The corpus location is an explicit parameter, not an assumption: point `open_store` at the
model directory of whichever scene you are working on.

## 1. Record verified pairs

`record_verified_pairs` upserts a frame's human-verified decisions into the corpus. Each
`VerifiedPair` is one reviewed camera rect.

```python
from xcalib.continual import VerifiedPair, record_verified_pairs

# The rects must be the ones in the frame's label_fusion file; these values are made up.
pairs = [
    VerifiedPair(rect=[792.0, 660.0, 1263.6, 867.2], decision="positive", lidar_key="id:7",
                 track="3", candidates=[{"lidar_key": "id:7", "score": 0.91}],
                 proposal={"decision": "positive", "lidar_key": "id:7"}),
    VerifiedPair(rect=[0.4, 496.6, 179.5, 752.8], decision="no_match"),
]
n = record_verified_pairs(store, data_root, scene, camera, frame, pairs)
```

| Field | Meaning |
| --- | --- |
| `rect` | `[x1, y1, x2, y2]` in pixels, exactly as the frame's `label_fusion` file has it. The observation key rounds it to four decimals. A rect that matches no label rect is recorded but can never be trained on, and refine then reports `work_remaining`. |
| `decision` | `"positive"` accepts `lidar_key`. `"no_match"` records an explicit rejection and drops any key you pass. A positive without a `lidar_key` raises `ValueError`. |
| `lidar_key` | `"id:<obj_id>"` for a box that has an id, `"psr:<sha256>"` for a box without one. |
| `track` | The camera track as digits (the `N` of the rect's `t<N>` label); anything else is stored as no track. The track decides which split the sample lands in (see [Refine](#2-refine)). |
| `candidates` | The ranked menu the reviewer saw. The accepted target is appended when the menu lacks it. |
| `proposal` | The model's original pick. It sets `provenance` and tells training which box the human replaced, so pass it when you have it. |

- The write is **idempotent**: `sample_key` is derived from the observation, so re-recording
  the same rect upserts the same row. `label_revision` goes up each time the decision changes.
- `provenance` is set for you: `verified_model_correct` when the recorded pair equals the
  `proposal`, else `human_corrected`. It is stored for audit.
- Each pair is stamped with a fingerprint of the frame as it is on disk now: the image and
  point-cloud file timestamps and sizes, plus the box geometry. Refine recomputes it and skips
  a frame whose fingerprint changed. Editing boxes after recording, or copying the scene
  without keeping file timestamps, does this. Record the frame again to refresh it.

`store.iter_corrections()` returns one page of the active corrections: at most 1000 rows per
call, ordered by `sample_key`. Pass `after_sample_key=` to continue.
`get_refine_workload(model_dir(data_root, scene, camera)).counts` has the totals.

## 2. Refine

```python
from xcalib.continual import run_continual_refine

result = run_continual_refine(data_root, scene, camera, run_id="nightly-42",
                              progress=lambda p: print(p.phase, p.message))
print(result.outcome, result.new_revision)
```

Use a new `run_id` for every call. It names the revision (`continual-<20 hex digits>`), and
reusing one that already published fails, first with `PublicationConflict` and on later calls
with `ContinualRefineUnavailable`.

With no `frame_loader` injected, refine reads each reviewed frame through the package's own
`scene_frame_loader` (built on `xcalib.gui.scene`). A call goes through these phases, the
values of `RefineProgress.phase`:

1. **indexing** the corpus,
2. **caching** frozen-base embeddings, one frame at a time. A frame that cannot be read, or
   whose fingerprint changed, is skipped and marked; the call fails only if no frame could be
   cached,
3. **classifying** each sample as easy or hard for the serving model,
4. **training** a warm-started candidate on the hard samples,
5. **validating** the candidate against the serving model on the validation split,
6. **auditing** the candidate on the audit split. The audit is reported, it does not gate,
7. **publishing** a new revision only if the gate accepts.

Samples are grouped by camera track and split by a stable hash into train (80%), validation
(10%) and audit (10%). A sample without a track is grouped by blocks of 100 frames.

| `outcome` | Meaning |
| --- | --- |
| `accepted` | The candidate passed the gate and is published. `result.new_revision` names it. |
| `rejected` | The candidate failed the gate, or validation was too small to judge. `result.gate_reasons` says why. The serving model is unchanged. |
| `work_remaining` | Frames still need caching (for example a skipped frame), or the remaining hard samples cannot be trained on. `result.remaining` has the counts; call again after fixing the frames. |
| `up_to_date` | The serving model already handles every training sample. |

`RefineConfig` sets the thresholds:

| Field | Default | Meaning |
| --- | --- | --- |
| `min_validation_samples` | 20 | Validation samples needed before a candidate is trained. |
| `min_validation_groups` | 2 | Distinct tracks among them. |
| `minimum_primary_delta` | 0.005 | Primary-score gain required over the serving model. |
| `max_top1_drop` | 0.01 | Allowed drop in positive top-1. |
| `max_hard_violation_increase` | 0.01 | Allowed rise in hard-negative violations. |
| `max_no_match_fpr_increase` | 0.01 | Allowed rise in no-match false positives. |
| `embed_limit` | 500 | Frames cached per call. |
| `hard_limit`, `mastered_probe_limit`, `mastered_anchor_limit` | 4096, 512, 256 | Page sizes for the samples scored and trained per call. |

Metrics that are not finite fail the gate.

!!! warning "The gate is the point"
    `run_continual_refine` will not publish a revision that fails the gate. On a small corpus
    expect `rejected` with `insufficient grouped validation: <n> sample(s), <g> group(s)`: the
    default needs 20 validation samples from 2 tracks, and the validation split fills as you
    record pairs for more tracks. Relax the thresholds only for demos, as the notebook does.

Each call is finite. It caches at most `embed_limit` frames and trains for at most 20 epochs,
2000 steps, or 30 minutes, and early stopping (patience 3) restores the best epoch. Call it
again to consume more of the corpus. It runs on the default torch device, a GPU when one is
available.

On Windows, publishing writes `<model_dir>\.continual-<20 hex>.tmp\<model>.pth.tmp-<32 hex>`.
Keep the model directory short enough that this path stays under 260 characters, or enable
long paths; otherwise the publish step fails with `FileNotFoundError`.

### Cancel a run

`run_continual_refine` calls `progress` at every phase boundary, after each cached frame, and
after each training epoch. Raise from the callback to stop. The exception propagates out of
the call, and no revision is published or half-written as long as you do not raise on the
final `publishing` tick.

```python
import threading

stop = threading.Event()  # call stop.set() from another thread


class Stopped(Exception):
    pass


def progress(p):
    if stop.is_set() and not (p.phase == "publishing" and p.current == p.total):
        raise Stopped
    print(p.phase, p.message)


try:
    run_continual_refine(data_root, scene, camera, "nightly-43", progress)
except Stopped:
    print("cancelled")
```

### Background jobs

A host that runs refine as a background job tracks it in `refine.sqlite` with
`start_refine_job`, `get_refine_job`, `get_refine_workload`, `cancel_refine_job` and
`recover_interrupted_runs`.

- `cancel_refine_job(model_dir, run_id)` stops a run cooperatively. It returns `True` when this
  call stopped the run, `False` when the run had already finished, raises `KeyError` for an
  unknown run, and raises `CancelUnavailable` while the run is publishing. The run ends as
  `interrupted` with `failure_detail == CANCELLED_DETAIL` (`"cancelled by user"`). The worker
  stops at its next progress tick, where its progress call raises `LeaseLost`. Nothing is
  killed, so no publication is torn.
- `get_refine_workload(model_dir)` returns the corpus counts (`frames`, `corrections`,
  `positives`, `no_matches`, and `new`, `hard`, `mastered` once a run has classified the
  corpus) with the active and latest job.
- `recover_interrupted_runs(model_dir, scene=..., camera=...)` marks runs whose worker stopped
  sending heartbeats (30 s lease) as `interrupted` and finishes any publication left half done.
- `start_refine_job(model_root, scene, camera, python, repo_root, worker=None)` starts a worker
  script in a separate process and tracks the run as a job. XCalib does not ship a worker, so
  write one. It receives `--data-root`, `--scene`, `--camera`, `--run-id` and `--lease-token`,
  opens `ContinualJobs.for_scene(data_root, scene, camera)`, and calls `run_continual_refine`
  while it holds the lease: `lease_heartbeat` keeps the lease alive, `record_progress` takes
  the progress callbacks, and `finish` or `fail` ends the run. Pass the script as `worker`; a
  relative path resolves against `repo_root`, and `None` means `tools/xcalib/refine_continual.py`
  under `repo_root`. Without a worker, call `run_continual_refine` directly; a direct call is
  not tracked as a job.

## What a published revision contains

```text
<camera>_model/
    current.json                         the pointer
    revisions/continual-<20 hex>/
        crlite.pth                       weights and metadata, see below
        crlite.yaml                      matcher config
```

`current.json` holds `schema_version`, `revision`, `parent_revision`, `model_name`, `weights`,
`config`, `sha256` and `published_at`. The revision's `.pth` holds the base weights
(`model_state_dict`), the trained adapters (`adapters_state_dict`), the acceptance head
(`continual_state_dict`: `acceptance.log_scale` and `acceptance.bias`, the threshold refine
learned), and `continual_meta` (`revision`, `parent_revision`, `base_fingerprint`,
`acceptance_source`, `train_config`, `metrics`). Refine never modifies the base `crlite.pth`
(the review window's one-shot **Save** does), and each revision records its parent, so
revisions form a chain back to the base.

XCalib saves the candidate to a temporary directory, reloads it, checks the hash, the tensors
and that scores are finite, and only then switches `current.json`. Every step is journalled in
`refine.sqlite`. `rollback_revision` points `current.json` back at an earlier revision under
`revisions/`.

Load the serving revision with `resolve_model`:

```python
from xcalib import Matcher
from xcalib.continual import resolve_model

ref = resolve_model(data_root, scene, camera)
matcher = Matcher.from_pretrained(ref.revision.model_name, weights=ref.revision.weights_path,
                                  config=ref.revision.config_path)
```

`Matcher` loads the adapters. It does not read the acceptance head. The next refine run
starts from the serving revision. The `xcalib-review` window does not follow `current.json`:
it keeps loading the `.pth` in the model directory.

## In the review GUI

```bash
pip install "xcalib[gui]"
xcalib-review --scene /data/MySite --camera cam_front
```

`--scene` is the scene directory (`<data_root>/<scene>`) and it needs a model directory as
described above. Step frames with ‹ ›, click a card and pick the
right #id or *— none —*, then **Apply this frame** or **Apply all** to write the decisions to
the scene labels. Two buttons then learn from them:

- **Refine on corrections** is the one-shot tier. It needs at least 8 verified pairs, shows the
  before and after retrieval accuracy, and asks whether to save the adapted weights over the
  scene's model.
- **Refine (accumulate)** is the continual tier. It records the applied decisions (accepted
  ids and explicit rejections) of up to 60 corrected frames into the corpus, then runs
  `run_continual_refine` with the default `RefineConfig` and shows the outcome. On a small scene
  the default gate usually reports `rejected`. The window has no cancel button.
