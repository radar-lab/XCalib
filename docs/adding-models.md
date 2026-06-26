# Adding a model

`xcalib` is a single package that hosts a growing family of calibration /
matching models (like model names under the `ultralytics` brand): the public
API stays `Matcher.from_pretrained("<name>", site=...)`, and new models are
new registry names. This page is the checklist for landing one.

## 1. Model class

Implement the architecture under `src/xcalib/models/` as a subclass of
`EdgeModelBase` (`src/xcalib/models/_base.py`). The class consumes an
`EdgeConfig` and must expose the standard forward contract used by the
inference wrappers (`src/xcalib/engine/wrappers.py`) — embeddings or
similarity logits over `(image_crops, lidar_crops, img_centers, lid_centers)`.

If the model needs a new control flow (not two-stage, cosine, or pairwise),
add a wrapper in `engine/wrappers.py` and an export path in
`engine/exporter.py`.

## 2. Registry entry

Register the name in `src/xcalib/models/registry.py`:

```python
_MODEL_REGISTRY = {
    ...
    "mynewmodel": MyNewModel,
}
```

That name immediately works in `Matcher.from_pretrained`, `xcalib.train`,
the CLI (`--model mynewmodel`), and the scripts. If the model is trainable
with the standard HDF5 loop, also add it to `TRAINABLE` in
`src/xcalib/engine/trainer.py`.

## 3. Config YAML

Add one packaged config per site to `src/xcalib/cfg/`:

```
src/xcalib/cfg/mynewmodel_utc4.yaml
src/xcalib/cfg/mynewmodel_utc3.yaml
src/xcalib/cfg/mynewmodel_a9_dataset_r02_s01.yaml
```

`default_config_path(model, site)` resolves these from package data — no
checkout needed at runtime. Keep keys consistent with the existing YAMLs
(`crop_size`, `point_cloud_size`, `bbox_expansion`, `top_k`, ...).

## 4. Weights

Name checkpoints `checkpoints/<model>_<site>_best.pth` and keep the matching
YAML config with the model release. Public releases may distribute approved
weights through the same `Matcher.from_pretrained("<model>", site=...)`
convention used by existing models.

Artifact release and visibility steps are intentionally not documented in the
public repository.

## 5. Tests + version bump

- Extend `tests/smoke/test_smoke.py` (random-weights forward) so CI covers the
  model without checkpoints.
- If the model exports to ONNX, add it to the `matcher.build("onnx")`
  parity test in `tests/integration/test_build_api.py`.
- A new model is a feature: bump the **minor** version in
  `src/xcalib/__init__.py` (e.g. `0.1.0` → `0.2.0`) and update the public
  model table / loading docs.
