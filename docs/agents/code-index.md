# XCalib Code Index

Open this when locating a public API, model, runtime backend, data flow,
continual-learning component, GUI surface, or validation command.

- [Package map](#package-map)
- [Inference and calibration flow](#inference-and-calibration-flow)
- [Training and continual flow](#training-and-continual-flow)
- [Distribution and documentation](#distribution-and-documentation)

## Package map

| Path | Responsibility |
| --- | --- |
| `src/xcalib/__init__.py` | Public imports and package version. |
| `src/xcalib/engine/matcher.py` | Main matcher construction and inference facade. |
| `src/xcalib/engine/exporter.py` | ONNX export and parity support. |
| `src/xcalib/engine/trainer.py` | Shared training behavior. |
| `src/xcalib/engine/trt.py` | TensorRT runtime integration. |
| `src/xcalib/models/registry.py` | Canonical model registration and construction. |
| `src/xcalib/cfg/` | Packaged model/site configuration. |
| `src/xcalib/protocol.py` | Public input and record protocol. |
| `src/xcalib/calibration_refine.py` | Calibration refinement and candidate selection. |
| `src/xcalib/oneshot/` | One-shot adaptation sessions, memory, and pseudo-labels. |
| `src/xcalib/continual/` | Continual jobs, revisions, training, stores, and provenance. |
| `src/xcalib/data/` | Crop and HDF5 cache loading. |
| `src/xcalib/hub/` | Public read-side weights and dataset retrieval. |
| `src/xcalib/gui/` | Review application and scene model. |
| `tests/smoke/` | Fast import and forward-pass checks. |
| `tests/unit/` | Component and contract tests. |
| `tests/integration/` | Multi-component build, training, and one-shot checks. |

## Inference and calibration flow

The CLI and public imports construct a matcher through the engine and model
registry. Packaged configuration selects the model/site behavior. Assignment,
calibration refinement, visualization, and optional runtime wrappers consume
the protocol-defined inputs without introducing backend-specific public APIs.

## Training and continual flow

HDF5 loaders feed shared training code and paper scripts. One-shot adaptation
uses session-local memory and pseudo-labels. Continual modules record revisions,
jobs, data provenance, training state, and publication-ready records while
preserving atomic state transitions.

## Distribution and documentation

`pyproject.toml` owns package metadata, dependencies, Pixi tasks, build rules,
and the dynamic version path. `mkdocs.yml` owns the public documentation portal.
`scripts/check_public_docs.py` guards tracked files against personal paths,
credentials and Hub write-side operations.
