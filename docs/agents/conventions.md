# XCalib Conventions

Open this when naming models or configuration, changing versions, handling
evidence or generated output, or reviewing compatibility.

- [Public boundary](#public-boundary)
- [Models and configuration](#models-and-configuration)
- [Versions and packaging](#versions-and-packaging)
- [Evidence and generated output](#evidence-and-generated-output)
- [Compatibility](#compatibility)

## Public boundary

All tracked source, tests, entry files, and documentation are public. Keep
artifact publication, restricted datasets, partner identifiers, credentials,
private paths, and internal operations outside this repository.

## Models and configuration

- Model implementation modules use stable lowercase identifiers registered in
  `src/xcalib/models/registry.py`.
- Packaged configuration files use `<model>_<site>.yaml` and ship under
  `src/xcalib/cfg/`.
- Site-specific exports use separate output directories so one site's weights
  cannot overwrite another's.
- Public configs and docs may name only data and artifacts that are actually
  available to their intended reader.

## Versions and packaging

The canonical version is `src/xcalib/__init__.py`; `pyproject.toml` reads it
dynamically. Use `major.minor.revision` and update frozen release metadata or
documentation in the same change when required. Wheels are pure Python and
package `src/xcalib`; checkpoints and large datasets are retrieved separately.

## Evidence and generated output

- `docs/evidence/` contains frozen shareable JSON evidence and its index.
- `reports/`, `runs/`, `onnx/`, `engines/`, `dist/`, and `site/` are generated
  or local outputs unless an owning workflow explicitly promotes an artifact.
- Never replace frozen evidence with a hand-edited number.
- Inventory ignored checkpoints, caches, and reports before deletion.

## Compatibility

- Python support and optional dependencies are declared in `pyproject.toml`.
- Keep public imports, protocol fields, config names, export formats, and CLI
  behavior backward compatible unless a documented migration justifies a
  breaking change.
- Tests requiring local checkpoints skip in public CI rather than weakening
  assertions or silently downloading restricted data.
