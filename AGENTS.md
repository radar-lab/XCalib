# AGENTS.md

XCalib is an open-source camera/LiDAR matching and calibration package. This
file is the single entry point for coding agents and contributors.

## Highest-Value Rules

- Treat every tracked file as public. Never commit credentials, private paths,
  unpublished partner data, restricted weights, or internal release operations.
- Preserve the public `Matcher` API, model registry contract, input protocol,
  package-data layout, and offline-test behavior unless an intentional migration
  is documented and tested.
- New model families register through `src/xcalib/models/registry.py`; do not
  add ad hoc name dispatch in callers.
- Frozen evidence under `docs/evidence/` changes only with reproducible source
  output and a documented reason; regenerated local reports are not evidence by
  themselves.
- Hub-facing package code remains read-only. Publication and write-side
  artifact operations live outside this public repository.

## Situation Dispatch

| Situation | Open exactly this first |
| --- | --- |
| Editing code, tests, docs, models, or packaging | [Workflow](docs/agents/workflow.md) |
| Finding the inference, training, continual, GUI, or data seam | [Code Index](docs/agents/code-index.md) |
| Checking naming, versions, evidence, generated files, or compatibility | [Conventions](docs/agents/conventions.md) |

## Core Validation

- `pixi run check-public-docs`
- `pixi run lint`
- `pixi run smoke` for fast import and forward-pass coverage
- `pixi run test` for the complete offline suite
- `pixi run docs-build` after documentation or MkDocs changes
- `git diff --check` and exact status/diff review
