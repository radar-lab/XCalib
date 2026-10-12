# XCalib Contributor Workflow

Open this before changing package code, models, tests, documentation, evidence,
or distribution metadata.

- [Mission and public boundary](#mission-and-public-boundary)
- [Change flow](#change-flow)
- [Edit policy](#edit-policy)
- [Verification duties](#verification-duties)
- [Research evidence](#research-evidence)
- [Commits and reviews](#commits-and-reviews)

## Mission and public boundary

XCalib provides camera/LiDAR matching, targetless calibration, continual
refinement, training, ONNX/TensorRT export, and a review GUI. Start from current
code, tests, package metadata, and reproducible evidence.

- Every tracked file and built documentation page is public.
- Never add credentials, private machine paths, restricted data identifiers,
  unpublished weights, or write-side artifact operations.
- Keep tests offline. Hub and network behavior is mocked unless a separately
  declared integration check explicitly requires access.

## Change flow

1. Inspect the owning module, public contract, configuration, and focused tests.
2. Work on a short-lived branch for one independently reviewable concern.
3. Add a regression test that fails without the intended behavior.
4. Run focused tests, then the required lint, public-boundary, and suite checks.
5. Update the owning guide when a public flag, config key, model, protocol, or
   workflow changes.
6. Submit through review; do not push directly to a protected release branch.

## Edit policy

| Path | Responsibility | Rule |
| --- | --- | --- |
| `src/xcalib/engine/` | Matching, export, training, runtime wrappers | Preserve backend parity and public builder behavior. |
| `src/xcalib/models/` | Model implementations and registry | Register models centrally and keep config names stable. |
| `src/xcalib/protocol.py` | Input and record schema | Treat as a public compatibility surface. |
| `src/xcalib/continual/` | Revision, adaptation, publication records | Preserve atomic state and provenance semantics. |
| `src/xcalib/hub/` | Public read-side model and dataset retrieval | Do not add repository creation or artifact-write APIs. |
| `src/xcalib/gui/` | Human review workflow | Preserve non-GUI imports and keyboard-accessible review behavior. |
| `scripts/paper/` | Reproduction, export, training, and benchmark scripts | Keep outputs explicit and avoid silently overwriting evidence. |
| `docs/evidence/` | Frozen shareable evidence | Update only from a reproducible run with provenance. |

## Verification duties

- `pixi run check-public-docs` for public-boundary and agent-guide checks.
- `pixi run lint` after Python changes.
- `pixi run smoke` for quick import and forward-pass coverage.
- `pixi run test` before review unless the change is documentation-only.
- Focused ONNX, training, GUI, or continual tests for the affected subsystem.
- `pixi run docs-build` after documentation or MkDocs changes.
- `git diff --check`; report every failed or skipped check.

## Research evidence

Do not treat old report values as current measurements. A claimed metric names
the model, site or dataset, precision/backend, hardware when relevant, and the
artifact that produced it. Preserve frozen evidence unless an intentional
rerun supplies replacement provenance.

## Commits and reviews

Use conventional commit types and one complete verified concern per commit.
Review descriptions state the public behavior change, compatibility impact,
tests and observed results, documentation/evidence changes, and verification
that could not be performed.
