# Models and Datasets

`xcalib` can load released paper checkpoints and the public A9 dataset cache
from the Hugging Face Hub. Local files also work, which is useful for offline
deployment, custom checkpoints, or experiments.

## Load A Pretrained Matcher

The usual path is `Matcher.from_pretrained(model, site=...)`. It resolves the
packaged config and downloads the matching checkpoint when the release artifact
is available:

```python
from xcalib import Matcher

matcher = Matcher.from_pretrained("crlite", site="a9_dataset_r02_s01")
```

Pin a release tag for reproducible loads:

```python
matcher = Matcher.from_pretrained(
    "crlite",
    site="a9_dataset_r02_s01",
    revision="v0.1.0",
)
```

## Download Weights Ahead Of Time

For machines that should not download at runtime:

```bash
xcalib pull-weights --model crlite --site a9_dataset_r02_s01 --out checkpoints/
```

Then load the local files:

```python
matcher = Matcher.from_pretrained(
    "crlite",
    weights="checkpoints/crlite_a9_dataset_r02_s01_best.pth",
    config="checkpoints/crlite_a9_dataset_r02_s01.yaml",
)
```

## Load The A9 Dataset Cache

`load_dataset()` first checks for a local cache and then falls back to released
Hub artifacts:

```python
from xcalib import load_dataset

loader = load_dataset("a9_dataset_r02_s01", split="test")
```

You can also pre-fetch a split:

```bash
xcalib pull-dataset --site a9_dataset_r02_s01 --split test --out datasets/
```

## Local Custom Weights

Custom or fine-tuned weights should be loaded from local paths with the matching
YAML config:

```python
matcher = Matcher.from_pretrained(
    "crlite",
    weights="runs/site42/best.pth",
    config="runs/site42/crlite_site42.yaml",
)
```

If the config changes model dimensions, the checkpoint must have been trained
with that same config.

## Integrity

For released artifacts, compare downloaded files against the checksums listed
in the model or dataset card.

PowerShell:

```powershell
Get-FileHash .\checkpoints\crlite_a9_dataset_r02_s01_best.pth -Algorithm SHA256
```

Linux/macOS:

```bash
sha256sum checkpoints/crlite_a9_dataset_r02_s01_best.pth
```

## Dataset Terms

The A9 HDF5 caches derive from the TUM Traffic / A9 dataset. Users remain
responsible for following upstream dataset terms and citations.

