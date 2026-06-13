# vllm-cloud

This repository is the downstream workspace for vLLM Ascend based cloud
adaptation work.

The intended upstream chain is:

1. `vllm-project/vllm`
2. `vllm-project/vllm-ascend`
3. this repository

The repository is intentionally initialized with project hygiene, CI, lint, and
formatting infrastructure before importing or adapting upstream code. The Python
lint baseline is kept compatible with vLLM Ascend to reduce churn when upstream
source is imported.

## Quick Start

Use a Python 3.11 environment for local validation. The local environment can be
managed by conda or another tool; environment creation is intentionally not
codified in this repository.

```bash
python -m pip install -r requirements-lint.txt
```

Run the local checks:

```bash
bash scripts/lint.sh
python -m pytest
```

Format files:

```bash
bash scripts/format.sh
```

The project records lint and test dependencies, but not a conda environment
definition.

## Full Installation

Two installation scripts are provided:

### A3 / DeepSeek V4 Pro

Uses the official vllm-ascend repo:

```bash
bash install_all.sh <vllm_version> <vllm_ascend_ref>
```

Example:

```bash
bash install_all.sh 0.20.2 0.20.2rc1
```

### A5 / Qwen3.5

Uses a custom vllm-ascend fork with NPU-specific tweaks (no triton uninstall,
`--no-build-isolation --no-deps`):

```bash
bash install_all_a5.sh <vllm_version> <vllm_ascend_commit>
```

Example:

```bash
# Uses TallMessiWu fork (Qwen3.5 baseline)
VLLM_ASCEND_REPO=https://github.com/TallMessiWu/vllm-ascend.git \
  bash install_all_a5.sh 0.20.2 1ba24186d0e422d1b7fdf76bbd2c6a234e6e166f
```

Set `VLLM_ASCEND_REPO` to override the vllm-ascend git URL (defaults to
`https://github.com/vllm-project/vllm-ascend.git`).

See [`docs/version-matrix.md`](docs/version-matrix.md) for the current baseline.

## Upstream Policy

Before importing upstream source, choose and record the exact baseline in
[`docs/version-matrix.md`](docs/version-matrix.md). The sync process is
documented in [`docs/upstream-sync.md`](docs/upstream-sync.md).
