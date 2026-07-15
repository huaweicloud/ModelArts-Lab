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

Three installation scripts are provided:

### A3 / DeepSeek V4 Pro

Uses the official vllm-ascend repo:

```bash
bash install_all.sh <vllm_version> <vllm_ascend_ref>
```

Example:

```bash
bash install_all.sh 0.20.2 0.20.2rc1
```

### A5 / Qwen3.5 AND DeepSeek V4

```bash
bash install_all_a5.sh <vllm_version> <vllm_ascend_commit/vllm_ascend_ref> [compile_mooncake=TRUE]
```

- `compile_mooncake=TRUE`: Install mooncake with source code; otherwise install mooncake with wheel package

Example:

```bash
# Uses vllm-ascend releases/v0.23.0 branch with specified commit id since RC version is not ready, install mooncake with source code
  bash install_all_a5.sh 0.23.0 cabfbf8906d78d083a51e1e552fea751a937a880 compile_mooncake=TRUE
```

```bash
# Uses vllm-ascend releases/v0.23.0 branch with specified commit id since RC version is not ready, install mooncake with wheel package
  bash install_all_a5.sh 0.23.0 cabfbf8906d78d083a51e1e552fea751a937a880
```

Set `VLLM_ASCEND_REPO` to override the vllm-ascend git URL (defaults to
`https://github.com/vllm-project/vllm-ascend.git`).


See [`docs/version-matrix.md`](docs/version-matrix.md) for the current baseline.

## Upstream Policy

Before importing upstream source, choose and record the exact baseline in
[`docs/version-matrix.md`](docs/version-matrix.md). The sync process is
documented in [`docs/upstream-sync.md`](docs/upstream-sync.md).
