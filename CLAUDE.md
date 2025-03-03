# Smallrunner - Claude Helper File

## Build & Installation
```bash
pip install git+ssh://git@github.com/brentyi/smallrunner.git
```

## Commands
- Run smallrunner: `smallrunner example/example.sh`
- Run with specific GPUs: `smallrunner example/example.sh --gpu_ids 0 1 2`
- Run concurrent jobs: `smallrunner example/example.sh --concurrent_jobs 2`
- Allocate multiple GPUs per job: `smallrunner example/example.sh --gpus_per_job 2`
- Use GPU topology for allocation: `smallrunner example/example.sh --gpus_per_job 2 --use_topology`
- Filter jobs: `smallrunner example/example.sh --job_index_cond "i<5"`
- Set memory threshold: `smallrunner example/example.sh --mem_ratio_threshold 0.2`

## GPU Allocation Features
- `--concurrent_jobs N`: Run N jobs in parallel per primary GPU
- `--gpus_per_job N`: Allocate N GPUs to each job
- `--use_topology`: Use NVML topology information to assign GPUs with optimal connections

## Code Style
- **Formatting**: 4-space indentation, PEP 8 compatible
- **Types**: Type annotations required for all functions
- **Naming**: Classes use CamelCase, functions/variables use snake_case
- **Docstrings**: Google style docstrings for classes and functions
- **Error Handling**: Assert statements for validation
- **Imports**: Standard library imports first, then third-party, then local

## Linting
- Uses ruff with E731 ignored: `ruff check .`