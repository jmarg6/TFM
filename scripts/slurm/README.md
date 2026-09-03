# Slurm launch scripts

These scripts are the launch files used on the DaSCI GPU infrastructure. They are preserved with their original experiment configuration and absolute cluster paths.

Definitive scripts:

| Script | Purpose | Node |
| --- | --- | --- |
| `extract_features_slurm.sh` | Frozen feature extraction | configurable job script |
| `run_preliminary_features_slurm.sh` | MNIST screening, B=100 | Hera |
| `run_main_b1000_features_slurm.sh` | Definitive GA/RS main stage, B=1000 | Zeus |
| `run_stratified_random_features_slurm.sh` | SRS baseline | Zeus |
| `run_global_k_center_greedy_features_slurm.sh` | Global KCG baseline | Zeus |
| `run_full100_zeus_slurm.sh` | 100% full-data baseline | Zeus |

The `legacy/` directory contains earlier launch scripts kept for traceability. They are not required to reproduce the definitive experiment matrix.

Before using these scripts on another cluster, update `PROJECT_DIR`, the Conda environment path, Slurm partition/node constraints, memory limits, and wall-time limits as appropriate.
