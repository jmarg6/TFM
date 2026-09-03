# Reproducibility notes

## Experimental protocol

All methods operate on a deterministic 80/20 split of the original training partition. The split seed is 42. Selection masks address only the base-training partition; validation data are used for candidate fitness, and held-out test data are accessed only after the subset is fixed.

Frozen ImageNet-pretrained representations are extracted once and cached. Every candidate subset trains only a fresh linear classifier for 10 epochs using Adam (`lr=1e-3`, weight decay `0`, batch size `32`). Within a guided search run, candidates share the same classifier initialization and training seed.

Search seeds:

```text
42, 123, 456, 789, 1024
```

Retention levels:

```text
10%, 25%, 50%, 75%, 90%
```

## Stage 1 — preliminary MNIST screening

- Models: AlexNet, ResNeXt-50 32×4d, EfficientNetV2-S, Swin-T
- Methods: RS, LNS, GA, MA
- Budget: B=100 candidate evaluations per search
- Matrix: 4 models × 4 methods × 5 retentions × 5 seeds = 400 searches
- Hardware: Hera (4× NVIDIA Titan RTX; each job used one GPU)

## Stage 2 — definitive main study

Datasets: CIFAR-10 and Tiny ImageNet. The selected frozen representation is AlexNet.

Guided searches:

- GA and RS
- B=1000 candidate evaluations
- 2 datasets × 2 methods × 5 retentions × 5 seeds = 100 searches

Additional baselines:

- Single Stratified Random Sample (SRS): one class-stratified subset, no validation-driven search
- Global k-center greedy (KCG): label-free farthest-first traversal in raw cached AlexNet feature space
- 100% full-data baseline

All definitive Stage-2 selection and final-evaluation jobs were executed on Zeus (4× NVIDIA GeForce RTX 2080 Ti; one GPU per job).

## Seeds

For search seed `r`:

```text
candidate-training seed = r + 100000
final-evaluation seed   = r + 200000
```

The final selected subset is retrained from the independent final initialization before held-out test evaluation.

## Determinism

The implementation seeds Python, NumPy, and PyTorch; enables deterministic PyTorch/cuDNN behavior; disables cuDNN benchmarking; and uses `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic CUDA matrix operations.

## Environment recorded in definitive manifests

```text
Python       3.12.13
PyTorch      2.5.1+cu121
Torchvision  0.20.1+cu121
NumPy        2.5.1
scikit-learn 1.9.0
```

## Large artifacts

The repository does not include raw datasets, frozen feature tensors, or KCG pairwise-distance caches. See `data/README.md` for the expected layout and regeneration procedure.
