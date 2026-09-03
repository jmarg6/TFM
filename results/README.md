# Archived experiment results

This directory contains the experiment artifacts used by `notebooks/final_results_analysis.ipynb` and by the thesis results section.

```text
results/
├── preliminary/mnist/
│   └── prelim_features_mnist_*        # 400 B=100 screening searches
├── main/
│   ├── guided_b1000/
│   │   └── main_b1000_*               # 100 definitive GA/RS searches
│   ├── srs/
│   │   └── main_srs_*                 # single stratified random subsets
│   ├── kcg/
│   │   └── main_global_kcg_*          # global k-center greedy runs
│   └── full_data/
│       └── main_full100_zeus_*        # 100% base-training reference
└── diagnostics/guided_b100/
    └── prelim_features_{cifar10,tinyimagenet}_*
                                        # earlier B=100 main-dataset runs
```

The diagnostic B=100 CIFAR-10/Tiny ImageNet runs are retained for the budget comparison only. They are not mixed with the definitive B=1000 main-stage results.

Individual experiment directories contain top-level summaries (`manifest.json`, `search_summary.csv`, `test_summary.csv`) and detailed per-run artifacts under `runs/`, including selected masks, configurations, fitness histories, and final metrics.
