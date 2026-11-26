# CondSiteProto

Reproduction code, fixed splits, and experiment configurations (Tables 2–5) for **Condition-Aware Site Prototypes for Joint Multi-Condition Protein Variant Prediction**.

## Setup

Create the environment with `conda env create -f environment.yml` and activate it with `conda activate condsiteproto`. CPU training is supported. For GPU training, use a PyTorch 2.8.0 build compatible with your CUDA driver.

Download the official [ESM-C 600M v0 checkpoint](https://huggingface.co/biohub/esmc-600m-2024-12/blob/main/data/weights/esmc_600m_2024_12_v0.pth) (`esm==3.2.3`) to extract frozen wild-type residue embeddings into `--cache-dir`, which is reused automatically across runs.

## Run

From the repository root, run the following for one Table 2 configuration. Select another configuration under `configs/` for the corresponding experiment.

```sh
python scripts/prepare_data.py --weights esmc_600m_2024_12_v0.pth --cache-dir cache
python scripts/train.py --config configs/table2_main/kras_id_d128_proto.yaml --cache-dir cache --device cpu
python scripts/evaluate.py --runs-dir runs
```

Use `--dataset` to prepare selected caches, `--check-only` to check the tables, and `--check-parameters` on the training command to verify a configuration without embeddings. Training runs the configured seeds and five folds by default. Use `--seed` and `--fold` to run individual jobs, and `--allow-partial` when evaluating incomplete runs. For a one-epoch wiring check, use `--smoke` with an existing cache and evaluate with `--runs-dir smoke-runs --allow-smoke`. The cache and run directories are local outputs.

The evaluator computes the square root of the mean held-out MSE over folds and conditions for each seed, and averages Pearson correlations over the same cells. It reports their mean and sample standard deviation across seeds.

## Licensing and Data Sources

- **Code**: Licensed under [MIT](LICENSE); parts of the pipeline build on [Cheng, Wu, and Ming (2026)](https://doi.org/10.34133/csbj.0144).

- **Datasets**: Processed tables retain their upstream licenses (RamR and DAOx: CC BY 4.0; KRAS: CC0 1.0; NorA: MIT, Copyright (c) 2025 Raman lab; APH(3′)-II: [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/) for noncommercial reuse from [Melnikov et al., 2014](https://doi.org/10.1093/nar/gku511)). Full provenance and file checksums are listed in [`data/sources.csv`](data/sources.csv) and [`data/processed/SHA256SUMS.txt`](data/processed/SHA256SUMS.txt).
