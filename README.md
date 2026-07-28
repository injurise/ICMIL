# ICMIL, In-Context Multiple-Instance Learning

ICMIL solves multiple-instance learning in context. You give it a handful of labelled
bags and it predicts the labels of unseen bags in one forward pass. No per-dataset
training, no fine-tuning, no hyper-parameter search.

This repo reproduces the paper's benchmark table. It also provides training and data generation code. 

## Setup

```bash
pip install -e .                 # reproduction only
pip install -e ".[datagen]"      # also the prior generator
```

## Reproducing the table

```bash
python -m icmil.reproduce                                    # the full table
python -m icmil.reproduce --tasks uci_musk1 --baselines mean_logreg   # seconds
```

Rows are models, columns are tasks, cells are AUROC as mean ± SEM. The ± is the spread
across runs, so across three trained seeds for ICMIL and across `--n-seeds` random seeds
for a baseline. Within one run the score is the mean over the task's frozen splits.

Each run writes `benchmark_table.md`, `results.json` at full precision, and
`run_meta.json` with versions, device, TF32 flags, checksums and timings. Keep the last
one. It is what makes a reported number checkable later.

Run `--help` for all options.

### Baselines

| Name | Method |
|---|---|
| `mean_logreg` | Mean-pool the bag, then cross-validated logistic regression |
| `svm_summ` | Per-bag summary statistics, then an RBF SVM |
| `abmil_refit` | Attention-based MIL (Ilse et al., 2018), 5-fold CV then refit |
| `acmil` | Attention-Challenging MIL (Zhang et al., 2023), same protocol |
| `tabpfn_concat` | Flatten the bag into one row for a frozen TabPFN v2 |
| `tabpfn_subsample` | Ten random instance subsets per bag, logits averaged |
| `cluster_tabpfn` | K-means selective pooling, then TabPFN per cluster |

### Tasks

Twelve benchmarks covering synthetic rules, natural MIL and medical imaging. They are
`tcga_fixed`, `rsna_ich_draws`, `mnist_xai_{smil,pos_neg,adjacent_pairs}`,
`uci_{musk1,musk2,letters,hepmass}` and `andrews_{fox,tiger,elephant}`.

The `.h5` files in `datasets/` carry frozen splits.

## Using the model

```python
from icmil import load_icmil

model = load_icmil(seed="c5trd795", device="cuda")

# X_train is (1, n_bags, bag_size, n_features), y_train is (1, n_bags).
logits = model(X_train, y_train, X_test)      # (1, n_query_bags, n_classes)
```

## Generating the prior

```bash
python -m icmil.datagen.generate --dry-run                  # inspect the recipe
python -m icmil.datagen.generate --num-batches 100 --out-dir /tmp/demo
```

## Training

```bash
python -m icmil.train --data-dir workdir/priors --out checkpoints/my-icmil.pt
```

## Licence

MIT, see `LICENSE`. `THIRD_PARTY.md` lists the methods and datasets this builds on.
