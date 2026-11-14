# In-Context Multiple-Instance Learning

Official code for the paper "In-Context Multiple Instance Learning".

ICMIL is a Prior-data Fitted Network (PFN) for Multiple Instance Learning. A Perceiver-style transformer is pretrained on synthetic bag-structured data and, at inference time, classifies new MIL tasks in a single forward pass. No gradient updates, no hyperparameter tuning, no task-specific finetuning.

## Highlights

- **In-context MIL:** solve a new bag classification task by feeding labeled context bags directly to the model.
- **Perceiver-style architecture** for hierarchical set inputs that handles scalability, task-dependent instance compression, and within-bag permutation invariance.
- **Synthetic priors** for bag-structured data (factorized and joint), with a mixture prior that combines their complementary inductive biases.


## Usage

#### Setup

```bash
pip install -e .
pip install -e ".[datagen]"
```

#### Forward Pass

```python
from icmil import load_icmil

model = load_icmil(seed="c5trd795", device="cuda")

# X_train is (1, n_bags, bag_size, n_features), y_train is (1, n_bags).
logits = model(X_train, y_train, X_test)  # (1, n_query_bags, n_classes)
```

#### Prior Generation

```bash
python -m icmil.datagen.generate --dry-run
python -m icmil.datagen.generate --num-batches 100 --out-dir /tmp/demo
```

#### Training

```bash
python -m icmil.train --data-dir workdir/priors --out checkpoints/my-icmil.pt
```

#### Reproducing the results table from the paper

```bash
python -m icmil.reproduce
python -m icmil.reproduce --tasks uci_musk1 --baselines mean_logreg
```

To also train ABMIL and ACMIL on the validation bags after model selection (refit), run `python -m icmil.reproduce --baselines abmil_refit,acmil_refit`.

TAD is not in the paper table and not part of the default run:

```bash
python -m icmil.reproduce --tasks tad_pca25
python -m icmil.reproduce --tasks tad_pca50 --icmil-seeds zeqy97nm,awaqfzxd,ldsebntm  # 50-feature checkpoints
```

### Single-cell benchmarks

Patient classification from scRNA-seq on two datasets of Do & Lähdesmäki (2025): COVID (50
patients, 35 COVID-19 vs 15 controls) and ICB (57 patients, 19 responders vs 38 non-responders to
immune checkpoint blockade). They are not in the paper table.

#### Constructing the datasets

The data are not redistributed here. Build them once:

```bash
pip install -e ".[scrna]"
python -m icmil.prepare_scrna                 # or: --raw-dir DIR with covid.h5ad / icb.h5ad already downloaded
```

This

1. downloads `covid.h5ad` and `icb.h5ad`, the preprocessed files shared by the authors
   (Google Drive folder linked from https://github.com/minhchaudo/hier-mil), to `datasets/raw/`
   and checks their sha256;
2. keeps per cell the features the authors use: 512-d scGPT embeddings (COVID) or 824
   log-normalised genes (ICB);
3. samples a fixed number of cells per patient, 1000 (COVID) or 242 (ICB), with replacement only
   for patients with fewer cells;
4. freezes the paper's folds: patients in file order, `StratifiedKFold(10, shuffle=True,
   random_state=r)` for repetitions `r = 0..9`;
5. writes `datasets/scrna_covid.h5` and `datasets/scrna_icb.h5` (`bags`, `labels`, `folds`).

PCA to 25 or 50 features is fitted per fold on the training cells at evaluation time.

#### Evaluation

Each task has 100 splits (10 repetitions × 10 folds); the reported AUC is computed per repetition
on its pooled test predictions, as in the paper.

```bash
python -m icmil.reproduce --tasks scrna_covid_pca25,scrna_icb_pca25
python -m icmil.reproduce --tasks scrna_covid_pca50,scrna_icb_pca50 --icmil-seeds zeqy97nm,awaqfzxd,ldsebntm
```

The supervised baselines are slow on 100 splits; pick a few with `--baselines`.

## Licence

MIT, see `LICENSE` and `THIRD_PARTY.md`.
