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

#### Number of labeled bags

```bash
python -m icmil.bag_count
python -m icmil.bag_count --benchmarks rsna_ich --bag-counts 25,100 --baselines none
```

## Licence

MIT, see `LICENSE` and `THIRD_PARTY.md`.
