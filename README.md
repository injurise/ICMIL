# In-Context Multiple-Instance Learning

ICMIL is a Prior-data Fitted Network for bag-structured data. It is pretrained
purely on synthetic MIL tasks and solves new ones in a single forward pass without
gradient updates, hyperparameter tuning, or task-specific training.

This repo provides code to train and run ICMIL models, generate data from our 
priors, evaluate baselines on a suite of 12 benchmarks, and to reproduce the 
benchmark table found in our paper using our checkpoints. 

## Setup

```bash
pip install -e .
pip install -e ".[datagen]"
```

## Using the model

```python
from icmil import load_icmil

model = load_icmil(seed="c5trd795", device="cuda")

# X_train is (1, n_bags, bag_size, n_features), y_train is (1, n_bags).
logits = model(X_train, y_train, X_test)      # (1, n_query_bags, n_classes)
```

## Prior

```bash
python -m icmil.datagen.generate --dry-run
python -m icmil.datagen.generate --num-batches 100 --out-dir /tmp/demo
```

## Training

```bash
python -m icmil.train --data-dir workdir/priors --out checkpoints/my-icmil.pt
```

## Reproducing the table

```bash
python -m icmil.reproduce
python -m icmil.reproduce --tasks uci_musk1 --baselines mean_logreg
```

## Licence

MIT, see `LICENSE` and `THIRD_PARTY.md`.
