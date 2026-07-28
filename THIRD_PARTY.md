# Third-party components

Methods, code and data this repository builds on.

## Methods reimplemented here

**Attention-based MIL** — the gated attention pooling in `icmil/mil_pooling.py`, used
by the `abmil_refit` and `acmil` baselines and by the prior generator's
`embedding_abmil` bag-label rule.

> M. Ilse, J. M. Tomczak, M. Welling. *Attention-based Deep Multiple Instance
> Learning.* ICML 2018. https://arxiv.org/abs/1802.04712

Written from the published equations (Eq. 8 and Eq. 9); layer names, shapes and
construction order follow the implementation these baselines were originally run
against, so seeded runs reproduce identical parameters.

**Attention-Challenging MIL** — `icmil/baselines/acmil_baseline.py` ports the ACMIL-GA
variant (multiple branch attention, stochastic top-K masking, attention-diversity
loss), rewritten to process a mini-batch of bags and to expose the same
`forward(X_train, y_train, X_test)` interface as the other baselines.

> Y. Zhang et al. *Attention-Challenging Multiple Instance Learning for Whole Slide
> Image Classification.* 2023. https://arxiv.org/abs/2311.07125 —
> reference implementation: https://github.com/dazhangyu123/ACMIL

## Packages

**TabPFN** — the `tabpfn_concat`, `tabpfn_subsample` and `cluster_tabpfn` baselines run
a frozen TabPFN v2 backbone. The package downloads its pretrained weights from the
upstream provider on first use; this repository does not redistribute them.

> N. Hollmann et al. *TabPFN: A Transformer That Solves Small Tabular Classification
> Problems in a Second.* ICLR 2023.

**TabICL** — the synthetic prior generator (`icmil/datagen/`, optional extra) uses
TabICL's structural causal models (`MLPSCM`, `TreeSCM`) and its regression-to-
classification utilities to build each synthetic dataset's latent function.

> J. Qu et al. *TabICL: A Tabular Foundation Model for In-Context Learning on Large
> Data.* ICML 2025.

**`cluster_tabpfn`** follows the K-means selective-pooling construction of Kopp et al.,
NeurIPS 2025 AITD Workshop.

## Benchmark data

The `.h5` files in `datasets/` are derived, feature-extracted, fixed-split versions of
public datasets. Original sources and terms:

| File | Source |
|---|---|
| `uci_benchmark.h5` | UCI Musk (v1, v2), Letter Recognition, HEPMASS |
| `andrews_mil_benchmark.h5` | Fox / Tiger / Elephant image-bag datasets, Andrews et al., NeurIPS 2002 |
| `mnist_xai_benchmark_100bags.h5` | MNIST, with MIL rules applied to construct bags |
| `tcga_uni2_luad_vs_lusc.h5` | TCGA LUAD/LUSC whole-slide images, patch features from a pathology foundation model |
| `rsna_ich_resnet50_draws_100bags.h5` | RSNA Intracranial Haemorrhage CT, ResNet-50 slice features (via the `torchmil` dataset release) |

Each retains the licence and usage terms of its source dataset. TCGA and RSNA are
distributed here only as extracted feature vectors, not as images.
