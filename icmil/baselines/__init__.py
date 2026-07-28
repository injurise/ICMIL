"""Comparison baselines evaluated alongside ICMIL.

Each baseline exposes the same in-context interface as ICMIL itself —
``forward(X_train, y_train, X_test) -> logits`` — but, unlike ICMIL, fits itself
from scratch on every split. Imports are left to the caller so that using one
baseline does not pull in the dependencies of the others (notably TabPFN, whose
weights download on first use).
"""
