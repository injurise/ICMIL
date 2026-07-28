"""ICMIL: In-Context Multiple-Instance Learning.

Public entry points for loading the trained ICMIL model and reproducing the
benchmark table. See ``icmil.reproduce`` for the single-command CLI.
"""

from icmil.artifacts import SEEDS, resolve_checkpoint, resolve_dataset
from icmil.model import ICMIL_ARCH, ICMILInference, build_icmil, load_icmil

__all__ = [
    "ICMIL_ARCH",
    "SEEDS",
    "ICMILInference",
    "build_icmil",
    "load_icmil",
    "resolve_checkpoint",
    "resolve_dataset",
]
