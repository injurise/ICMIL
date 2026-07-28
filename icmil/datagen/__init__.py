"""Synthetic prior generation and the H5 format the trainer reads.

``generate.py`` writes one H5 per prior arm; ``h5_dataset.py`` reads them back and
mixes the arms during training. ``config.py`` holds the recipe used for the paper.
"""
