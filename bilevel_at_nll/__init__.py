# -*- coding: utf-8 -*-
"""Bi-level DCT adversarial training package."""

from .config import TrainSummary, build_parser


def train(*args, **kwargs):
    """Import the training entry point lazily to keep ``python -m`` clean."""
    from .train import train as _train

    return _train(*args, **kwargs)

__all__ = ["TrainSummary", "build_parser", "train"]
