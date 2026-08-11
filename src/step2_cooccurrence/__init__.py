"""
Step 2: Co-occurrence Matrix Builder
Builds the co-occurrence matrix M and additional INFO
from identified patches produced by Step 1.
"""
from .cooccurrence import CooccurrenceBuilder

__all__ = ["CooccurrenceBuilder"]
