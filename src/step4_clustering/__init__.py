"""
Step 4: Contextual Clustering Module
Labels unidentified patches using contextual clustering,
guided by the co-occurrence matrix M and INFO from Step 2.
"""
from .contextual_clustering import ContextualClusterer

__all__ = ["ContextualClusterer"]
