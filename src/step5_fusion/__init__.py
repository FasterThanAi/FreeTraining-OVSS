"""
Step 5: Final Mask Fusion Module
Merges identified patches (Step 1/2) with labeled unidentified patches (Step 4)
to produce the complete labeled output mask.
"""
from .mask_fusion import MaskFusion

__all__ = ["MaskFusion"]
