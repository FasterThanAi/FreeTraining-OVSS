"""
Step 1: Initial SAM 3-Pass Module
Runs SAM (Segment Anything Model) in three passes to produce
identified patches and unidentified regions from an input image + class names.
"""
from .sam_pass import SAMPass

__all__ = ["SAMPass"]
