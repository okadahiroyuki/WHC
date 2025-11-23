"""WHC: Waving hand classification."""

from .model import WHC, HSC  # HSC alias kept for compatibility
from .pipeline import main

__all__ = ["WHC", "HSC", "main"]
