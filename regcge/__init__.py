"""regcge: a flexible recursive-dynamic regional tax CGE model.

Inspired by the Texas State Tax Analysis Modeling Program (STAMP; Beacon Hill
Institute) and the California DRAM, with utility-consistent households, exact
welfare measures, a generic tax-instrument layer and switchable closures.
"""

from .data.dataset import Dataset
from .model.model import RegionalCGE
from .model.calibrate import Calibration
from .model.closures import Closure

__version__ = "0.1.0"
__all__ = ["Dataset", "RegionalCGE", "Calibration", "Closure"]
