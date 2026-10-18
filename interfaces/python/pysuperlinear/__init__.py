"""superlinear's native C++ inference, without subprocesses or CFG files."""
from ._core import SuperlinearModel
from .calculator import SuperlinearCalculator

__all__ = ["SuperlinearModel", "SuperlinearCalculator"]
__version__ = "0.1.0"
