import os
from typing import Literal

PathType = str | os.PathLike

INTERPOLATION_METHODS = Literal["linear", "spline", "time"]

DATASETS = Literal["mihaela", "telco"]
