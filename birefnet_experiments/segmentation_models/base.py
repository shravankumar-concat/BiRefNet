from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class SegmentationModelAdapter(ABC):
    """Common interface for segmentation model benchmarking."""

    @abstractmethod
    def predict(self, image_path: Path) -> np.ndarray:
        """Return a float probability map in [0, 1] with input image resolution."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...
