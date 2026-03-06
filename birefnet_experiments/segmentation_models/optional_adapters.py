from __future__ import annotations

from pathlib import Path
import numpy as np

from .base import SegmentationModelAdapter


class _UnavailableAdapter(SegmentationModelAdapter):
    model_name = "unavailable"

    def __init__(self, **_: object) -> None:
        self._name = self.model_name

    @property
    def name(self) -> str:
        return self._name

    def predict(self, image_path: Path) -> np.ndarray:
        raise NotImplementedError(
            f"{self.model_name} adapter is a placeholder. "
            "Install the corresponding project/dependencies and implement `predict` in "
            "birefnet_experiments/segmentation_models/optional_adapters.py."
        )


class U2NetAdapter(_UnavailableAdapter):
    model_name = "U2Net"


class MODNetAdapter(_UnavailableAdapter):
    model_name = "MODNet"


class SAMAdapter(_UnavailableAdapter):
    model_name = "SAM"


class ISNetAdapter(_UnavailableAdapter):
    model_name = "IS-Net"
