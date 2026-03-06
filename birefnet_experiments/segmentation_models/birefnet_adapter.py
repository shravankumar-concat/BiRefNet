from __future__ import annotations

from pathlib import Path
import numpy as np

from .base import SegmentationModelAdapter
from birefnet_experiments.inference.birefnet_inference import BiRefNetInference


class BiRefNetAdapter(SegmentationModelAdapter):
    def __init__(self, checkpoint: Path, backbone: str, resolution: int | None, device: str, use_half: bool) -> None:
        self._name = f"BiRefNet[{backbone}]"
        self._runner = BiRefNetInference(
            checkpoint=checkpoint,
            backbone=backbone,
            resolution=resolution,
            device=device,
            use_half=use_half,
        )

    @property
    def name(self) -> str:
        return self._name

    def predict(self, image_path: Path) -> np.ndarray:
        return self._runner.predict_image(image_path)["probability"]
