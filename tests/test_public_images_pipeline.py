from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import pytest

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birefnet_experiments.inference.birefnet_inference import collect_inputs, save_outputs


PUBLIC_TEST_IMAGES = {
    "example_jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Example.jpg/320px-Example.jpg",
    "transparency_png": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/320px-PNG_transparency_demonstration_1.png",
}


def _download_public_images(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    for name, url in PUBLIC_TEST_IMAGES.items():
        suffix = ".png" if url.lower().endswith(".png") else ".jpg"
        destination = output_dir / f"{name}{suffix}"
        try:
            urlretrieve(url, destination)
        except (URLError, OSError) as exc:
            pytest.skip(f"Network-restricted environment blocked public image download: {exc}")
        image_paths.append(destination)

    return image_paths


def test_collect_inputs_and_save_outputs_with_public_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_paths = _download_public_images(image_dir)

    collected = collect_inputs(image_dir)
    assert collected == sorted(image_paths)

    for image_path in collected:
        image = Image.open(image_path).convert("RGB")
        h, w = image.height, image.width

        # Synthetic probability map/mask to validate serialization logic.
        prob = np.linspace(0, 1, num=h * w, dtype=np.float32).reshape(h, w)
        mask = (prob >= 0.5).astype(np.uint8) * 255
        masked = np.array(image, dtype=np.uint8)
        masked[mask == 0] = 0

        save_outputs(
            output_dir=tmp_path / "predictions",
            stem=image_path.stem,
            outputs={
                "probability": prob,
                "mask": mask,
                "masked_image": masked,
            },
        )

        sample_dir = tmp_path / "predictions" / image_path.stem
        assert (sample_dir / "mask.png").exists()
        assert (sample_dir / "masked_image.png").exists()
        assert (sample_dir / "raw_probability_map.npy").exists()

        loaded_prob = np.load(sample_dir / "raw_probability_map.npy")
        assert loaded_prob.shape == (h, w)
        assert loaded_prob.dtype == np.float32
