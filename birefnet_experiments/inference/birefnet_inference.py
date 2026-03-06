"""Reusable BiRefNet inference wrapper.

Usage examples
--------------
python birefnet_experiments/inference/birefnet_inference.py \
  --checkpoint /path/to/epoch_120.pth \
  --input /data/example.jpg \
  --output-dir birefnet_experiments/results/inference

python birefnet_experiments/inference/birefnet_inference.py \
  --checkpoint /path/to/epoch_120.pth \
  --input /data/images_dir \
  --output-dir birefnet_experiments/results/inference \
  --backbone swin_v1_t --resolution 1024 --half
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config as birefnet_config

BACKBONE_CHANNELS = {
    "vgg16": [512, 512, 256, 128],
    "vgg16bn": [512, 512, 256, 128],
    "resnet50": [2048, 1024, 512, 256],
    "swin_v1_l": [1536, 768, 384, 192],
    "swin_v1_b": [1024, 512, 256, 128],
    "swin_v1_s": [768, 384, 192, 96],
    "swin_v1_t": [768, 384, 192, 96],
    "pvt_v2_b5": [512, 320, 128, 64],
    "pvt_v2_b2": [512, 320, 128, 64],
    "pvt_v2_b1": [512, 320, 128, 64],
    "pvt_v2_b0": [256, 160, 64, 32],
    "dino_v3_7b": [4096, 4096, 4096, 4096],
    "dino_v3_h_plus": [1280, 1280, 1280, 1280],
    "dino_v3_l": [1024, 1024, 1024, 1024],
    "dino_v3_b": [768, 768, 768, 768],
    "dino_v3_s_plus": [384, 384, 384, 384],
    "dino_v3_s": [384, 384, 384, 384],
}


def _apply_backbone_override(backbone: str) -> None:
    """Patch Config.__init__ to support runtime backbone selection."""
    if backbone not in BACKBONE_CHANNELS:
        raise ValueError(f"Unsupported backbone '{backbone}'.")

    original_init = birefnet_config.Config.__init__

    def patched_init(self) -> None:
        original_init(self)
        self.bb = backbone
        self.freeze_bb = "dino_v3" in backbone
        channels = BACKBONE_CHANNELS[backbone]
        if self.mul_scl_ipt == "cat":
            channels = [ch * 2 for ch in channels]
        self.lateral_channels_in_collection = channels
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num :] if self.cxt_num else []

    birefnet_config.Config.__init__ = patched_init


class BiRefNetInference:
    def __init__(
        self,
        checkpoint: Path,
        backbone: str,
        resolution: Optional[int],
        device: str,
        use_half: bool,
    ) -> None:
        _apply_backbone_override(backbone)

        from models.birefnet import BiRefNet
        from utils import check_state_dict

        self.device = torch.device(device)
        self.use_half = use_half and self.device.type == "cuda"
        self.resolution = resolution

        self.model = BiRefNet(bb_pretrained=False)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(check_state_dict(weights), strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.image_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _prepare_image(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        image = image.convert("RGB")
        original_size = (image.height, image.width)
        if self.resolution is not None:
            image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        tensor = self.image_tf(image).unsqueeze(0)
        return tensor, original_size

    def predict_image(self, image_path: Path) -> Dict[str, np.ndarray]:
        image = Image.open(image_path)
        x, (h, w) = self._prepare_image(image)
        x = x.to(self.device)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_half
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            pred = self.model(x)
            if isinstance(pred, list):
                pred = pred[-1]
            prob = torch.sigmoid(pred)
            prob = torch.nn.functional.interpolate(prob, size=(h, w), mode="bilinear", align_corners=True)

        prob_map = prob.squeeze().float().cpu().numpy().clip(0.0, 1.0)
        mask = (prob_map >= 0.5).astype(np.uint8) * 255

        img_np = np.array(image.convert("RGB"), dtype=np.uint8)
        masked = img_np.copy()
        masked[mask == 0] = 0

        return {
            "probability": prob_map,
            "mask": mask,
            "masked_image": masked,
        }


def collect_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([p for p in input_path.iterdir() if p.suffix.lower() in valid_ext])


def save_outputs(output_dir: Path, stem: str, outputs: Dict[str, np.ndarray]) -> None:
    sample_dir = output_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(outputs["mask"]).save(sample_dir / "mask.png")
    Image.fromarray(outputs["masked_image"]).save(sample_dir / "masked_image.png")
    np.save(sample_dir / "raw_probability_map.npy", outputs["probability"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BiRefNet inference wrapper")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="swin_v1_l", choices=sorted(BACKBONE_CHANNELS.keys()))
    parser.add_argument("--resolution", type=int, default=None, help="If omitted, uses native image resolution")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true", help="Use fp16 autocast on CUDA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inferencer = BiRefNetInference(
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        resolution=args.resolution,
        device=args.device,
        use_half=args.half,
    )

    image_paths = collect_inputs(args.input)
    if not image_paths:
        raise RuntimeError(f"No images found at {args.input}")

    start = time.perf_counter()
    for image_path in image_paths:
        outputs = inferencer.predict_image(image_path)
        save_outputs(args.output_dir, image_path.stem, outputs)
    elapsed = time.perf_counter() - start

    print(f"Processed {len(image_paths)} images in {elapsed:.3f}s ({elapsed / len(image_paths):.3f}s / image).")


if __name__ == "__main__":
    main()
