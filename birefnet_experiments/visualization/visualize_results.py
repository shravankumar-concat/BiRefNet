"""Visualization utilities for qualitative segmentation inspection."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path: Path, mode: str) -> np.ndarray:
    return np.array(Image.open(path).convert(mode))


def make_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = np.zeros_like(image_rgb)
    color[..., 1] = 255
    binary = (mask > 127).astype(np.uint8)[..., None]
    return (image_rgb * (1 - alpha * binary) + color * (alpha * binary)).astype(np.uint8)


def fig_to_base64(fig) -> str:
    buff = BytesIO()
    fig.savefig(buff, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def visualize_sample(image_path: Path, pred_path: Path, gt_path: Path, out_path: Path) -> str:
    image = load_image(image_path, "RGB")
    pred = load_image(pred_path, "L")
    gt = load_image(gt_path, "L")
    overlay = make_overlay(image, pred)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(pred, cmap="gray"); axes[1].set_title("Predicted mask"); axes[1].axis("off")
    axes[2].imshow(gt, cmap="gray"); axes[2].set_title("Ground truth"); axes[2].axis("off")
    axes[3].imshow(overlay); axes[3].set_title("Overlay"); axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig_to_base64(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True, help="dataset/images and dataset/masks")
    p.add_argument("--predictions", type=Path, required=True, help="Directory with mask files")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=25)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.dataset / "images"
    gt_dir = args.dataset / "masks"

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in valid_ext][: args.limit]

    rows: List[str] = []
    for image_path in images:
        pred_path = args.predictions / f"{image_path.stem}.png"
        gt_path = gt_dir / f"{image_path.stem}.png"
        if not (pred_path.exists() and gt_path.exists()):
            continue

        out_plot = args.output_dir / f"{image_path.stem}_viz.png"
        img_b64 = visualize_sample(image_path, pred_path, gt_path, out_plot)
        rows.append(f"<h3>{image_path.name}</h3><img src='data:image/png;base64,{img_b64}'/>")

    html = """<html><head><meta charset='utf-8'><title>BiRefNet qualitative report</title></head><body>
    <h1>Segmentation qualitative report</h1>
    {rows}
    </body></html>""".format(rows="\n".join(rows))

    (args.output_dir / "report.html").write_text(html, encoding="utf-8")
    print(f"Saved visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
