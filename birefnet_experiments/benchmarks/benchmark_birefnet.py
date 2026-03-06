"""Baseline benchmark pipeline for binary segmentation models."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from birefnet_experiments.segmentation_models import create_model


@dataclass
class SampleResult:
    image: str
    iou: float
    dice: float
    mae: float
    precision: float
    recall: float
    f_measure: float
    latency_ms: float


def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    return (np.array(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.uint8)


def compute_metrics(prob: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(np.uint8)

    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    eps = 1e-8
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f_measure = (2 * precision * recall) / (precision + recall + eps)
    mae = np.abs(prob - gt.astype(np.float32)).mean()

    return {
        "iou": float(iou),
        "dice": float(dice),
        "mae": float(mae),
        "precision": float(precision),
        "recall": float(recall),
        "f_measure": float(f_measure),
    }


def list_pairs(dataset_dir: Path) -> List[tuple[Path, Path]]:
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Expected {image_dir} and {mask_dir}")

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in valid_ext])

    pairs = []
    for image_path in images:
        candidate_masks = [mask_dir / f"{image_path.stem}{ext}" for ext in valid_ext]
        matches = [p for p in candidate_masks if p.exists()]
        if not matches:
            continue
        pairs.append((image_path, matches[0]))
    return pairs


def summarize(results: Sequence[SampleResult]) -> Dict[str, float]:
    keys = ["iou", "dice", "mae", "precision", "recall", "f_measure", "latency_ms"]
    summary = {}
    for key in keys:
        values = np.array([getattr(r, key) for r in results], dtype=np.float64)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std())
    summary["num_samples"] = len(results)
    return summary


def write_csv(results: Sequence[SampleResult], path: Path) -> None:
    fieldnames = list(asdict(results[0]).keys()) if results else list(SampleResult.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(asdict(item))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark binary segmentation models")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root with images/ and masks/")
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--model", type=str, default="birefnet", choices=["birefnet", "u2net", "modnet", "sam", "isnet"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="swin_v1_l")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resolutions", type=int, nargs="*", default=None, help="Optional sweep resolutions, e.g. 512 1024 2048")
    return parser.parse_args()


def benchmark_once(args: argparse.Namespace, resolution: int | None) -> Dict[str, object]:
    model = create_model(
        args.model,
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        resolution=resolution,
        device=args.device,
        use_half=args.half,
    )

    pairs = list_pairs(args.dataset)
    if not pairs:
        raise RuntimeError("No image/mask pairs found.")

    results: List[SampleResult] = []
    peak_mem_mb = 0.0

    for image_path, mask_path in pairs:
        gt = load_mask(mask_path)

        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        prob = model.predict(image_path)
        latency_ms = (time.perf_counter() - t0) * 1000

        metrics = compute_metrics(prob, gt, threshold=args.threshold)
        results.append(
            SampleResult(
                image=image_path.name,
                latency_ms=latency_ms,
                **metrics,
            )
        )

        if torch.cuda.is_available() and args.device.startswith("cuda"):
            peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() / (1024 ** 2))

    return {
        "model_name": model.name,
        "resolution": resolution,
        "results": results,
        "summary": summarize(results),
        "peak_gpu_memory_mb": peak_mem_mb,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep = args.resolutions if args.resolutions else [args.resolution]
    comparison_rows = []

    default_results = None
    for resolution in sweep:
        run = benchmark_once(args, resolution)
        suffix = f"reso_{resolution if resolution is not None else 'native'}"

        csv_path = args.output_dir / f"results_{suffix}.csv"
        json_path = args.output_dir / f"results_summary_{suffix}.json"
        write_csv(run["results"], csv_path)

        summary = run["summary"]
        summary["model_name"] = run["model_name"]
        summary["resolution"] = run["resolution"]
        summary["peak_gpu_memory_mb"] = run["peak_gpu_memory_mb"]

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        comparison_rows.append(summary)
        if resolution == args.resolution:
            default_results = run["results"]

    # Required default names.
    final_csv = args.output_dir / "results.csv"
    final_json = args.output_dir / "results_summary.json"
    if default_results is None:
        default_results = benchmark_once(args, args.resolution)["results"]
    write_csv(default_results, final_csv)
    with final_json.open("w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2)

    print(f"Saved benchmark outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
