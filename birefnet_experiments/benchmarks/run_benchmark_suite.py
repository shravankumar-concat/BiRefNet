"""Run benchmark_birefnet.py across dataset profiles and resolutions."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", type=Path, default=Path("birefnet_experiments/benchmarks/dataset_profiles.yaml"))
    p.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset keys in profiles")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backbone", default="swin_v1_l")
    p.add_argument("--device", default="cuda")
    p.add_argument("--half", action="store_true")
    p.add_argument("--resolutions", nargs="*", type=int, default=[512, 1024, 2048])
    p.add_argument("--output-root", type=Path, default=Path("birefnet_experiments/results/benchmarks"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = yaml.safe_load(args.profiles.read_text())
    datasets = data["datasets"]
    selected = args.datasets or list(datasets.keys())

    args.output_root.mkdir(parents=True, exist_ok=True)
    suite_summary = {}

    for ds_name in selected:
        ds_meta = datasets[ds_name]
        ds_path = Path(ds_meta["path"])
        output_dir = args.output_root / ds_name
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "birefnet_experiments/benchmarks/benchmark_birefnet.py",
            "--dataset",
            str(ds_path),
            "--output-dir",
            str(output_dir),
            "--checkpoint",
            str(args.checkpoint),
            "--backbone",
            args.backbone,
            "--device",
            args.device,
            "--resolutions",
            *[str(r) for r in args.resolutions],
        ]
        if args.half:
            cmd.append("--half")

        subprocess.run(cmd, check=True)

        suite_summary[ds_name] = json.loads((output_dir / "results_summary.json").read_text())

    with (args.output_root / "suite_summary.json").open("w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)


if __name__ == "__main__":
    main()
