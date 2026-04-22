import argparse
import json

import numpy as np

from common import compute_metrics, project_root


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval results with full ranking mAP.")
    parser.add_argument("--input-dir", default="runs/dinov2_full")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = project_root() / args.input_dir

    indices = np.load(run_dir / "retrieval_indices.npy")
    labels = np.load(run_dir / "retrieval_labels.npy")
    metrics = compute_metrics(indices, labels)

    print("Evaluation results:")
    print(f"mAP        : {metrics['mAP']:.4f}")
    print(f"Accuracy@1 : {metrics['Accuracy@1']:.2f}%")
    print(f"Accuracy@10: {metrics['Accuracy@10']:.2f}%")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
