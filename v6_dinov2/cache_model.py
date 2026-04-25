import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Download/cache DINOv2 for offline cluster jobs.")
    parser.add_argument("--torch-home", default="/home/woody/iwi5/iwi5419h/torch_cache")
    parser.add_argument("--model-name", default="dinov2_vits14")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["TORCH_HOME"] = args.torch_home
    print(f"TORCH_HOME={args.torch_home}")
    print(f"Caching DINOv2 model: {args.model_name}")
    model = torch.hub.load("facebookresearch/dinov2", args.model_name)
    print(f"Cached model class: {type(model).__name__}")


if __name__ == "__main__":
    main()
