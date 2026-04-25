import argparse

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from common import image_files, load_or_create_split, project_root, save_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/v5")
    parser.add_argument("--crops-dir", default="runs/v5/crops")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def load_model(device):
    print("Loading ResNet50 (pretrained, no classifier head)")
    base = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(base.children())[:-1])
    model.eval().to(device)
    return model


def extract_split(model, folders, base, crops_dir, device, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    embeddings, labels, paths = [], [], []
    batch, batch_labels, batch_paths = [], [], []

    def flush():
        if not batch:
            return
        tensor = torch.stack(batch).to(device)
        with torch.no_grad():
            feats = model(tensor).squeeze(-1).squeeze(-1)
        embeddings.extend(feats.detach().cpu().numpy())
        labels.extend(batch_labels)
        paths.extend(batch_paths)
        batch.clear(); batch_labels.clear(); batch_paths.clear()

    for label, folder in enumerate(folders):
        for img_path in image_files(folder):
            rel = img_path.relative_to(base)
            crop_path = crops_dir / rel
            load_path = crop_path if crop_path.exists() else img_path
            try:
                img = Image.open(load_path).convert("RGB")
                batch.append(transform(img))
                batch_labels.append(label)
                batch_paths.append(img_path)
                if len(batch) >= batch_size:
                    flush()
            except Exception as e:
                print(f"Skipped {img_path}: {e}")
    flush()
    return np.asarray(embeddings, dtype=np.float32), np.asarray(labels, dtype=np.int64), paths


def main():
    args = parse_args()
    base = project_root()
    out = base / args.output_dir
    crops_dir = base / args.crops_dir
    out.mkdir(parents=True, exist_ok=True)

    train_folders, test_folders = load_or_create_split(base)
    print(f"Train: {len(train_folders)} | Test: {len(test_folders)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    model = load_model(device)

    print("Extracting train features (from SAM crops where available)...")
    train_emb, train_lbl, train_paths = extract_split(model, train_folders, base, crops_dir, device, args.batch_size)
    np.save(out / "train_embeddings.npy", train_emb)
    np.save(out / "train_labels.npy", train_lbl)
    save_paths(out / "train_image_paths.txt", train_paths)
    print(f"Train embeddings: {train_emb.shape}")

    print("Extracting test features (from SAM crops where available)...")
    test_emb, test_lbl, test_paths = extract_split(model, test_folders, base, crops_dir, device, args.batch_size)
    np.save(out / "test_embeddings.npy", test_emb)
    np.save(out / "test_labels.npy", test_lbl)
    save_paths(out / "test_image_paths.txt", test_paths)
    print(f"Test embeddings: {test_emb.shape}")

    print(f"Done. Features saved to: {out}")


if __name__ == "__main__":
    main()
