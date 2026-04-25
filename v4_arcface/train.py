import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_metric_learning import losses, samplers

from common import SEED, project_root


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="runs/v4")
    parser.add_argument("--output-dir", default="runs/v4")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--margin", type=float, default=28.6)
    parser.add_argument("--scale", type=float, default=64.0)
    return parser.parse_args()


def main():
    args = parse_args()
    base = project_root()
    in_dir = base / args.input_dir
    out_dir = base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_emb = np.load(in_dir / "train_embeddings.npy")
    train_lbl = np.load(in_dir / "train_labels.npy")
    input_dim = train_emb.shape[1]
    num_classes = int(train_lbl.max()) + 1

    dataset = EmbeddingDataset(train_emb, train_lbl)
    sampler = samplers.MPerClassSampler(
        train_lbl, m=2, batch_size=args.batch_size, length_before_new_iter=len(train_lbl),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    model = nn.Sequential(
        nn.Linear(input_dim, args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, args.output_dim),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using: {device} | Input dim: {input_dim} | Classes: {num_classes}")

    loss_fn = losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=args.output_dim,
        margin=args.margin,
        scale=args.scale,
    ).to(device)

    # ArcFace has learnable proxy parameters — optimise jointly
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss, batches = 0.0, 0
        for emb, lbl in loader:
            emb, lbl = emb.to(device), lbl.to(device)
            output = F.normalize(model(emb), p=2, dim=1)
            loss = loss_fn(output, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {total_loss / max(1, batches):.4f}")

    checkpoint = {
        "model": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.output_dim,
    }
    torch.save(checkpoint, out_dir / "model.pth")
    print(f"Done. Model saved to: {out_dir / 'model.pth'}")


if __name__ == "__main__":
    main()
