import argparse

import faiss
import numpy as np
import torch
from torch import nn

from common import l2_normalize, project_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="runs/pose_triplet")
    parser.add_argument("--model-path", default=None)
    return parser.parse_args()


def load_projection(model_path, input_dim):
    checkpoint = torch.load(model_path, map_location="cpu")
    output_dim = int(checkpoint.get("output_dim", 64))
    hidden_dim = int(checkpoint.get("hidden_dim", 128))
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main():
    args = parse_args()
    base = project_root()
    run_dir = base / args.input_dir

    test_emb = np.load(run_dir / "test_embeddings.npy").astype("float32")
    test_lbl = np.load(run_dir / "test_labels.npy")

    if args.model_path:
        print(f"Applying projection model: {args.model_path}")
        model = load_projection(base / args.model_path, test_emb.shape[1])
        with torch.no_grad():
            test_emb = model(torch.tensor(test_emb, dtype=torch.float32)).numpy().astype("float32")

    test_emb = l2_normalize(test_emb).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(test_emb.shape[1])
    index.add(test_emb)
    print(f"FAISS index built with {index.ntotal} vectors")

    print("Searching full test ranking...")
    distances, indices = index.search(test_emb, k=len(test_emb))

    np.save(run_dir / "retrieval_distances.npy", distances)
    np.save(run_dir / "retrieval_indices.npy", indices)
    np.save(run_dir / "retrieval_labels.npy", test_lbl)
    print(f"Done. Retrieval results saved to: {run_dir}")


if __name__ == "__main__":
    main()
