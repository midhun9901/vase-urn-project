import os
BASE = os.environ.get("VASE_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import faiss
from torch import nn

print("Loading embeddings...")
test_emb = np.load(os.path.join(BASE, "test_embeddings.npy"))
test_lbl = np.load(os.path.join(BASE, "test_labels.npy"))

print("Loading model...")
model = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512, 128)
)
model.load_state_dict(torch.load(os.path.join(BASE, "model.pth")))
model.eval()

print("Running model on test embeddings...")
with torch.no_grad():
    refined = model(torch.tensor(test_emb, dtype=torch.float32)).numpy()

refined = refined / np.linalg.norm(refined, axis=1, keepdims=True)

print("Building FAISS index...")
index = faiss.IndexFlatL2(128)
index.add(refined)
print(f"FAISS index built with {index.ntotal} vectors")

print("Searching...")
D, I = index.search(refined, k=10)

np.save(os.path.join(BASE, "retrieval_indices.npy"), I)
np.save(os.path.join(BASE, "retrieval_labels.npy"), test_lbl)
print("Done! Retrieval results saved.")
