import os
BASE = "/home/hpc/iwi5/iwi5419h/vase_urn_project" if os.path.exists("/home/hpc") else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import faiss

print("Loading embeddings...")
test_emb = np.load(os.path.join(BASE, "test_embeddings.npy"))
test_lbl = np.load(os.path.join(BASE, "test_labels.npy"))

# Normalize for cosine similarity
test_emb = test_emb / np.linalg.norm(test_emb, axis=1, keepdims=True)

print("Building FAISS index (raw 2048-dim, no metric learning)...")
index = faiss.IndexFlatL2(2048)
index.add(test_emb)
print(f"FAISS index built with {index.ntotal} vectors")

print("Searching...")
D, I = index.search(test_emb, k=10)

np.save(os.path.join(BASE, "retrieval_indices_baseline.npy"), I)
np.save(os.path.join(BASE, "retrieval_labels_baseline.npy"), test_lbl)
print("Done! Baseline retrieval results saved.")
