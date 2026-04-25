import os
BASE = "/home/hpc/iwi5/iwi5419h/vase_urn_project" if os.path.exists("/home/hpc") else os.path.dirname(os.path.abspath(__file__))

import torch
import numpy as np
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_emb = np.load(os.path.join(BASE, "train_embeddings.npy"))
train_lbl = np.load(os.path.join(BASE, "train_labels.npy"))

class VaseDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

dataset = VaseDataset(train_emb, train_lbl)
g = torch.Generator()
g.manual_seed(SEED)
loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)

model = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512, 128)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using: {device}")

num_classes = int(train_lbl.max()) + 1
loss_fn = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=128, margin=28.6, scale=64).to(device)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(loss_fn.parameters()),
    lr=0.001,
)

epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for emb, lbl in loader:
        emb, lbl = emb.to(device), lbl.to(device)

        output = model(emb)
        loss = loss_fn(output, lbl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), os.path.join(BASE, "model.pth"))
print("Done! Model saved.")