import os
BASE = os.environ.get("VASE_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
CROPS_DIR = os.path.join(BASE, "crops")

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract(folders):
    embeddings = []
    labels = []
    for label, folder in enumerate(folders):
        for file in os.listdir(folder):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, file)
            # use SAM crop if available, otherwise full image
            rel = os.path.relpath(path, BASE)
            crop_path = os.path.join(CROPS_DIR, rel)
            load_path = crop_path if os.path.exists(crop_path) else path
            try:
                img = Image.open(load_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(img).squeeze().cpu().numpy()
                embeddings.append(feat)
                labels.append(label)
            except Exception as e:
                print(f"Skipped {path}: {e}")
    return np.array(embeddings), np.array(labels)

with open(os.path.join(BASE, "train_folders.txt")) as f:
    train_folders = f.read().splitlines()

with open(os.path.join(BASE, "test_folders.txt")) as f:
    test_folders = f.read().splitlines()

print("Extracting train features...")
train_emb, train_lbl = extract(train_folders)
np.save(os.path.join(BASE, "train_embeddings.npy"), train_emb)
np.save(os.path.join(BASE, "train_labels.npy"), train_lbl)
print(f"Train: {train_emb.shape}")

print("Extracting test features...")
test_emb, test_lbl = extract(test_folders)
np.save(os.path.join(BASE, "test_embeddings.npy"), test_emb)
np.save(os.path.join(BASE, "test_labels.npy"), test_lbl)
print(f"Test: {test_emb.shape}")

print("Done! Embeddings saved.")
