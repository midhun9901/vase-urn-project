import os
import random

BASE = os.environ.get("VASE_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))

def first_existing(paths):
    for path in paths:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("None of these dataset paths exist:\n" + "\n".join(paths))

pt1 = first_existing([
    os.path.join(BASE, "data", "pairs_pt1", "Bildpaare Teil 1"),
    os.path.join(BASE, "pairs_pt1 (1)", "Bildpaare Teil 1"),
    os.path.join(BASE, "pairs_pt1", "Bildpaare Teil 1"),
])
pt2 = first_existing([
    os.path.join(BASE, "data", "pairs_pt2", "Bildpaare-Triplets Teil 2"),
    os.path.join(BASE, "pairs_pt2 (1)", "Bildpaare-Triplets Teil 2"),
    os.path.join(BASE, "pairs_pt2", "Bildpaare-Triplets Teil 2"),
])

folders = []

for base in [pt1, pt2]:
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            folders.append(path)

print(f"Total folders found: {len(folders)}")

random.seed(42)
random.shuffle(folders)

split = int(0.8 * len(folders))
train = folders[:split]
test = folders[split:]

print(f"Train: {len(train)} | Test: {len(test)}")

with open(os.path.join(BASE, "train_folders.txt"), "w") as f:
    f.write("\n".join(train))

with open(os.path.join(BASE, "test_folders.txt"), "w") as f:
    f.write("\n".join(test))

print("Done! train_folders.txt and test_folders.txt saved.")
