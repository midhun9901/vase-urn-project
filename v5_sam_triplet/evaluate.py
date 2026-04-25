import os
BASE = os.environ.get("VASE_PROJECT_DIR", os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

print("Loading retrieval results...")
indices = np.load(os.path.join(BASE, "retrieval_indices.npy"))
labels = np.load(os.path.join(BASE, "retrieval_labels.npy"))

def compute_metrics(indices, labels):
    n = len(labels)
    ap_list = []
    acc1 = 0
    acc10 = 0

    for i in range(n):
        query_label = labels[i]
        ranked = indices[i]
        ranked_labels = labels[ranked[1:]]

        if ranked_labels[0] == query_label:
            acc1 += 1

        if query_label in ranked_labels[:10]:
            acc10 += 1

        correct = 0
        precision_sum = 0
        for rank, lbl in enumerate(ranked_labels):
            if lbl == query_label:
                correct += 1
                precision_sum += correct / (rank + 1)

        total_relevant = np.sum(labels == query_label) - 1
        if total_relevant > 0:
            ap_list.append(precision_sum / total_relevant)
        else:
            ap_list.append(0.0)

    mAP = np.mean(ap_list)
    acc1 = acc1 / n * 100
    acc10 = acc10 / n * 100

    return mAP, acc1, acc10

print("Computing metrics...")
mAP, acc1, acc10 = compute_metrics(indices, labels)

print(f"mAP        : {mAP:.4f}")
print(f"Accuracy@1 : {acc1:.2f}%")
print(f"Accuracy@10: {acc10:.2f}%")
print("Evaluation done!")
