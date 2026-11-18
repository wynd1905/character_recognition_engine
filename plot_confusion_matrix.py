import matplotlib.pyplot as plt
import numpy as np

from confusion_matrix import ROMAJI, compute_confusion_matrix
from network import KanaNetwork

# 1. Load test data and best saved weights
test_imgs = np.load("data/kmnist-test-imgs.npz")["arr_0"]
test_labels = np.load("data/kmnist-test-labels.npz")["arr_0"]
X_test = test_imgs.reshape(-1, 784).astype(np.float64) / 255.0

net = KanaNetwork(input_size=784, hidden_size=128, output_size=10)
net.load_weights("kana_weights.npz")

# 2. Compute 10x10 confusion matrix via your C backend
cm = compute_confusion_matrix(net, X_test, test_labels)
overall_acc = (np.trace(cm) / cm.sum()) * 100.0

# 3. Render presentation-ready heatmap
fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Number of Samples")

# Configure axis ticks and labels
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(ROMAJI, fontsize=11)
ax.set_yticklabels(ROMAJI, fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel("Actual Label", fontsize=12, fontweight="bold", labelpad=10)
ax.set_title(
    f"KMNIST Confusion Matrix (Test Accuracy: {overall_acc:.2f}%)",
    fontsize=14,
    fontweight="bold",
    pad=14,
)

# Annotate every cell with its count (white text on dark cells, black on light)
thresh = cm.max() / 2.0
for i in range(10):
    for j in range(10):
        color = "white" if cm[i, j] > thresh else "black"
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=10)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
print("Saved confusion_matrix.png")
plt.show()
