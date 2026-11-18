import matplotlib.pyplot as plt
import numpy as np

# Load metrics from the latest training run
data = np.load("training_history.npz")
train_loss = data["loss"].tolist()
test_acc = data["acc"].tolist()
epochs = list(range(1, len(train_loss) + 1))

plt.rcParams.update(
    {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

# Left plot: Training Loss
ax1.plot(epochs, train_loss, marker="o", linewidth=2.5, color="#1f77b4")
ax1.set_title("Training Loss per Epoch", fontsize=14, fontweight="bold", pad=10)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_xticks(epochs)
ax1.grid(True, linestyle="--", alpha=0.4)

# Right plot: Test Accuracy
best_acc = max(test_acc)
best_epoch = epochs[test_acc.index(best_acc)]

ax2.plot(epochs, test_acc, marker="o", linewidth=2.5, color="#2ca02c")
ax2.scatter(
    [best_epoch],
    [best_acc],
    color="#d62728",
    s=80,
    zorder=5,
    label=f"Best: {best_acc:.2f}% (Epoch {best_epoch})",
)
ax2.set_title("KMNIST Test Accuracy per Epoch", fontsize=14, fontweight="bold", pad=10)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_xticks(epochs)
ax2.set_ylim(min(test_acc) - 2, max(test_acc) + 2)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(frameon=False, loc="lower right")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300, bbox_inches="tight")
print("Saved training_curves.png")
plt.show()
