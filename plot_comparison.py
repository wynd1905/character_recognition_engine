import numpy as np
import matplotlib.pyplot as plt

runs = [
    ("Pure SGD (BS=1)", "history_pure_sgd.npz", "#d62728"),
    ("Vanilla Mini-Batch (BS=64)", "history_vanilla_minibatch.npz", "#ff7f0e"),
    ("Mini-Batch + Momentum + L2 + Decay", "training_history.npz", "#2ca02c"),
]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

for label, filename, color in runs:
    data = np.load(filename)
    loss = data["loss"]
    acc = data["acc"]
    epochs = list(range(1, len(loss) + 1))

    if "times" in data:
        label = f"{label} ({np.mean(data['times']):.1f}s/ep)"

    ax1.plot(epochs, loss, marker="o", linewidth=2.2, color=color, label=label)
    ax2.plot(epochs, acc, marker="o", linewidth=2.2, color=color, label=label)

ax1.set_title("Training Loss Comparison", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_xticks(epochs)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2.set_title("KMNIST Test Accuracy Comparison", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Test Accuracy (%)")
ax2.set_xticks(epochs)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(frameon=False, loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("optimizer_comparison.png", dpi=300, bbox_inches="tight")
print("Saved optimizer_comparison.png")
plt.show()
