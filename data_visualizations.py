import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Load KMNIST Test Data & Saved Weights
# ------------------------------------------------------------------
ROMAJI = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]

test_imgs = np.load("data/kmnist-test-imgs.npz")["arr_0"]      # (10000, 28, 28)
test_labels = np.load("data/kmnist-test-labels.npz")["arr_0"]  # (10000,)
X_test = test_imgs.reshape(-1, 784).astype(np.float64) / 255.0

weights = np.load("kana_weights.npz")
W1, B1 = weights["W1"], weights["B1"]  # (128, 784), (128, 1)
W2, B2 = weights["W2"], weights["B2"]  # (10, 128),  (10, 1)

# Vectorized forward pass matching your C engine (column-major)
Z1 = W1 @ X_test.T + B1
A1 = np.maximum(0.0, Z1)               # (128, 10000)

Z2 = W2 @ A1 + B2
Z2_shift = Z2 - np.max(Z2, axis=0, keepdims=True)
exp_Z2 = np.exp(Z2_shift)
A2 = exp_Z2 / np.sum(exp_Z2, axis=0, keepdims=True)  # (10, 10000)

preds = np.argmax(A2, axis=0)
confidences = np.max(A2, axis=0)
correct_mask = preds == test_labels
overall_acc = np.mean(correct_mask) * 100.0
print(f"Loaded kana_weights.npz | Test Accuracy: {overall_acc:.2f}%")

plt.rcParams.update({"font.size": 11})


# ------------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------------
cm = np.zeros((10, 10), dtype=int)
for t, p in zip(test_labels, preds):
    cm[t, p] += 1

cm_errors = np.ma.masked_where(np.eye(10, dtype=bool), cm)

fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=300)
im = ax.imshow(cm_errors, interpolation="nearest", cmap="Reds")
ax.set_facecolor("#dceefb")  # Soft blue for the masked diagonal

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Misclassified Samples", fontsize=11, fontweight="bold")

ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(ROMAJI, fontsize=11)
ax.set_yticklabels(ROMAJI, fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
ax.set_ylabel("Actual Label", fontsize=12, fontweight="bold")
ax.set_title(
    f"KMNIST Error-Focused Confusion Matrix (Accuracy: {overall_acc:.2f}%)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)

# Subtle white gridlines between cells
ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

max_err = cm_errors.max()
for i in range(10):
    for j in range(10):
        val = cm[i, j]
        if i == j:
            pct = (val / cm[i].sum()) * 100.0
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center", color="#0b3d91",
                fontsize=8.5, fontweight="bold",
            )
        else:
            color = "white" if val > max_err * 0.55 else "#222222"
            weight = "bold" if val >= 25 else "normal"
            ax.text(
                j, i, f"{val}",
                ha="center", va="center", color=color,
                fontsize=9.5, fontweight=weight,
            )

plt.tight_layout()
plt.savefig("confusion_matrix_redesigned.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved confusion_matrix_redesigned.png")


# ------------------------------------------------------------------
# Hidden Layer Weight Templates (W1 reshaped to 28x28)
# ------------------------------------------------------------------
# Pick the 16 neurons with the strongest weight norms (most active filters)
neuron_norms = np.linalg.norm(W1, axis=1)
top_neurons = np.argsort(neuron_norms)[::-1][:16]

fig, axes = plt.subplots(4, 4, figsize=(7.5, 7.5), dpi=300)
vmax = np.percentile(np.abs(W1[top_neurons]), 99)

for ax, idx in zip(axes.flat, top_neurons):
    template = W1[idx].reshape(28, 28)
    ax.imshow(template, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Neuron #{idx}", fontsize=9)
    ax.axis("off")

fig.suptitle(
    "Learned Hidden-Layer Weight Templates (W1)\nRed = Excitatory (+) | Blue = Inhibitory (-)",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("w1_weight_templates.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved w1_weight_templates.png")


# ------------------------------------------------------------------
# High-Confidence Failure Cases (Error Gallery)
# ------------------------------------------------------------------
wrong_indices = np.where(~correct_mask)[0]
# Sort wrong predictions by highest confidence first
top_wrong = wrong_indices[np.argsort(confidences[wrong_indices])[::-1][:10]]

fig, axes = plt.subplots(2, 5, figsize=(11, 5.2), dpi=300)
for ax, idx in zip(axes.flat, top_wrong):
    ax.imshow(test_imgs[idx], cmap="gray_r")
    true_lbl = ROMAJI[test_labels[idx]]
    pred_lbl = ROMAJI[preds[idx]]
    conf = confidences[idx] * 100.0
    ax.set_title(
        f"True: {true_lbl} | Pred: {pred_lbl}\n({conf:.1f}% conf)",
        fontsize=9.5,
        color="#b22222",
        fontweight="bold",
    )
    ax.axis("off")

fig.suptitle(
    "Top 10 Highest-Confidence Misclassifications (Test Set)",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("error_gallery.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved error_gallery.png")


# ------------------------------------------------------------------
# Prediction Confidence Histogram (Correct vs. Incorrect)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
bins = np.linspace(0.0, 1.0, 26)

ax.hist(
    confidences[correct_mask],
    bins=bins,
    density=True,
    alpha=0.65,
    color="#2ca02c",
    edgecolor="white",
    label=f"Correct (n={np.sum(correct_mask)})",
)
ax.hist(
    confidences[~correct_mask],
    bins=bins,
    density=True,
    alpha=0.65,
    color="#d62728",
    edgecolor="white",
    label=f"Incorrect (n={np.sum(~correct_mask)})",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Winning Softmax Probability (Confidence)", fontsize=11, fontweight="bold")
ax.set_ylabel("Density", fontsize=11, fontweight="bold")
ax.set_title(
    "Model Calibration: Confidence on Correct vs. Incorrect Predictions",
    fontsize=13,
    fontweight="bold",
)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(frameon=False, fontsize=10.5)

plt.tight_layout()
plt.savefig("confidence_histogram.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved confidence_histogram.png")


# ------------------------------------------------------------------
# 2D PCA Projection of Hidden Layer Activations (A1)
# ------------------------------------------------------------------
# Subsample 2,000 test images for a clean, readable scatter plot
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(test_labels), size=2000, replace=False)

H = A1[:, sample_idx].T  # (2000, 128)
H_centered = H - np.mean(H, axis=0, keepdims=True)
_, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
H_2d = H_centered @ Vt[:2].T

var_explained = (S[:2] ** 2) / np.sum(S ** 2) * 100.0

fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
cmap = plt.get_cmap("tab10")

for cls in range(10):
    mask = test_labels[sample_idx] == cls
    ax.scatter(
        H_2d[mask, 0],
        H_2d[mask, 1],
        s=14,
        alpha=0.65,
        color=cmap(cls),
        label=ROMAJI[cls],
        edgecolors="none",
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel(f"PC 1 ({var_explained[0]:.1f}% variance)", fontsize=11, fontweight="bold")
ax.set_ylabel(f"PC 2 ({var_explained[1]:.1f}% variance)", fontsize=11, fontweight="bold")
ax.set_title(
    "2D PCA of 128-D Hidden Layer Representations (A1)",
    fontsize=13,
    fontweight="bold",
)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(title="Character", frameon=False, markerscale=1.8, bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig("pca_hidden_layer.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved pca_hidden_layer.png")
