import ctypes

import numpy as np

import neural_network_engine as nn
from network import KanaNetwork

# Romaji for clean terminal column alignment, Hiragana for the summary
ROMAJI = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]
KANA = [
    "お (o)",
    "き (ki)",
    "す (su)",
    "つ (tsu)",
    "な (na)",
    "は (ha)",
    "ま (ma)",
    "や (ya)",
    "れ (re)",
    "を (wo)",
]


def compute_confusion_matrix(net, X_test, y_test, batch_size=256):
    cm = np.zeros((10, 10), dtype=np.int64)
    num_samples = len(X_test)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        current_batch = end - start

        # Transpose to (784, current_batch) for C matrix layout
        X_batch = X_test[start:end].T
        X = nn.numpy_to_c_matrix(X_batch, 784, current_batch)

        # Forward pass in C
        A1, A2 = net.forward(X)

        # Copy (10, current_batch) probabilities back to NumPy and take argmax
        probs = nn.c_matrix_to_numpy(A2)
        preds = np.argmax(probs, axis=0)

        # Increment cm[true_label, pred_label] for every sample in the batch
        np.add.at(cm, (y_test[start:end], preds), 1)

        # Free C memory immediately
        nn.c_math.M_destroy(ctypes.byref(X))
        nn.c_math.M_destroy(ctypes.byref(A1))
        nn.c_math.M_destroy(ctypes.byref(A2))

    return cm


def print_report(cm):
    # 1. Print formatted 10x10 Confusion Matrix (Rows = Actual, Cols = Predicted)
    print("\nConfusion Matrix (Rows = Actual, Cols = Predicted)\n")
    header = "      " + "".join(f"{name:>6}" for name in ROMAJI) + "  | Acc"
    print(header)
    print("-" * len(header))

    for i in range(10):
        row_total = cm[i].sum()
        row_acc = (cm[i, i] / row_total) * 100.0 if row_total > 0 else 0.0
        row_str = "".join(f"{cm[i, j]:>6}" for j in range(10))
        print(f"{ROMAJI[i]:>4} |{row_str}  | {row_acc:5.1f}%")

    overall_acc = (np.trace(cm) / cm.sum()) * 100.0
    print("-" * len(header))
    print(f"Overall Test Accuracy: {overall_acc:.2f}%\n")

    # 2. Rank the most common misclassifications (off-diagonal entries)
    errors = []
    for true_idx in range(10):
        for pred_idx in range(10):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                errors.append((cm[true_idx, pred_idx], true_idx, pred_idx))

    errors.sort(reverse=True)

    print("Top 10 Most Frequent Misclassifications:")
    for count, true_idx, pred_idx in errors[:10]:
        print(
            f"  {KANA[true_idx]:<10} misclassified as  {KANA[pred_idx]:<10} : {count:3d} times"
        )


if __name__ == "__main__":
    test_imgs = np.load("data/kmnist-test-imgs.npz")["arr_0"]
    test_labels = np.load("data/kmnist-test-labels.npz")["arr_0"]
    X_test = test_imgs.reshape(-1, 784).astype(np.float64) / 255.0

    net = KanaNetwork(input_size=784, hidden_size=128, output_size=10)
    net.load_weights("kana_weights.npz")

    cm = compute_confusion_matrix(net, X_test, test_labels)
    print_report(cm)
