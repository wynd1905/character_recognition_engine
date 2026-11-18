import ctypes
import time

import numpy as np

import neural_network_engine as nn
from network import KanaNetwork

# 1. Load KMNIST arrays
train_imgs = np.load("data/kmnist-train-imgs.npz")["arr_0"]  # (60000, 28, 28)
train_labels = np.load("data/kmnist-train-labels.npz")["arr_0"]  # (60000,)
test_imgs = np.load("data/kmnist-test-imgs.npz")["arr_0"]  # (10000, 28, 28)
test_labels = np.load("data/kmnist-test-labels.npz")["arr_0"]  # (10000,)

# 2. Flatten (28x28 -> 784) and normalize to [0.0, 1.0]
X_train = train_imgs.reshape(-1, 784).astype(np.float64) / 255.0
X_test = test_imgs.reshape(-1, 784).astype(np.float64) / 255.0


def one_hot(label, num_classes=10):
    vec = np.zeros(num_classes, dtype=np.float64)
    vec[label] = 1.0
    return vec


def evaluate(net, images, labels):
    correct = 0
    for i in range(len(images)):
        X = nn.numpy_to_c_matrix(images[i], 784, 1)
        A1, A2 = net.forward(X)

        # Find index of highest probability in A2 (argmax)
        probs = [A2.data[j] for j in range(10)]
        pred = probs.index(max(probs))
        if pred == labels[i]:
            correct += 1

        nn.c_math.M_destroy(ctypes.byref(X))
        nn.c_math.M_destroy(ctypes.byref(A1))
        nn.c_math.M_destroy(ctypes.byref(A2))

    return (correct / len(images)) * 100.0


# 3. Initialize network and hyperparameters
net = KanaNetwork(input_size=784, hidden_size=128, output_size=10)
epochs = 10
batch_size = 64
lr = 0.1
num_samples = len(X_train)
best_acc = 0.0
history_loss = []
history_acc = []

# 4. Main Training Loop (Stochastic Gradient Descent)
for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0.0
    num_batches = 0

    indices = np.random.permutation(num_samples)

    for start in range(0, num_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        current_batch = len(batch_idx)

        # Transpose to (784, current_batch) so each column is one image
        X_batch = X_train[batch_idx].T

        # Build one-hot matrix of shape (10, current_batch)
        Y_batch = np.zeros((10, current_batch), dtype=np.float64)
        Y_batch[train_labels[batch_idx], np.arange(current_batch)] = 1.0

        X = nn.numpy_to_c_matrix(X_batch, 784, current_batch)
        Y = nn.numpy_to_c_matrix(Y_batch, 10, current_batch)

        total_loss += net.train_step(X, Y, lr=lr)
        num_batches += 1

        nn.c_math.M_destroy(ctypes.byref(X))
        nn.c_math.M_destroy(ctypes.byref(Y))

    elapsed = time.time() - start_time
    avg_loss = total_loss / num_batches
    test_acc = evaluate(net, X_test, test_labels)

    # Record metrics for plotting
    history_loss.append(avg_loss)
    history_acc.append(test_acc)

    print(
        f"Epoch {epoch + 1:2d}/{epochs} | {elapsed:.2f}s | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%"
    )

    # Only save when test accuracy improves
    if test_acc > best_acc:
        best_acc = test_acc
        net.save_weights("kana_weights.npz")

    # Decay learning rate so weights settle into the minimum
    lr *= 0.85

np.savez("training_history.npz", loss=history_loss, acc=history_acc)
print(
    f"Training complete. Best Test Accuracy: {best_acc:.2f}% | Saved training_history.npz"
)
