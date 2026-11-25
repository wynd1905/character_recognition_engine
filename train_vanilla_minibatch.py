import ctypes
import random
import time
import numpy as np
import neural_network_engine as nn
from network import KanaNetwork

random.seed(42)
np.random.seed(42)

# Load & normalize KMNIST
train_imgs = np.load("data/kmnist-train-imgs.npz")["arr_0"]
train_labels = np.load("data/kmnist-train-labels.npz")["arr_0"]
test_imgs = np.load("data/kmnist-test-imgs.npz")["arr_0"]
test_labels = np.load("data/kmnist-test-labels.npz")["arr_0"]

X_train = train_imgs.reshape(-1, 784).astype(np.float64) / 255.0
X_test = test_imgs.reshape(-1, 784).astype(np.float64) / 255.0


def evaluate(net, images, labels):
    correct = 0
    for i in range(len(images)):
        X = nn.numpy_to_c_matrix(images[i], 784, 1)
        A1, A2 = net.forward(X)
        probs = [A2.data[j] for j in range(10)]
        if probs.index(max(probs)) == labels[i]:
            correct += 1
        nn.c_math.M_destroy(ctypes.byref(X))
        nn.c_math.M_destroy(ctypes.byref(A1))
        nn.c_math.M_destroy(ctypes.byref(A2))
    return (correct / len(images)) * 100.0


net = KanaNetwork(input_size=784, hidden_size=128, output_size=10)
epochs = 10
batch_size = 64
lr = 0.1
num_samples = len(X_train)

history_loss, history_acc, history_time = [], [], []

for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0.0
    num_batches = 0
    indices = np.random.permutation(num_samples)

    for start in range(0, num_samples, batch_size):
        batch_idx = indices[start : start + batch_size]
        current_batch = len(batch_idx)

        X_batch = X_train[batch_idx].T
        Y_batch = np.zeros((10, current_batch), dtype=np.float64)
        Y_batch[train_labels[batch_idx], np.arange(current_batch)] = 1.0

        X = nn.numpy_to_c_matrix(X_batch, 784, current_batch)
        Y = nn.numpy_to_c_matrix(Y_batch, 10, current_batch)

        total_loss += net.train_step(X, Y, lr=lr, beta=0.0, weight_decay=0.0)
        num_batches += 1

        nn.c_math.M_destroy(ctypes.byref(X))
        nn.c_math.M_destroy(ctypes.byref(Y))

    elapsed = time.time() - start_time
    avg_loss = total_loss / num_batches
    test_acc = evaluate(net, X_test, test_labels)

    history_loss.append(avg_loss)
    history_acc.append(test_acc)
    history_time.append(elapsed)

    print(
        f"[Vanilla MB] Epoch {epoch + 1:2d}/{epochs} | {elapsed:.2f}s | Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}%"
    )

np.savez(
    "history_vanilla_minibatch.npz",
    loss=history_loss,
    acc=history_acc,
    times=history_time,
)
print("Saved history_vanilla_minibatch.npz")
