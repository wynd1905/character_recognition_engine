import ctypes
import random
import sys

import numpy as np

import neural_network_engine as nn
from network import KanaNetwork

# Official KMNIST-10 class mapping (index 0-9)
KMNIST_CLASSES = [
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


def render_ascii(img_28x28):
    """Print a 28x28 grayscale image directly in the terminal."""
    chars = " .:-=+*#%@"
    for row in img_28x28:
        line = "".join(chars[int(pixel / 256 * len(chars))] for pixel in row)
        print(line)


def predict_single(net, img_28x28):
    # Flatten and normalize to [0.0, 1.0]
    flat_img = img_28x28.reshape(784).astype(np.float64) / 255.0
    X = nn.numpy_to_c_matrix(flat_img, 784, 1)

    # Run forward pass in C
    A1, A2 = net.forward(X)

    # Read the 10 Softmax probabilities back into Python
    probs = [A2.data[i] for i in range(10)]
    pred_idx = probs.index(max(probs))
    confidence = probs[pred_idx] * 100.0

    # Clean up C memory immediately
    nn.c_math.M_destroy(ctypes.byref(X))
    nn.c_math.M_destroy(ctypes.byref(A1))
    nn.c_math.M_destroy(ctypes.byref(A2))

    return pred_idx, confidence, probs


if __name__ == "__main__":
    # 1. Load test dataset
    test_imgs = np.load("data/kmnist-test-imgs.npz")["arr_0"]
    test_labels = np.load("data/kmnist-test-labels.npz")["arr_0"]

    # 2. Initialize network and load trained C weights
    net = KanaNetwork(input_size=784, hidden_size=128, output_size=10)
    net.load_weights("kana_weights.npz")

    # 3. Pick a specific index via CLI argument, or a random one
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    else:
        idx = random.randint(0, len(test_imgs) - 1)

    img = test_imgs[idx]
    true_idx = int(test_labels[idx])

    # 4. Run prediction and display results
    pred_idx, confidence, probs = predict_single(net, img)

    print(f"\n--- Test Image #{idx} ---")
    render_ascii(img)
    print(f"\nPredicted : {KMNIST_CLASSES[pred_idx]} ({confidence:.2f}% confidence)")
    print(f"Actual    : {KMNIST_CLASSES[true_idx]}")
    print("Result    :", "CORRECT" if pred_idx == true_idx else "INCORRECT")
