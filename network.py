import ctypes
import random

import numpy as np

import neural_network_engine as nn


class KanaNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        print("Initializing network weights: ")

        # Generate random small weight (-0.1 < w < 0.1)
        w1_data = [
            [random.uniform(-0.1, 0.1) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        w2_data = [
            [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
            for _ in range(output_size)
        ]

        # Initialize biases
        b1_data = [[0.0] for _ in range(hidden_size)]
        b2_data = [[0.0] for _ in range(output_size)]

        # Convert to C matrix
        self.W1 = nn.convert_to_c_mat(w1_data)
        self.W2 = nn.convert_to_c_mat(w2_data)
        self.B1 = nn.convert_to_c_mat(b1_data)
        self.B2 = nn.convert_to_c_mat(b2_data)

        # Velocity matrices for momentum (calloc initializes all entries to 0.0)
        self.V_W1 = nn.c_math.M_create(hidden_size, input_size)
        self.V_B1 = nn.c_math.M_create(hidden_size, 1)
        self.V_W2 = nn.c_math.M_create(output_size, hidden_size)
        self.V_B2 = nn.c_math.M_create(output_size, 1)

        print("Initialized and converted to C matrix")

    # Free up memory
    def __del__(self):
        for mat in (
            self.W1,
            self.B1,
            self.W2,
            self.B2,
            self.V_W1,
            self.V_B1,
            self.V_W2,
            self.V_B2,
        ):
            nn.c_math.M_destroy(ctypes.byref(mat))

    def forward(self, X):
        # Layer 1: Z1 = W1 * X + B1 (in-place bias + ReLU)
        A1 = nn.c_math.M_Multiplication(self.W1, X)
        nn.c_math.M_AddBias(ctypes.byref(A1), self.B1)
        nn.c_math.M_ReLU(ctypes.byref(A1))

        # Layer 2: Z2 = W2 * A1 + B2 (in-place bias + Softmax)
        A2 = nn.c_math.M_Multiplication(self.W2, A1)
        nn.c_math.M_AddBias(ctypes.byref(A2), self.B2)
        nn.c_math.M_Softmax(ctypes.byref(A2))

        return A1, A2

    def backward(self, X, Y, A1, A2, lr, beta, weight_decay):
        batch_size = X.cols

        # Output layer error & bias gradient
        dZ2 = nn.c_math.M_Subtract(A2, Y)
        dB2 = nn.c_math.M_SumCols(dZ2)

        # Layer 2 weight gradient: dW2 = dZ2 * A1^T
        A1_T = nn.c_math.M_Transpose(A1)
        dW2 = nn.c_math.M_Multiplication(dZ2, A1_T)

        # Hidden layer error & bias gradient
        W2_T = nn.c_math.M_Transpose(self.W2)
        hidden_err = nn.c_math.M_Multiplication(W2_T, dZ2)
        nn.c_math.M_ReLU_Derivative(ctypes.byref(A1))
        dZ1 = nn.c_math.M_Hadamard(hidden_err, A1)
        dB1 = nn.c_math.M_SumCols(dZ1)

        # Layer 1 weight gradient: dW1 = dZ1 * X^T
        X_T = nn.c_math.M_Transpose(X)
        dW1 = nn.c_math.M_Multiplication(dZ1, X_T)

        # Gradient Descent (averaged via scaled_lr)
        nn.c_math.M_UpdateWeightsMomentum(
            ctypes.byref(self.W2),
            ctypes.byref(self.V_W2),
            dW2,
            lr,
            beta,
            weight_decay,
            batch_size,
        )
        nn.c_math.M_UpdateWeightsMomentum(
            ctypes.byref(self.B2),
            ctypes.byref(self.V_B2),
            dB2,
            lr,
            beta,
            0.0,
            batch_size,
        )
        nn.c_math.M_UpdateWeightsMomentum(
            ctypes.byref(self.W1),
            ctypes.byref(self.V_W1),
            dW1,
            lr,
            beta,
            weight_decay,
            batch_size,
        )
        nn.c_math.M_UpdateWeightsMomentum(
            ctypes.byref(self.B1),
            ctypes.byref(self.V_B1),
            dB1,
            lr,
            beta,
            0.0,
            batch_size,
        )

        # Destroy all temporary matrices (including dB1 and dB2)
        for temp in (dZ2, dB2, A1_T, dW2, W2_T, hidden_err, dZ1, dB1, X_T, dW1):
            nn.c_math.M_destroy(ctypes.byref(temp))

    def train_step(self, X, Y, lr=0.01, beta=0.9, weight_decay=1e-4):
        # Forward pass
        A1, A2 = self.forward(X)

        # Compute loss before A2 is freed
        loss = nn.c_math.M_CrossEntropy(A2, Y)

        # Backward pass and weight updates
        self.backward(X, Y, A1, A2, lr, beta, weight_decay)

        # Destroy the activations returned by forward()
        nn.c_math.M_destroy(ctypes.byref(A1))
        nn.c_math.M_destroy(ctypes.byref(A2))

        return loss

    def save_weights(self, filepath="kana_weights.npz"):
        np.savez(
            filepath,
            W1=nn.c_matrix_to_numpy(self.W1),
            B1=nn.c_matrix_to_numpy(self.B1),
            W2=nn.c_matrix_to_numpy(self.W2),
            B2=nn.c_matrix_to_numpy(self.B2),
        )
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath="kana_weights.npz"):
        data = np.load(filepath)

        # Free the existing C matrices before overwriting them
        for mat in (self.W1, self.B1, self.W2, self.B2):
            nn.c_math.M_destroy(ctypes.byref(mat))

        # Load the saved arrays directly into fresh C matrices
        self.W1 = nn.numpy_to_c_matrix(data["W1"], *data["W1"].shape)
        self.B1 = nn.numpy_to_c_matrix(data["B1"], *data["B1"].shape)
        self.W2 = nn.numpy_to_c_matrix(data["W2"], *data["W2"].shape)
        self.B2 = nn.numpy_to_c_matrix(data["B2"], *data["B2"].shape)
        print(f"Weights loaded from {filepath}")
