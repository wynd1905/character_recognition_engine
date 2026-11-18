import ctypes
import random
import neural_network_engine as nn

class KanaNetwork:
    def __init__(self, input_size = 784, hidden_size = 128, output_size = 10):
        print("Initializing network weights: ")

        # Generate random small weight (-0.1 < w < 0.1)
        w1_data = [[random.uniform(-0.1,0.1) for _ in range(input_size)] for _ in range(hidden_size)]
        w2_data = [[random.uniform(-0.1,0.1) for _ in range(hidden_size)] for _ in range(output_size)]

        # Initialize biases
        b1_data = [[0.0] for _ in range(hidden_size)]
        b2_data = [[0.0] for _ in range(output_size)]

        # Convert to C matrix
        self.W1 = nn.convert_to_c_mat(w1_data)
        self.W2 = nn.convert_to_c_mat(w2_data)
        self.B1 = nn.convert_to_c_mat(b1_data)
        self.B2 = nn.convert_to_c_mat(b2_data)

        print("Initialized and converted to C matrix")

    # Free up memory
    def __del__(self):
        nn.c_math.M_destroy(ctypes.byref(self.W1))
        nn.c_math.M_destroy(ctypes.byref(self.W2))
        nn.c_math.M_destroy(ctypes.byref(self.B1))
        nn.c_math.M_destroy(ctypes.byref(self.B2))
