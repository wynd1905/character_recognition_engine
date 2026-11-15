import neural_network_engine as nn
import ctypes

# 1. Define predictable test data
A = nn.convert_to_c_mat([
    [1.0, 2.0],
    [3.0, 4.0]
])

B = nn.convert_to_c_mat([
    [2.0, 0.0],
    [1.0, 2.0]
])

# 2. Test M_Multiplication
# Expected output: [[4, 4], [10, 8]]
C = nn.c_math.M_Multiplication(A, B)
nn.show_matrix(C, "A * B")

# 3. Test M_Transpose
# Expected output: [[4, 10], [4, 8]]
C_T = nn.c_math.M_Transpose(C)
nn.show_matrix(C_T, "Transpose of (A * B)")

# 4. Test In-Place M_ReLU with negatives
D = nn.convert_to_c_mat([
    [-1.0, 5.0],
    [ 0.0, -3.2]
])
nn.c_math.M_ReLU(ctypes.byref(D))
nn.show_matrix(D, "ReLU of D (Expected all negatives to be 0)")

# 5. Prevent Memory Leaks
nn.c_math.M_destroy(ctypes.byref(A))
nn.c_math.M_destroy(ctypes.byref(B))
nn.c_math.M_destroy(ctypes.byref(C))
nn.c_math.M_destroy(ctypes.byref(C_T))
nn.c_math.M_destroy(ctypes.byref(D))
