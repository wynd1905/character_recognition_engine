import ctypes
import os

# Define the Python equivalent of the C struct
class Matrix(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_double))
    ]

# Load the compiled C shared library
lib_path = os.path.abspath("./libmatrix.so")
c_math = ctypes.CDLL(lib_path)

# M_create: Takes two ints, returns a Matrix struct by value
c_math.M_create.argtypes = [ctypes.c_int, ctypes.c_int]
c_math.M_create.restype = Matrix

# M_destroy: Takes a pointer to a Matrix, returns nothing (void)
c_math.M_destroy.argtypes = [ctypes.POINTER(Matrix)]
c_math.M_destroy.restype = None

# M_Multiplication: Takes two Matrix structs by value, returns a Matrix struct by value
c_math.M_Multiplication.argtypes = [Matrix, Matrix]
c_math.M_Multiplication.restype = Matrix

c_math.M_Transpose.argtypes = [Matrix]
c_math.M_Transpose.restype = Matrix

c_math.M_Add.argtypes = [Matrix, Matrix]
c_math.M_Add.restype = Matrix
# M_ReLU: Takes a pointer to a Matrix, returns nothing (void)
c_math.M_ReLU.argtypes = [ctypes.POINTER(Matrix)]
c_math.M_ReLU.restype = None

# NEED TO IMPLEMENT SOME WRAPPERS

# Convert list to C matrix/pointer
def convert_to_c_mat(py_2d_list):
    rows = len(py_2d_list);
    cols = len(py_2d_list[0])

    c_mat = c_math.M_create(rows,cols)
    # Convert python list to C pointer
    for row in range(rows):
        for col in range(cols):
            c_mat.data[(row * cols) + col] = py_2d_list[row][col];


    return c_mat

# Print result to screen
def show_matrix(c_mat, name="Matrix"):
    if not c_mat.data:
        print(f"{name} is empty")
        return

    print(f"({c_mat.rows} by {c_mat.cols}) {name}:\n")
    for row in range(c_mat.rows):
        row_vals = [c_mat.data[(row * c_mat.cols) + col] for col in range(c_mat.cols)]
        print("  [" + ", ".join(f"{val:6.2f}" for val in row_vals) + "]")
    print()




