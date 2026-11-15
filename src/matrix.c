#include "../include/matrix.h"
#include <stdio.h>
#include <stdlib.h>

// Matrix Constructor
Matrix M_create(int rows, int cols) {
  Matrix m1 = {rows, cols};
  m1.data = calloc(rows * cols, sizeof(double));
  if (m1.data == NULL) {
    fprintf(stderr, "Cannot allocate memory for data\n");
    return (Matrix){0, 0, NULL};
  }
  return m1;
}

// Matrix Destructor
void M_destroy(Matrix *M) {
  free(M->data);
  M->data = NULL;
  M->rows = 0;
  M->cols = 0;
}

// Matrix Addition
Matrix M_Add(Matrix A, Matrix B) {
  if (A.rows != B.rows || A.cols != B.cols) {
    fprintf(stderr, "Not same matrix size\n");
    return (Matrix){0, 0, NULL};
  }
  Matrix C = M_create(A.rows, A.cols);
  for (int i = 0; i < C.rows * C.cols; i++) {
    C.data[i] = A.data[i] + B.data[i];
  }
  return C;
}

// Matrix substraction
Matrix M_Subtract(Matrix A, Matrix B) {
  if (A.rows != B.rows || A.cols != B.cols) {
    fprintf(stderr, "Not same matrix size\n");
    return (Matrix){0, 0, NULL};
  }
  Matrix C = M_create(A.rows, A.cols);
  for (int i = 0; i < C.rows * C.cols; i++) {
    C.data[i] = A.data[i] - B.data[i];
  }
  return C;
}

// Matrix element wise multiplication
Matrix M_Hadamard(Matrix A, Matrix B) {
  if (A.rows != B.rows || A.cols != B.cols) {
    fprintf(stderr, "Not same matrix size\n");
    return (Matrix){0, 0, NULL};
  }
  Matrix C = M_create(A.rows, A.cols);
  for (int i = 0; i < C.rows * C.cols; i++) {
    C.data[i] = A.data[i] * B.data[i];
  }
  return C;
}

// Matrix scalar multiplication
Matrix M_ScalarMultiply(Matrix M, double scalar) {
  if (M.data == NULL) {
    fprintf(stderr, "Source matrix is null in M_ScalarMultiply\n");
    return (Matrix){0, 0, NULL};
  }

  Matrix C = M_create(M.rows, M.cols);
  // If run out if memory
  if (C.data == NULL) {
    return C;
  }
  for (int i = 0; i < C.rows * C.cols; i++) {
    C.data[i] = M.data[i] * scalar;
  }
  return C;
}

// Matrix trasnpose
Matrix M_Transpose(Matrix M) {
  if (M.data == NULL) {
    fprintf(stderr, "Empty matrix\n");
    return (Matrix){0, 0, NULL};
  }
  Matrix C = M_create(M.cols, M.rows);
  if (C.data == NULL) {
    return C;
  }
  int idx = 0;
  for (int row = 0; row < M.rows; row++) {
    for (int col = 0; col < M.cols; col++) {
      idx = (row * M.cols) + col;
      C.data[(col * M.rows) + row] = M.data[idx];
    }
  }
  return C;
}

// Matrix multiplication
Matrix M_Multiplication(Matrix A, Matrix B) {
  if (A.cols != B.rows) {
    fprintf(stderr, "Wrong matrix dimension\n");
    return (Matrix){0, 0, NULL};
  }
  Matrix C = M_create(A.rows, B.cols);
  if (C.data == NULL) {
    return C;
  }
  int idx = 0;
  for (int i = 0; i < A.rows; i++) {
    for (int k = 0; k < A.cols; k++) {
      double a_val = A.data[(i * A.cols) + k];
      for (int j = 0; j < B.cols; j++) {
        idx = (i * C.cols) + j;
        C.data[idx] += a_val * B.data[(k * B.cols) + j];
      }
    }
  }
  return C;
}

// Rectified Linear Unit
void M_ReLU(Matrix *M) {
  if (M == NULL || M->data == NULL) {
    return;
  }
  for (int i = 0; i < M->rows * M->cols; i++) {
    if (M->data[i] < 0) {
      M->data[i] = 0;
    }
  }
}

// Rectified Linear Unit derivative
void M_ReLU_Derivative(Matrix *M) {
  if (M == NULL || M->data == NULL) {
    return;
  }
  for (int i = 0; i < M->rows * M->cols; i++) {
    if (M->data[i] <= 0) {
      M->data[i] = 0;
    } else {
      M->data[i] = 1;
    }
  }
}
