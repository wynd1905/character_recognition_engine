#include "../include/matrix.h"
#include <math.h>
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
  if (A.data == NULL || B.data == NULL) {
    fprintf(stderr, "Source matrices is null\n");
    return (Matrix){0, 0, NULL};
  }
  if (A.rows != B.rows || A.cols != B.cols) {
    fprintf(stderr, "Not same matrix size\n");
    return (Matrix){0, 0, NULL};
  }
  Matrix C = M_create(A.rows, A.cols);
  if (C.data == NULL) {
    return C;
  }
  for (int i = 0; i < C.rows * C.cols; i++) {
    C.data[i] = A.data[i] + B.data[i];
  }
  return C;
}

// Matrix substraction
Matrix M_Subtract(Matrix A, Matrix B) {
  if (A.data == NULL || B.data == NULL) {
    fprintf(stderr, "Source matrices is null\n");
    return (Matrix){0, 0, NULL};
  }
  if (A.rows != B.rows || A.cols != B.cols) {
    fprintf(stderr, "Not same matrix size\n");
    return (Matrix){0, 0, NULL};
  }
  Matrix C = M_create(A.rows, A.cols);
  if (C.data == NULL) {
    return C;
  }
  for (int i = 0; i < C.rows * C.cols; i++) {
    C.data[i] = A.data[i] - B.data[i];
  }
  return C;
}

// Matrix element wise multiplication
Matrix M_Hadamard(Matrix A, Matrix B) {
  if (A.data == NULL || B.data == NULL) {
    fprintf(stderr, "Source matrices is null\n");
    return (Matrix){0, 0, NULL};
  }
  if (A.rows != B.rows || A.cols != B.cols) {
    fprintf(stderr, "Not same matrix size\n");
    return (Matrix){0, 0, NULL};
  }
  Matrix C = M_create(A.rows, A.cols);
  if (C.data == NULL) {
    return C;
  }
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

// Softmax calculate
void M_Softmax(Matrix *M) {
  if (M == NULL || M->data == NULL)
    return;

  for (int col = 0; col < M->cols; col++) {
    // Find the maximum value in this column for numerical stability
    double max_val = M->data[col]; // Row 0, current col
    for (int row = 1; row < M->rows; row++) {
      double val = M->data[(row * M->cols) + col];
      if (val > max_val) {
        max_val = val;
      }
    }

    // Compute exp(x - max_val) and accumulate the sum
    double sum = 0.0;
    for (int row = 0; row < M->rows; row++) {
      int idx = (row * M->cols) + col;
      M->data[idx] = exp(M->data[idx] - max_val);
      sum += M->data[idx];
    }

    // Normalize by dividing every element by the sum
    for (int row = 0; row < M->rows; row++) {
      int idx = (row * M->cols) + col;
      M->data[idx] /= sum;
    }
  }
}

// Calculate Cross Entropy
double M_CrossEntropy(Matrix pred, Matrix target) {
  if (pred.data == NULL || target.data == NULL || pred.rows != target.rows ||
      pred.cols != target.cols) {
    fprintf(stderr, "Invalid matrices in M_CrossEntropy\n");
    return -1.0;
  }

  double epsilon = 1e-15;
  double total_loss = 0.0;
  int total_elements = pred.rows * pred.cols;

  for (int i = 0; i < total_elements; i++) {
    // Only the true class (where target == 1.0) contributes to the loss
    if (target.data[i] > 0.0) {
      total_loss -= target.data[i] * log(pred.data[i] + epsilon);
    }
  }

  // Divide by number of columns (batch size) to get the mean loss
  return total_loss / pred.cols;
}

// Update weights
void M_UpdateWeights(Matrix *W, Matrix dW, double lr) {
  if (W == NULL || W->data == NULL || dW.data == NULL || W->rows != dW.rows ||
      W->cols != dW.cols) {
    fprintf(stderr, "Invalid matrices in M_UpdateWeights\n");
    return;
  }

  for (int i = 0; i < W->rows * W->cols; i++) {
    W->data[i] -= lr * dW.data[i];
  }
}

// Broadcasts a (rows x 1) bias vector B across every column of M (in-place)
void M_AddBias(Matrix *M, Matrix B) {
  if (M == NULL || M->data == NULL || B.data == NULL || M->rows != B.rows ||
      B.cols != 1) {
    fprintf(stderr, "Invalid dimensions in M_AddBias\n");
    return;
  }

  for (int row = 0; row < M->rows; row++) {
    double b_val = B.data[row];
    int row_offset = row * M->cols;
    for (int col = 0; col < M->cols; col++) {
      M->data[row_offset + col] += b_val;
    }
  }
}

// Sums a (rows x cols) matrix across all columns into a (rows x 1) vector
Matrix M_SumCols(Matrix M) {
  if (M.data == NULL) {
    fprintf(stderr, " NULL matrix in M_SumCols\n");
    return (Matrix){0, 0, NULL};
  }

  Matrix res = M_create(M.rows, 1);
  if (res.data == NULL)
    return res;

  for (int row = 0; row < M.rows; row++) {
    double sum = 0.0;
    int row_offset = row * M.cols;
    for (int col = 0; col < M.cols; col++) {
      sum += M.data[row_offset + col];
    }
    res.data[row] = sum;
  }

  return res;
}

void M_UpdateWeightsMomentum(Matrix *W, Matrix *V, Matrix dW, double lr,
                             double beta, double weight_decay, int batch_size) {
  if (W == NULL || W->data == NULL || V == NULL || V->data == NULL ||
      dW.data == NULL || W->rows != V->rows || W->cols != V->cols ||
      W->rows != dW.rows || W->cols != dW.cols || batch_size <= 0) {
    fprintf(stderr, "Invalid matrices in M_UpdateWeightsMomentum\n");
    return;
  }

  double inv_batch = 1.0 / (double)batch_size;
  int total = W->rows * W->cols;

  for (int i = 0; i < total; i++) {
    // Average the mini-batch gradient, then add the L2 penalty (lambda * W)
    double grad = (dW.data[i] * inv_batch) + (weight_decay * W->data[i]);
    V->data[i] = (beta * V->data[i]) - (lr * grad);
    W->data[i] += V->data[i];
  }
}
