#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
  int rows;
  int cols;
  double *data;
} Matrix;

// Memory Management
Matrix M_create(int rows, int cols);
void M_destroy(Matrix *M);

// Matrix operation
Matrix M_Add(Matrix A, Matrix B);
Matrix M_Subtract(Matrix A, Matrix B);
Matrix M_Transpose(Matrix M);
Matrix M_Multiplication(Matrix A, Matrix B);
Matrix M_Inverse(Matrix M);
Matrix M_ScalarMultiply(Matrix M, double scalar);
Matrix M_Hadamard(Matrix A, Matrix B); // Element-wise multiplication

// Activations (Modifies the matrix in-place to save memory)
void M_ReLU(Matrix *M);
void M_ReLU_Derivative(Matrix *M);

#endif
