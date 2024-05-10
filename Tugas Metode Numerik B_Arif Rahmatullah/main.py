import numpy as np
import scipy.linalg

# Metode matriks balikan
def solve_using_inverse(matrix_A, vector_b):
    A_inv = np.linalg.inv(matrix_A)
    x = np.dot(A_inv, vector_b)
    return x

# Metode dekomposisi LU Gauss
def solve_using_lu_gauss(matrix_A, vector_b):
    P, L, U = scipy.linalg.lu(matrix_A)
    y = np.linalg.solve(L, np.dot(P, vector_b))
    x = np.linalg.solve(U, y)
    return x

# Metode dekomposisi Crout
def solve_using_crout(matrix_A, vector_b):
    LU, piv = scipy.linalg.lu_factor(matrix_A)
    L = np.tril(LU, k=-1) + np.eye(len(matrix_A))
    U = np.triu(LU)
    y = np.linalg.solve(L, vector_b)
    x = np.linalg.solve(U, y)
    return x

