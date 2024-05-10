import numpy as np
import scipy.linalg
from main import solve_using_inverse, solve_using_lu_gauss, solve_using_crout


# Kode testing
def test():
    matrix_A = np.array([[2, 3, -1], [4, 4, -3], [-2, 3, -1]])
    vector_b = np.array([5, 3, 1])
    
    # Testing metode matriks balikan
    x_inverse = solve_using_inverse(matrix_A, vector_b)
    print("Solusi menggunakan metode matriks balikan:", x_inverse)

    # Testing metode dekomposisi LU Gauss
    x_lu_gauss = solve_using_lu_gauss(matrix_A, vector_b)
    print("Solusi menggunakan metode dekomposisi LU Gauss:", x_lu_gauss)

    # Testing metode dekomposisi Crout
    x_crout = solve_using_crout(matrix_A, vector_b)
    print("Solusi menggunakan metode dekomposisi Crout:", x_crout)

if __name__ == "__main__":
    test()
