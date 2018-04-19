"""
input: A is m*p matrix, B is p*n matrix.
return: C is m*n matrix.
"""
import numpy as np
from matrix import gene_matrix
from timer import timer
from numba import jit

@jit
def matrix_mul(A, B):
    # get the dim info from A and B.
    m, p = A.shape
    n = B.shape[1]

    # init C with 0-matrix.
    C = np.zeros((m, n))

    for i in range(0, m):
        for j in range(0, n):
            for k in range(0, p):
                C[i, j] += A[i, k] * B[k, j]
    return C

if __name__ == "__main__":

    A = gene_matrix(row=1024, col=1024, show=False)
    B = gene_matrix(row=1024, col=1024)
    # A = np.random.randint(0, 10, (5, 3))
    # B = np.random.randint(0, 10, (3, 4))

    timer = timer()
    print("running...")
    timer.start()
    C = matrix_mul(A, B)
    timer.stop()

    print(C)
