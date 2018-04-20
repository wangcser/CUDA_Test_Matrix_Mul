"""
input: A is m*p matrix, B is p*n matrix.
return: C is m*n matrix.
"""
from matrix import gene_matrix
from timer import timer
from CUDA_kernel_matrix_mul import naive_mul, matmul, fast_matmul


if __name__ == "__main__":

    A = gene_matrix(row=100, col=100, show=False)
    B = gene_matrix(row=100, col=100, show=False)
    C = gene_matrix(row=100, col=100, rand=False, show=False)
    # A = np.random.randint(0, 10, (5, 3))
    # B = np.random.randint(0, 10, (3, 4))

    timer = timer()
    print("running...")
    timer.start()
    #naive_mul(A, B, C)
    matmul(A, B, C)
    # C = fast_matmul(A, B, C)
    timer.stop()

    print(C)
