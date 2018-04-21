"""
# this file defined 6 types of matrix multiple implement, in practice
# only use 4 of them.
"""
from numba import cuda, jit, float32


def matmul(A, B, C):
    """
    host: cpu with 1 thread.
    func: use naive element-wise mul cal the result.
    """
    m, p = A.shape
    n = B.shape[1]
    for i in range(0, m):
        for j in range(0, n):
            for k in range(0, p):
                C[i, j] += A[i, k] * B[k, j]


@cuda.jit
def faster_matmul(A, B, C):
    """
    host: cpu with 1 thread, compile optimise.
    func: use naive element-wise mul cal the result.
    """
    m, p = A.shape
    n = B.shape[1]
    for i in range(0, m):
        for j in range(0, n):
            for k in range(0, p):
                C[i, j] += A[i, k] * B[k, j]


@jit(nogil=True, parallel=True)
def faster_matmul_parallel(A, B, C):
    """
    host: cpu with 6 thread, compile optimise.
    func: use naive element-wise mul cal the result.
    """
    m, p = A.shape
    n = B.shape[1]
    for i in range(0, m):
        for j in range(0, n):
            for k in range(0, p):
                C[i, j] += A[i, k] * B[k, j]


@cuda.jit
def cuda_matmul(A, B, C):
    """
    host: gpu
    func: Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
            C[i, j] = tmp


@cuda.jit
def cuda_matmul_2(A, B, C):
    """
    host: gpu
    func: each thread cal a result in C[i,j] with element-wise mul.
    """
    # get the local thread, block and grid index
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    # get the global index for thread
    x = tx + bx * bw
    y = ty + by * bh

    # cal the size of square matrix
    row, col = A.shape
    mat_size = row * col

    # check the block boundary
    if x >= mat_size or y >= mat_size:
        return

    for i in range(mat_size):
        C[y, x] += A[y, i] * B[i, x]


@cuda.jit('(float32[:,:], float32[:,:], float32[:,:])')
def faster_cuda_matmul(A, B, C):
    """
    host: gpu
    func: use the share memory to accelerate the speed.
    detail: Controls threads per block and shared memory usage.
            The computation will be done on blocks of TPBxTPB elements.
            Define an array in the shared memory
            The size and type of the arrays must be known at compile time
    """
    # define: thread_per_block, when modified in analysis, you must modify this TPB
    TPB = 16
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    # get the global thread index
    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # block per grid
    bpg = cuda.gridDim.x

    if x >= C.shape[0] and y >= C.shape[1]:
        # quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()
        C[x, y] = tmp
