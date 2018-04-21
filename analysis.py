"""
# for analysis propose, i designed all four algorithms running in the same RAM / GPU memory space.
# for the memory limit and i can't find the right way to release the GPU-memory in the document, so
# i need to choose the matrix size to a small scale.
"""
from numba import cuda
from mat_gen import gene_matrix
from mat_mul import matmul, faster_matmul, faster_matmul_parallel, cuda_matmul, faster_cuda_matmul
from timer import timer as t
import numpy as np


thread_per_block = 16
block_per_grid = 32

if __name__ == "__main__":

    mat_size = block_per_grid * thread_per_block

    # init matrix
    A = gene_matrix(mat_size, mat_size, rand=True, show=False)
    B = gene_matrix(mat_size, mat_size, rand=True, show=False)
    C_1 = gene_matrix(mat_size, mat_size, rand=False, show=False)
    C_2 = gene_matrix(mat_size, mat_size, rand=False, show=False)
    C_3 = gene_matrix(mat_size, mat_size, rand=False, show=False)
    C_4 = gene_matrix(mat_size, mat_size, rand=False, show=False)

    print("matrix size : %d x %d" % (mat_size, mat_size))

    # init timer
    cpu_timer = t()
    faster_cpu_timer = t()
    gpu_timer = t()
    faster_gpu_timer = t()

    # CPU compute
    cpu_timer.start()
    matmul(A, B, C_1)
    t_cpu = cpu_timer.stop()

    # faster CPU compute
    faster_cpu_timer.start()
    faster_matmul_parallel(A, B, C_2)
    t_faster_cpu = faster_cpu_timer.stop()

    # GPU compute
    gpu_timer.start()
    stream = cuda.stream()
    with stream.auto_synchronize():
        # 将数据传入GPU
        dA = cuda.to_device(A, stream)
        dB = cuda.to_device(B, stream)
        dC_3 = cuda.to_device(C_3, stream)
        cuda_matmul[(block_per_grid, block_per_grid), (thread_per_block, thread_per_block), stream](dA, dB, dC_3)
        # 将结果取回CPU
        dC_3.to_host(stream)
    t_gpu = gpu_timer.stop()

    # faster GPU compute
    faster_gpu_timer.start()
    stream = cuda.stream()
    with stream.auto_synchronize():
        # 将数据传入GPU
        dA = cuda.to_device(A, stream)
        dB = cuda.to_device(B, stream)
        dC_4 = cuda.to_device(C_4, stream)
        faster_cuda_matmul[(block_per_grid, block_per_grid), (thread_per_block, thread_per_block), stream](dA, dB, dC_4)
        # 将结果取回CPU
        dC_4.to_host(stream)
    t_faster_gpu = faster_gpu_timer.stop()

    # Check result
    assert np.allclose(C_1, C_4)

    # output the result and cal the speedup over default cpu method.
    result = '''
        default cpu mul: {:f} s, speedup: {:.2f}x,
        faster  cpu mul: {:f} s, speedup: {:.2f}x,
        default gpu mul: {:f} s, speedup: {:.2f}x,
        faster  gpu mul: {:f} s, speedup: {:.2f}x,
    
    '''.format(
        t_cpu, t_cpu / t_cpu,
        t_faster_cpu, t_cpu / t_faster_cpu,
        t_gpu, t_cpu / t_gpu,
        t_faster_gpu, t_cpu / t_faster_gpu,
    )
    print(result)

