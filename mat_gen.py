import numpy as np


def gene_matrix(row, col, rand=True, show=False):
    if rand:
        # matrix = np.random.randint(0, 10, (row, col))
        matrix = np.array(np.random.random((row, col)), dtype=np.float32)
    else:
        matrix = np.array(np.zeros((row, col)), dtype=np.float32)

    if show:
        print(matrix)

    return matrix
