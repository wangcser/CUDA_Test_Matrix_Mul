import numpy as np


def gene_matrix(row, col, rand=True, show=False):
    if rand:
        matrix = np.random.randint(0, 10, (row, col))
    else:
        matrix = np.zeros([row, col])

    if show:
        print(matrix)

    return matrix
