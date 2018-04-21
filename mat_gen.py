import numpy as np


def gene_matrix(row, col, rand=True, show=False):
    """
    func: 给出矩阵的行和列大小，输出指定类型的矩阵并初始化
    args: rand 对矩阵随机初始化，否则初始化为零矩阵
          show 输出矩阵信息，用于 DEBUG
    """
    if rand:
        matrix = np.array(np.random.random((row, col)), dtype=np.float32)
    else:
        matrix = np.array(np.zeros((row, col)), dtype=np.float32)

    if show:
        print(matrix)

    return matrix
