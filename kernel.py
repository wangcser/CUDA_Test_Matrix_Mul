from numba import cuda
import numpy as np


@cuda.jit
def increment_by_one(array):
    """
       Increment all array elements by one.
    """
    theads_per_block = 32
    blocks_per_grid = (array.size + (theads_per_block - 1)) // theads_per_block
    increment_by_one[blocks_per_grid, theads_per_block](array)



if __name__ == "__main__":

    arr = np.arange(1000)
    d_arr = cuda.to_device(arr)

    increment_by_one[100, 100](d_arr)

    result_array = d_arr.copy_to_host()

    print(result_array)
