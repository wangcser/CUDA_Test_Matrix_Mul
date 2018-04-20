# Image Convolution with CUDA

## 00 environment set up
if you just want to use numba to accelerate your code, you can just install the numba pkgs.
But if you want to use high-api with numba in cuda, you need to install the cudatoolkit through
        
    conda install cudatoolkit

then you can check you env. by 
   
    from numba import cuda
   
    cuda.is_available() # return the GPU available for you.

## 01 Motivation

## 02 Standard Implementation of Convolution - std

## 03 A Naïve Parallel Implementation with CUDA - Parallel

## 04 Efficiency Optimizing with CUDA - Faster Parallel

## 05 Performance analysis 

## 06 Conclusion

## 07 Reference

