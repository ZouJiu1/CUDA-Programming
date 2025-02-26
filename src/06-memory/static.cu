/*
nvcc -O3 -arch=compute_86 -code=sm_86  --ptxas-options=-v --maxrregcount=20 static.cu && ./a.out
d_x = 1, d_y[0] = 11, d_y[1] = 21.
h_y[0] = 11, h_y[1] = 21.
*/
#include <cuda_runtime.h>
#include <cuda.h>

#include "error.cuh"
#include <stdio.h>
__device__ int d_x = 1;
__device__ int d_y[2];

void __global__ my_kernel(void)
{
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}

int main(void)
{
    int h_y[2] = {10, 20};
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));     // 复制host的内容到device
    
    my_kernel<<<1, 1>>>();         //调用核函数，给定执行配置的<<<1, 1>>>
    CHECK(cudaDeviceSynchronize());                          //同步host和device，核函数执行完以前，不会执行后续内容
    
    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));   // 复制device的内容到host
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);
    
    return 0;
}

