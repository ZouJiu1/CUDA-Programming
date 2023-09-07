// nvcc -arch=compute_60 -code=sm_60  hello2.cu
#pragma once
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<1, 1>>>();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());         
   //同步host和device，核函数执行完以前，不会执行后续内容
    cudaDeviceSynchronize();         
   //同步host和device，核函数执行完以前，不会执行后续内容
    return 0;
}

