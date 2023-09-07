// nvcc -arch=compute_86 -code=sm_86  --ptxas-options=-v --maxrregcount=20 hello3.cu
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();         
   //同步host和device，核函数执行完以前，不会执行后续内容
    return 0;
}
/*
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
Hello World from the GPU!
*/
