#include "error.cuh" 
#include <stdio.h>
#include <stdlib.h>
#include <cusolverDn.h>

int main(void)
{
    int N = 2;
    int N2 = N * N;

    cuDoubleComplex *A_cpu = (cuDoubleComplex *) 
        malloc(sizeof(cuDoubleComplex) * N2);    // 分配复数矩阵内存
    for (int n = 0; n < N2; ++n)  //对复数矩阵赋值
    {
        A_cpu[0].x = 0;
        A_cpu[1].x = 0;
        A_cpu[2].x = 0;
        A_cpu[3].x = 0;
        A_cpu[0].y = 0; 
        A_cpu[1].y = 1;
        A_cpu[2].y = -1;
        A_cpu[3].y = 0;
    }
    cuDoubleComplex *A;  // 定义device的矩阵
    CHECK(cudaMalloc((void**)&A, sizeof(cuDoubleComplex) * N2)); // 分配显存的
    CHECK(cudaMemcpy(A, A_cpu, sizeof(cuDoubleComplex) * N2,    // 数据传输的，host 到 device
        cudaMemcpyHostToDevice));

    double *W_cpu = (double*) malloc(sizeof(double) * N);        // 分配本征值的内存
    double *W; 
    CHECK(cudaMalloc((void**)&W, sizeof(double) * N));     //  分配本征值的显存

    cusolverDnHandle_t handle = NULL;  
    cusolverDnCreate(&handle);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    int lwork = 0;
// 确定运算需要多少缓冲空间
    cusolverDnZheevd_bufferSize(handle, jobz, uplo, 
        N, A, N, W, &lwork);
    cuDoubleComplex* work;
    CHECK(cudaMalloc((void**)&work, 
        sizeof(cuDoubleComplex) * lwork));   // 分配缓冲空间显存

    int* info;
    CHECK(cudaMalloc((void**)&info, sizeof(int)));     //  返回值
    cusolverDnZheevd(handle, jobz, uplo, N, A, N, W, 
        work, lwork, info);                                // 算出本征值的
    cudaMemcpy(W_cpu, W, sizeof(double) * N, 
        cudaMemcpyDeviceToHost);

    printf("Eigenvalues are:\n");
    for (int n = 0; n < N; ++n)
    {
        printf("%g\n", W_cpu[n]);
    }

    cusolverDnDestroy(handle);

    free(A_cpu);
    free(W_cpu);
    CHECK(cudaFree(A));
    CHECK(cudaFree(W));
    CHECK(cudaFree(work));
    CHECK(cudaFree(info));

    return 0;
}
