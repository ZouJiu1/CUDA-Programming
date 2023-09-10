#include "error.cuh" 
#include <stdio.h>
#include <cublas_v2.h>

void print_matrix(int R, int C, double* A, const char* name);

int main(void)
{
    int M = 2;  // 左矩阵的行
    int K = 3;  // 左矩阵的列，或者右矩阵的行
    int N = 2;  // 右矩阵的列
    int MK = M * K;  // 左矩阵的数字个数，行*列
    int KN = K * N;  // 右矩阵的数字个数，行*列
    int MN = M * N;  // 输出矩阵的数字个数，行*列

    double *h_A = (double*) malloc(sizeof(double) * MK); // 给左矩阵分配内存
    double *h_B = (double*) malloc(sizeof(double) * KN); // 给右矩阵分配内存
    double *h_C = (double*) malloc(sizeof(double) * MN); // 给输出矩阵分配内存
    for (int i = 0; i < MK; i++)
    {
        h_A[i] = i;                 //  给左矩阵赋值的
    }
    print_matrix(M, K, h_A, "A");     
    for (int i = 0; i < KN; i++)
    {
        h_B[i] = i;                 //  给右矩阵赋值的
    }
    print_matrix(K, N, h_B, "B");
    for (int i = 0; i < MN; i++)
    {
        h_C[i] = 0;                //  初始化输出矩阵到全0
    }

    double *g_A, *g_B, *g_C;
    CHECK(cudaMalloc((void **)&g_A, sizeof(double) * MK))   // 给左矩阵分配显存
    CHECK(cudaMalloc((void **)&g_B, sizeof(double) * KN))   // 给右矩阵分配显存
    CHECK(cudaMalloc((void **)&g_C, sizeof(double) * MN))   // 给输出矩阵分配显存

//  参数依次是  --长度--element占用的字节数bytes--host指针---host的stride---device指针----device的stride
    cublasSetVector(MK, sizeof(double), h_A, 1, g_A, 1);    // cublas库移动左矩阵从host到device，并产生专用向量格式
    cublasSetVector(KN, sizeof(double), h_B, 1, g_B, 1);    // cublas库移动右矩阵从host到device，并产生专用向量格式
    cublasSetVector(MN, sizeof(double), h_C, 1, g_C, 1);    // cublas库移动左矩阵从host到device，并产生专用向量格式

    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0;
    double beta = 0.0;

//  D是double，gemm是GEneral matrix-matrix multiplication
//  CUBLAS_OP_N(A)=A,    CUBLAS_OP_T(A)=A的转置
//  公式是  g_C = alpha * (g_A * g_B) + beta * g_C
//  g_A, M, g_B, K, &beta, g_C, M 内的 M, K, M 都是行数的
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha, g_A, M, g_B, K, &beta, g_C, M);
    cublasDestroy(handle);

    cublasGetVector(MN, sizeof(double), g_C, 1, h_C, 1);  // 输出矩阵从 device 移动到 host
    print_matrix(M, N, h_C, "C = A x B");

//  释放host和device的内存
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK(cudaFree(g_A)) 
    CHECK(cudaFree(g_B))
    CHECK(cudaFree(g_C))
    return 0;
}

void print_matrix(int R, int C, double* A, const char* name)
{
    printf("%s = \n", name);
    for (int r = 0; r < R; ++r)  // 行循环
    {
        for (int c = 0; c < C; ++c)   // 列循环
        {
            printf("%10.6f", A[c * R + r]);   //  c*R + r，先列后行输出，列相邻
        }
        printf("\n");
    }
}

