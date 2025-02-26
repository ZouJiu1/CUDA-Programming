#include "error.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_DP
    typedef double real;
#else
    typedef float real;
#endif

int N; // number of atoms
const int NUM_REPEATS = 20; // number of timings
const int MN = 10; // maximum number of neighbors for each atom

const real cutoff = 1.9; // in units of Angstrom
const real cutoff_square = cutoff * cutoff;

void read_xy(std::vector<real>& , std::vector<real>& );
void timing(int *, int *, const real *, const real *, const bool);
void print_neighbor(const int *, const int *, const bool);

int main(void)
{
    std::vector<real> v_x, v_y;
    read_xy(v_x, v_y);  //读取x,y坐标到向量内
    N = v_x.size();
    int mem1 = sizeof(int) * N;
    int mem2 = sizeof(int) * N * MN;
    int mem3 = sizeof(real) * N;
    int *h_NN = (int*) malloc(mem1);
    int *h_NL = (int*) malloc(mem2);
    int *d_NN, *d_NL;
    real *d_x, *d_y;
    CHECK(cudaMalloc(&d_NN, mem1)); // 分配GPU显存给 d_NN，用来 record 每个碳atoms邻居个数的
    CHECK(cudaMalloc(&d_NL, mem2));  // 分配GPU显存该 d_NL，用来 record 每个碳atoms的邻居，可以看作矩阵的
    CHECK(cudaMalloc(&d_x, mem3));   //  分配GPU显存给 d_x，用来存x坐标
    CHECK(cudaMalloc(&d_y, mem3));   // 分配GPU显存给 d_y，用来存y坐标
    CHECK(cudaMemcpy(d_x, v_x.data(), mem3, cudaMemcpyHostToDevice)); //  复制x到d_x
    CHECK(cudaMemcpy(d_y, v_y.data(), mem3, cudaMemcpyHostToDevice));  //  复制y到d_y

    std::cout << std::endl << "using atomicAdd:" << std::endl;
    timing(d_NN, d_NL, d_x, d_y, true);   //  耗时
    std::cout << std::endl << "not using atomicAdd:" << std::endl;
    timing(d_NN, d_NL, d_x, d_y, false);  //  耗时


    CHECK(cudaMemcpy(h_NN, d_NN, mem1, cudaMemcpyDeviceToHost)); // 复制到host h_NN
    CHECK(cudaMemcpy(h_NL, d_NL, mem2, cudaMemcpyDeviceToHost)); // 复制到host h_NL

    print_neighbor(h_NN, h_NL, false);

    CHECK(cudaFree(d_NN)); //  释放显存的
    CHECK(cudaFree(d_NL));
    CHECK(cudaFree(d_x)); 
    CHECK(cudaFree(d_y));
    free(h_NN);           //  释放host内存的
    free(h_NL);
    return 0;
}

void read_xy(std::vector<real>& v_x, std::vector<real>& v_y)
{
    std::ifstream infile("xy.txt");
    std::string line, word;
    if(!infile)
    {
        std::cout << "Cannot open xy.txt" << std::endl;
        exit(1);
    }
    while(std::getline(infile, line))
    {
        std::istringstream words(line);
        if(line.length()==0)
        {
            continue;
        }
        for(int i=0;i<2;i++)
        {
            if(words >> word)
            {
                if(i==0)
                {
                    v_x.push_back(std::stod(word));
                }
                if(i==1)
                {
                    v_y.push_back(std::stod(word));
                }
            }
            else
            {
                std::cout << "Error for reading xy.txt" << std::endl;
                exit(1);
            }
        }
    }
    infile.close();
}

//  使用了atomic函数
void __global__ find_neighbor_atomic  
(
    int *d_NN, int *d_NL, const real *d_x, const real *d_y,
    const int N, const real cutoff_square
)
{
    const int n1 = blockIdx.x * blockDim.x + threadIdx.x; // 拿到线程的标号index，表示第n1个碳 atoms 所在线程的
    if (n1 < N)  //  线程没有越界就可以
    {
        d_NN[n1] = 0;            //初始化，第n1个碳的邻居个数 = 0
        const real x1 = d_x[n1];      // 第n1个碳atoms的x坐标
        const real y1 = d_y[n1];      // 第n1个碳atoms的y坐标
        for (int n2 = n1 + 1; n2 < N; ++n2)
        {
            const real x12 = d_x[n2] - x1;  // n1和n2的x坐标相减
            const real y12 = d_y[n2] - y1;  // n1和n2的y坐标相减
            const real distance_square = x12 * x12 + y12 * y12;    // 求出坐标差的平方和
            if (distance_square < cutoff_square) //若是平方和 < 给定的阀值，则算邻居的
            {
                //  record邻居的index，d_NL可以看作矩阵，n1是行，d_NN[n1]是列也是n1的邻居个数，atomicAdd等价 “++”自增操作的
                d_NL[n1 * MN + atomicAdd(&d_NN[n1], 1)] = n2;
   //  record邻居的index，d_NL可以看作矩阵，n2是行，d_NN[n2]是列也是n1的邻居个数，atomicAdd等价 “++”自增操作的
                d_NL[n2 * MN + atomicAdd(&d_NN[n2], 1)] = n1;
            }
        }
    }
}

//  不使用atomic函数
void __global__ find_neighbor_no_atomic
(
    int *d_NN, int *d_NL, const real *d_x, const real *d_y,
    const int N, const real cutoff_square
)
{
    // 这的 n1 可以看作是列，N 是碳atoms的个数
    const int n1 = blockIdx.x * blockDim.x + threadIdx.x; // 拿到线程的标号index，表示第n1个碳 atoms 所在线程的
    if (n1 < N)
    {
        int count = 0;  // record邻居的个数，放在某个线程寄存器内部，访问速度最好
        const real x1 = d_x[n1];  // 第n1个碳atoms的x坐标
        const real y1 = d_y[n1];  // 第n1个碳atoms的y坐标
        // n2 可以看作是行的，但保存数据用的是count，保证record是continuous的
        for (int n2 = 0; n2 < N; ++n2)  //没有截断一半，而是从0开始，因不能使用atomic函数
        {
            const real x12 = d_x[n2] - x1;  // n1和n2的x坐标相减
            const real y12 = d_y[n2] - y1;  // n1和n2的y坐标相减
            const real distance_square = x12 * x12 + y12 * y12;   // 求出坐标差的平方和
            if ((distance_square < cutoff_square) && (n2 != n1)) //若是平方和 < 给定的阀值，则算邻居的，且不是同一个碳atoms
            {
            // 相邻的threadIdx.x对应的线程的 d_x, d_y, d_NL也是相邻的，所以n1看作列；
            // count看作行，n1 可以看作是列，相邻的n1也就是相邻的threadIdx.x对应的d_NL也是相邻的；
            // 相邻的，所以写入操作是合并的；写入方向是矩阵的列
                d_NL[(count++) * N + n1] = n2;
            }
        }
        d_NN[n1] = count; // record 第 n1个碳的邻居个数
    }
}

void timing
(
    int *d_NN, int *d_NL, const real *d_x, const real *d_y, 
    const bool atomic
)
{
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        int block_size = 128; //block size = blockDim.x，一个block块内部线程的个数
        int grid_size = (N + block_size - 1) / block_size; // grid size = gridDim.x，一个网格grid内部的block个数

        if (atomic)
        {
            find_neighbor_atomic<<<grid_size, block_size>>> // 写入执行配置的
            (d_NN, d_NL, d_x, d_y, N, cutoff_square);
        }
        else
        {
            find_neighbor_no_atomic<<<grid_size, block_size>>>
            (d_NN, d_NL, d_x, d_y, N, cutoff_square);
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        std::cout << "Time = " << elapsed_time << " ms." << std::endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
}

void print_neighbor(const int *NN, const int *NL, const bool atomic)
{
    std::ofstream outfile("neighbor.txt");
    if (!outfile)
    {
        std::cout << "Cannot open neighbor.txt" << std::endl;
    }
    for (int n = 0; n < N; ++n)
    {
        if (NN[n] > MN)
        {
            std::cout << "Error: MN is too small." << std::endl;
            exit(1);
        }
        outfile << NN[n];
        for (int k = 0; k < MN; ++k)
        {
            if(k < NN[n])
            {
                int tmp = atomic ? NL[n * MN + k] : NL[k * N + n];
                outfile << " " << tmp;
            }
            else
            {
                outfile << " NaN";
            }
        }
        outfile << std::endl;
    }
    outfile.close();
}


