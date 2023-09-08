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

int N; // number of atoms                  //atoms的个数
const int NUM_REPEATS = 20; // number of timings    重复运算耗时的次数
const int MN = 10; // maximum number of neighbors for each atom      //邻居的最大个数
const real cutoff = 1.9; // in units of Angstrom         // 距离的最大值
const real cutoff_square = cutoff * cutoff;   // 距离的平方

void read_xy(std::vector<real>& x, std::vector<real>& y);    //从x, y坐标的txt档案读取坐标
void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y);  //耗时
void print_neighbor(const int *NN, const int *NL); //输出到txt档案
 
int main(void)
{
    std::vector<real> x, y;    //初始化vector向量数组的
    read_xy(x, y);             //读取
    N = x.size();              //atoms的个数
    int *NN = (int*) malloc(N * sizeof(int));   //分配atoms个数个内存
    int *NL = (int*) malloc(N * MN * sizeof(int)); //分配矩阵内存，每个碳atoms分配最大邻居10个
    
    timing(NN, NL, x, y);   // 运算邻居的呢
    print_neighbor(NN, NL);   //  输出邻居的呢

    free(NN);  //释放内存的
    free(NL);  //释放内存的
    return 0;
}

void read_xy(std::vector<real>& v_x, std::vector<real>& v_y)   // 参数是x, y方向的坐标
{
    std::ifstream infile("xy.txt");          // 读取输入坐标x,y档案
    std::string line, word;    //初始化 string 字符串
    if(!infile)    // 文件是否正常open
    {
        std::cout << "Cannot open xy.txt" << std::endl;     //文件open出错了
        exit(1);
    }
    while (std::getline(infile, line))   //  从读取的文件内拿到一行
    {
        std::istringstream words(line);     //  从这一行内读取字符串
        if(line.length() == 0)   //是否空
        {
            continue;
        }
        for (int i = 0; i < 2; i++)   //两个x，y坐标的
        {
            if(words >> word)   // 字符串依次赋值给word
            { 
                if(i == 0)
                {
                    v_x.push_back(std::stod(word));   // 首先是x坐标
                }
                if(i==1)
                {
                    v_y.push_back(std::stod(word)); //  然后是y坐标
                }
            }
            else
            {
                std::cout << "Error for reading xy.txt" << std::endl;  //读取出错的
                exit(1);
            }
        }
    }
    infile.close();  //关闭打开的文件
}

// 参数是 NN数组 N个碳atoms，NL邻居列表的 NxMN也就是Nx10，x是x坐标，y是y坐标
void find_neighbor(int *NN, int *NL, const real* x, const real* y)  
{
    for (int n = 0; n < N; n++)   
    {
        NN[n] = 0;     //  N个碳atoms的邻居个数初始化到 0
    }

    for (int n1 = 0; n1 < N; ++n1)  //遍历 N 个碳atoms
    {
        real x1 = x[n1];  // 第n1个碳atoms的x坐标
        real y1 = y[n1];  // 第n1个碳atoms的y坐标
        for (int n2 = n1 + 1; n2 < N; ++n2)        //遍历其他的碳atoms，因n1, n2或者n2, n1是相同的内容，所以截断一半
        {
            real x12 = x[n2] - x1; // n1和n2的x坐标相减
            real y12 = y[n2] - y1; // n1和n2的y坐标相减
            real distance_square = x12 * x12 + y12 * y12;  // 求出坐标差的平方和
            if (distance_square < cutoff_square)       //若是平方和 < 给定的阀值，则算邻居的
            {
                NL[n1 * MN + NN[n1]++] = n2;//  record邻居的index，NL可以看作矩阵，n1是行，NN[n1]是列也是n1的邻居个数
                NL[n2 * MN + NN[n2]++] = n1;//  record邻居的index，NL可以看作矩阵，n2是行，NN[n2]是列也是n2的邻居个数
            }
        }
    }
}

void timing(int *NN, int *NL, std::vector<real> x, std::vector<real> y) // 耗时
{
    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)  // 算耗时需要重复NUM_REPEATS次
    {
        cudaEvent_t start, stop;                  //  用来算耗时的
        CHECK(cudaEventCreate(&start));           //  初始化
        CHECK(cudaEventCreate(&stop));//  初始化
        CHECK(cudaEventRecord(start));
        while(cudaEventQuery(start)!=cudaSuccess){}
        find_neighbor(NN, NL, x.data(), y.data());

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop)); //  算耗时
        std::cout << "Time = " << elapsed_time << " ms." << std::endl;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
}

void print_neighbor(const int *NN, const int *NL)        // 输出到档案内部
{
    std::ofstream outfile("neighbor.txt");               //  open输出档案
    if (!outfile)                                        //  输出档案是否open
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
        outfile << NN[n];                       // 首先给出第n个碳atoms的邻居个数
        for (int k = 0; k < MN; ++k)            //  遍历MN个邻居
        {
            if(k < NN[n])                       //  k要满足 < 邻居的个数 NN[n]
            {
                outfile << " " << NL[n * MN + k];    //  输出第n个碳atoms，第k个邻居的标号index，行n,列k
            }
            else
            {
                outfile << " NaN";         // 没有这么多邻居，输出NaN
            }
        }
        outfile << std::endl;   // 结束符
    }
    outfile.close();   //  关闭输出档案
}