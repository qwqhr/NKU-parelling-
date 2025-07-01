#include <mpi.h>
#include <cstring>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#define ll long long

const int N = 300050;
const int G = 3;

struct BarrettReduction {
    ll mod;
    ll mu;  // μ = ⌊2^(2k)/mod⌋
    int k;  // mod的位数
    
    BarrettReduction() {}
    
    BarrettReduction(ll m) {
        mod = m;
        k = 64;  
        mu = ((__int128)1 << k) / m;
    }
    
    // 模乘
    ll mul(ll a, ll b) {
        ll ab = a * b;
        ll q = ((__int128)mu * ab) >> k;  // 近似商
        ll r = ab - q * mod;
        return r >= mod ? r - mod : r;
    }
};

BarrettReduction barrett;
int rank, size;

void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数 - 使用相对路径
    std::string str1 = "nttdata/"; // <--- 修改点：去掉开头的 '/', 保证末尾有 '/'
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in"; // 路径现在是 "nttdata/0.in", "nttdata/1.in" 等

    // char data_path[strin.size() + 1];
    // std::copy(strin.begin(), strin.end(), data_path);
    // data_path[strin.size()] = '\0';
    // 使用 C++ string 的 c_str() 方法更简洁
    const char* data_path = strin.c_str();

    std::ifstream fin;
    fin.open(data_path, std::ios::in);

    // --- 添加文件打开检查 ---
    if (!fin.is_open()) {
        std::cerr << "错误：无法打开输入文件: " << data_path << std::endl;
        std::cerr << "请确保在程序运行目录下存在 'nttdata' 子目录，并且其中包含 '" << str2 << ".in' 文件。" << std::endl;
        exit(EXIT_FAILURE); // 文件打不开，直接退出程序
    }
    // --- 检查结束 ---

    fin >> *n >> *p;
    // 添加简单的读取检查（可选，但有助于调试）
    if (fin.fail()) {
         std::cerr << "错误：读取 n 或 p 时发生错误，文件: " << data_path << std::endl;
         exit(EXIT_FAILURE);
    }

    for (int i = 0; i < *n; i++){
        fin >> a[i];
         if (fin.fail() && !fin.eof()) { // 检查读取失败，且不是因为读到文件末尾
            std::cerr << "错误：从文件 " << data_path << " 读取 a[" << i << "] 时出错。" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < *n; i++){
        fin >> b[i];
         if (fin.fail() && !fin.eof()) {
            std::cerr << "错误：从文件 " << data_path << " 读取 b[" << i << "] 时出错。" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    fin.close(); // 好习惯：关闭文件流
}

void fCheck(int *ab, int n, int input_id){
    // 判断多项式乘法结果是否正确
    std::string str1 = "/root/lab2/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if(input_id==0) std::cout<<x<<" "<<ab[i]<<" "<<i<<std::endl;
        if(x != ab[i]){
            std::cout<<std::endl<<" "<<x<<" "<<ab[i]<<" "<<i<<std::endl;
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
}

ll _pow(ll a, ll p,ll Mod){
    ll ans = 1, mul = a;
    while(p){
        if(p & 1) ans = (ans * mul) % Mod;
        mul = (mul * mul) % Mod;
        p >>= 1;
    }
    return ans;
}

int idx[N << 2], MAXN;

// MPI并行版本的NTT
void parallel_NTT(int *f, int Mod, int type){
    // 1. 位逆序置换 - 并行处理
    int chunk_size = MAXN / size;
    int start = rank * chunk_size;
    int end = (rank + 1) * chunk_size;
    
    for(int i = start; i < end; ++i) {
        if(i < idx[i]) std::swap(f[i], f[idx[i]]);
    }
    
    // 同步位逆序结果
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                  f, chunk_size, MPI_INT, MPI_COMM_WORLD);

    // 2. 蝶形变换
    for(int len = 2; len <= MAXN; len <<= 1){
        int mid = len >> 1;
        ll W = _pow(G, (Mod - 1) / len, Mod);
        
        // 每个进程处理自己负责的数据块
        for(int l = start; l < end && l < MAXN; l += len){
            if(l + len > end) break; // 避免越界
            
            ll w = 1;
            for(int i = l; i < l + mid && i < end; ++i){
                if(i + mid < MAXN) {
                    int a = f[i], b = barrett.mul(w, f[i + mid]);
                    f[i] = (a + b) % Mod;
                    f[i + mid] = (a - b + Mod) % Mod;
                    w = barrett.mul(w, W);
                }
            }
        }
        
        // 每层变换后同步数据
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                      f, chunk_size, MPI_INT, MPI_COMM_WORLD);
    }

    if(type == 1){
        // 逆变换处理
        if(rank == 0) {
            std::reverse(f + 1, f + MAXN);
        }
        MPI_Bcast(f, MAXN, MPI_INT, 0, MPI_COMM_WORLD);
        
        ll inv = _pow(MAXN, Mod - 2, Mod);
        for(int i = start; i < end; ++i) {
            f[i] = barrett.mul(f[i], inv);
        }
        
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                      f, chunk_size, MPI_INT, MPI_COMM_WORLD);
    }
}

// 并行点值乘法
void parallel_pointwise_multiply(int *a, int *b, int *ab) {
    int chunk_size = MAXN / size;
    int start = rank * chunk_size;
    int end = (rank + 1) * chunk_size;
    
    for(int i = start; i < end; ++i){
        ab[i] = barrett.mul(a[i], b[i]);
    }
    
    // 收集所有结果
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                  ab, chunk_size, MPI_INT, MPI_COMM_WORLD);
}

int P, nm;

void poly_multiply(int *a, int *b, int *ab, int n, int p){
    barrett = BarrettReduction(p);

    nm = n + n - 1;
    MAXN = 1, P = 0;
    while(MAXN < nm) MAXN <<= 1, P++;//p表示MAXN是2的几次幂
    
    // 只有rank 0计算idx数组
    if(rank == 0) {
        for(int i = 0; i < MAXN; ++i) {
            idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
        }
    }
    
    // 广播idx数组到所有进程
    MPI_Bcast(idx, MAXN, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 数据预处理
    if(rank == 0) {
        memset(a + n, 0, sizeof(int) * (MAXN - n));
        memset(b + n, 0, sizeof(int) * (MAXN - n));
    }
    
    // 广播数据到所有进程
    MPI_Bcast(a, MAXN, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, MAXN, MPI_INT, 0, MPI_COMM_WORLD);

    // 并行NTT
    parallel_NTT(a, p, 0);
    parallel_NTT(b, p, 0);
    
    // 并行点值乘法
    parallel_pointwise_multiply(a, b, ab);
    
    // 并行逆NTT
    parallel_NTT(ab, p, 1);
}

int a[N << 2], b[N << 2], ab[N << 1];

int main(int argc, char *argv[])
{
    // 初始化MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入的后三个大模数分别为 7340033 104857601 469762049 
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 3;
    
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        
        // 只有rank 0读取数据
        if(rank == 0) {
            fRead(a, b, &n_, &p_, i);
            memset(ab, 0, sizeof(ab));
        }
        
        // 广播n_和p_到所有进程
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        //auto Start = std::chrono::high_resolution_clock::now();
        // 使用并行版本的多项式乘法
        poly_multiply(a, b, ab, n_, p_);
        //auto End = std::chrono::high_resolution_clock::now();
        
       // std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        //ans += elapsed.count();
        
        // 只有rank 0进行检查和输出
        if(rank == 0) {
            fCheck(ab, n_, i);
            std::cout<<"[MPI Rank 0] average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
            //fWrite(ab, n_, i);
        }
    }
    
    // 清理MPI
    MPI_Finalize();
    return 0;
}