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
    ll mu;
    int k;
    
    BarrettReduction() {}
    
    BarrettReduction(ll m) {
        mod = m;
        k = 64;  
        mu = ((__int128)1 << k) / m;
    }
    
    ll mul(ll a, ll b) {
        ll ab = a * b;
        ll q = ((__int128)mu * ab) >> k;
        ll r = ab - q * mod;
        return r >= mod ? r - mod : r;
    }
};

BarrettReduction barrett;
int rank, size;

struct MPITimer {
    double start_time;
    double end_time;
    double total_time;
    
    void reset() {
        start_time = end_time = total_time = 0.0;
    }
    
    void start() {
        MPI_Barrier(MPI_COMM_WORLD);  // 确保所有进程同步开始
        start_time = MPI_Wtime();
    }
    
    void end() {
        MPI_Barrier(MPI_COMM_WORLD);  // 确保所有进程同步结束
        end_time = MPI_Wtime();
        total_time = end_time - start_time;
    }
    
    void print_results(int n, int p,int mod) {
        if(rank == 0) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "=== Performance Results (n=" << n << ", p=" << p << ", mod=" << mod << ") ===" << std::endl;
            std::cout << "Total Time: " << total_time * 1000 << " ms" << std::endl;
            }
    }
};

MPITimer timer;

// 文件操作函数保持不变...
void fRead(int *a, int *b, int *n, int *p, int input_id){
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){   
        fin>>b[i];
    }
}

void fCheck(int *ab, int n, int input_id){
    std::string str1 = "/nttdata/";
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
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<" "<<x<<" "<<ab[i]<<" "<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
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

ll _pow(ll a, ll p, ll Mod){
    ll ans = 1, mul = a;
    while(p){
        if(p & 1) ans = (ans * mul) % Mod;
        mul = (mul * mul) % Mod;
        p >>= 1;
    }
    return ans;
}

int idx[N << 2], MAXN;

// 主从策略：主进程执行NTT，从进程等待
void master_slave_NTT(int *f, int Mod, int type){
    if(rank == 0) {
        // 主进程执行完整的NTT
        for(int i = 0; i < MAXN; ++i) {
            if(i < idx[i]) std::swap(f[i], f[idx[i]]);
        }

        for(int len = 2; len <= MAXN; len <<= 1){
            int mid = len >> 1;
            ll W = _pow(G, (Mod - 1) / len, Mod);

            for(int l = 0; l < MAXN; l += len){
                ll w = 1;
                for(int i = l; i < l + mid; ++i){
                    int a = f[i], b = barrett.mul(w, f[i + mid]);
                    f[i] = (a + b) % Mod;
                    f[i + mid] = (a - b + Mod) % Mod;
                    w = barrett.mul(w, W);
                }
            }
        }

        if(type == 1){
            std::reverse(f + 1, f + MAXN);
            ll inv = _pow(MAXN, Mod - 2, Mod);
            for(int i = 0; i < MAXN; ++i) {
                f[i] = barrett.mul(f[i], inv);
            }
        }
    }
    
    // 广播结果给所有进程
    MPI_Bcast(f, MAXN, MPI_INT, 0, MPI_COMM_WORLD);
}

// 并行点值乘法
void parallel_pointwise_multiply(int *a, int *b, int *ab) {
    int chunk_size = MAXN / size;
    int start = rank * chunk_size;
    int end = (rank + 1) * chunk_size;
    
    for(int i = start; i < end; ++i){
        ab[i] = barrett.mul(a[i], b[i]);
    }
    
    // 收集所有结果到所有进程
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                  ab, chunk_size, MPI_INT, MPI_COMM_WORLD);
}

int P, nm;

void poly_multiply(int *a, int *b, int *ab, int n, int p){
    barrett = BarrettReduction(p);

    nm = n + n - 1;
    MAXN = 1, P = 0;
    while(MAXN < nm) MAXN <<= 1, P++;
    
    // 只有rank 0计算idx数组
    if(rank == 0) {
        for(int i = 0; i < MAXN; ++i) {
            idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
        }
    }
    
    // 广播idx数组到所有进程
    MPI_Bcast(idx, MAXN, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 主线程数据预处理
    if(rank == 0) {
        memset(a + n, 0, sizeof(int) * (MAXN - n));
        memset(b + n, 0, sizeof(int) * (MAXN - n));
    }
    
    MPI_Bcast(a, MAXN, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, MAXN, MPI_INT, 0, MPI_COMM_WORLD);

    master_slave_NTT(a, p, 0);
    master_slave_NTT(b, p, 0);
    
    // 并行点值乘法
    parallel_pointwise_multiply(a, b, ab);
    
    master_slave_NTT(ab, p, 1);
}

int a[N << 2], b[N << 2], ab[N << 2];

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int test_begin = 0;
    int test_end = 3;
    
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        
        if(rank == 0) {
            fRead(a, b, &n_, &p_, i);
            memset(ab, 0, sizeof(ab));
        }
        
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        timer.reset();
        timer.start();

        poly_multiply(a, b, ab, n_, p_);
        
        timer.end();

        if(rank == 0) {
            fCheck(ab, n_, i);
            timer.print_results(n_, size, p_);
            fWrite(ab, n_, i);
        }
    }
    
    MPI_Finalize();
    return 0;
}



/*
多项式乘法结果正确
=== Performance Results (n=4, p=8, mod=7340033) ===
Total Time: 1.804113 ms
多项式乘法结果正确
=== Performance Results (n=131072, p=8, mod=7340033) ===
Total Time: 75.749636 ms
多项式乘法结果正确
=== Performance Results (n=131072, p=8, mod=104857601) ===
Total Time: 72.449207 ms
多项式乘法结果正确
=== Performance Results (n=131072, p=8, mod=469762049) ===
Total Time: 72.235584 ms
*/

