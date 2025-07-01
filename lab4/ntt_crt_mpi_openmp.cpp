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
#include <pthread.h>
#include <vector>
#include <mpi.h>
#define ll long long

const int N = 300050;
const int G = 3;
ll mod;

const ll MODS[4] = {469762049, 998244353, 1004535809, 1224736769};

void fRead(ll *a, ll *b, int *n, ll *p, int input_id){
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

void fCheck(ll *ab, int n, int input_id){
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        ll x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(ll *ab, int n, int input_id){
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

ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    ll x1, y1;
    ll gcd = exgcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return gcd;
}

ll inv(ll a, ll mod) {
    ll x, y;
    exgcd(a, mod, x, y);
    return (x % mod + mod) % mod;
}

int idx[N << 2], MAXN;

void NTT_OpenMP(ll *f, ll Mod, int type, int maxn) {
    for(int i = 0; i < maxn; ++i) {
        if(i < idx[i]) {
            ll temp = f[i];
            f[i] = f[idx[i]];
            f[idx[i]] = temp;
        }
    }

    for(int len = 2; len <= maxn; len <<= 1){
        int mid = len >> 1;
        ll W = _pow(G, (Mod - 1) / len, Mod);

        //#pragma omp parallel for
        for(int l = 0; l < maxn; l += len){
            ll w = 1;
            for(int i = l; i < l + mid; ++i){
                ll a = f[i], b = (w * f[i + mid]) % Mod;
                f[i] = (a + b) % Mod;
                f[i + mid] = (a - b + Mod) % Mod;
                w = (w * W) % Mod;
            }
        }
    }

    if(type == 1){
        std::reverse(f + 1, f + maxn);
        ll Inv = _pow(maxn, Mod - 2, Mod);
        
        for(int i = 0; i < maxn; ++i) {
            f[i] = (f[i] * Inv) % Mod;
        }
    }
}

class CRT_Solver {
private:
    ll inv_m[4][4]; 
    const ll m[4] = {469762049, 998244353, 1004535809, 1224736769};
    ll mul_mod(ll a, ll b, ll m) {
        return ((__int128)a * b) % m;
    }

public:
    CRT_Solver() {
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < j; ++i) { 
                inv_m[i][j] = inv(m[i], m[j]); 
            }
        }
    }

    ll reconstruct(ll r[4], ll p) {
        ll v[4];

        for (int i = 0; i < 4; ++i) {
            v[i] = r[i]; 
            for (int j = 0; j < i; ++j) {
                v[i] = (v[i] - v[j]); 
                v[i] = (v[i] % m[i] + m[i]) % m[i]; 
                v[i] = (v[i] * inv_m[j][i]) % m[i]; 
            }
        }

        ll result = 0;
        ll current_m_prod = 1;

        for (int i = 0; i < 4; ++i) {
            ll term = mul_mod(v[i], current_m_prod, p); 
            result = (result + term) % p;
            current_m_prod = mul_mod(current_m_prod, m[i], p); 
        }

        return (result % p + p) % p;
    }
};

CRT_Solver crt_solver;

// OpenMP优化的MPI CRT NTT函数
void mpi_crt_ntt_openmp(ll *a, ll *b, ll *result, int maxn, int rank, int size) {
    ll *a_local = new ll[maxn];
    ll *b_local = new ll[maxn];
    ll *mod_result = new ll[maxn];
    ll *all_mod_results = nullptr;
    
    if (rank == 0) {
        all_mod_results = new ll[4 * maxn];
    }
    
    int mods_per_proc = (4 + size - 1) / size;  
    int start_mod = rank * mods_per_proc;
    int end_mod = std::min(start_mod + mods_per_proc, 4);
    
    std::vector<ll> local_results(maxn * (end_mod - start_mod));
    
    for(int mod_idx = start_mod; mod_idx < end_mod; ++mod_idx) {
        ll current_mod = MODS[mod_idx];
        
        memset(a_local, 0, sizeof(ll) * maxn);
        memset(b_local, 0, sizeof(ll) * maxn);
        memset(mod_result, 0, sizeof(ll) * maxn);

        for(int j = 0; j < maxn; ++j) {
            a_local[j] = a[j] % current_mod;
            b_local[j] = b[j] % current_mod;
        }
        
        NTT_OpenMP(a_local, current_mod, 0, maxn);
        NTT_OpenMP(b_local, current_mod, 0, maxn);

        for(int j = 0; j < maxn; ++j) {
            mod_result[j] = (a_local[j] * b_local[j]) % current_mod;
        }

        NTT_OpenMP(mod_result, current_mod, 1, maxn);
        
        for(int j = 0; j < maxn; ++j) {
            local_results[(mod_idx - start_mod) * maxn + j] = mod_result[j];
        }
    }
    
    if (rank == 0) {
        for(int i = 0; i < (end_mod - start_mod) * maxn; ++i) {
            all_mod_results[i] = local_results[i];
        }
        
        for(int p = 1; p < size; ++p) {
            int p_start = p * mods_per_proc;
            int p_end = std::min(p_start + mods_per_proc, 4);
            if(p_start < 4) {
                int recv_size = (p_end - p_start) * maxn;
                MPI_Recv(all_mod_results + p_start * maxn, recv_size, MPI_LONG_LONG, 
                        p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        for(int i = 0; i < maxn; ++i) {
            ll remainders[4];
            for(int j = 0; j < 4; ++j) {
                remainders[j] = all_mod_results[j * maxn + i];
            }
            result[i] = crt_solver.reconstruct(remainders, mod);
        }
    } else {
        if(start_mod < 4) {
            int send_size = (end_mod - start_mod) * maxn;
            MPI_Send(local_results.data(), send_size, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    MPI_Bcast(result, maxn, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    delete[] a_local;
    delete[] b_local;
    delete[] mod_result;
    if (rank == 0) {
        delete[] all_mod_results;
    }
}

int P, nm;

void poly_multiply(ll *a, ll *b, ll *ab, int n, ll p, int rank, int size){
    nm = n + n - 1;
    MAXN = 1, P = 0;
    mod = p;
    while(MAXN < nm) MAXN <<= 1, P++;
    
    for(int i = 0; i < MAXN; ++i) {
        idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
    }

    #pragma omp parallel for
    for(int i = n; i < MAXN; ++i) {
        a[i] = 0;
        b[i] = 0;
    }

    mpi_crt_ntt_openmp(a, b, ab, MAXN, rank, size);
}

ll a[N << 2], b[N << 2], ab[N << 2];

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    omp_get_max_threads(); 
    
    
    int test_begin = 0;
    int test_end = 4;
    
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_;
        ll p_;
        
        if (rank == 0) {
            fRead(a, b, &n_, &p_, i);
            memset(ab, 0, sizeof(ab));
        }
 
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(a, n_, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(b, n_, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        
        memset(ab, 0, sizeof(ab));
        
        MPI_Barrier(MPI_COMM_WORLD);
        auto Start = std::chrono::high_resolution_clock::now();
        
        poly_multiply(a, b, ab, n_, p_, rank, size);
        
        MPI_Barrier(MPI_COMM_WORLD);
        auto End = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        
        if (rank == 0) {
            fCheck(ab, n_, i);
            std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
            fWrite(ab, n_, i);
        }
    }
    
    MPI_Finalize();
    return 0;
}
/*

1线
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.093578 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 124.072 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 122.642 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 120.234 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 1337006139375617 : 123.381 (us)

2线
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.40413 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 89.1112 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 84.6348 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 81.3968 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 1337006139375617 : 84.8917 (us)

3线
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.523005 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 98.7992 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 96.0585 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 95.1088 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 1337006139375617 : 97.8036 (us) 

4线
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 2.96084 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 113.869 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 97.687 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 96.6613 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 1337006139375617 : 101.43 (us)

5线
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 6.53892 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 114.28 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 109.836 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 110.252 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 1337006139375617 : 113.168 (us)

6线
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 10.4462 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 121.944 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 117.384 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 112.949 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 1337006139375617 : 118.226 (us) 

7线
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 8.70255 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 121.085 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 122.167 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 120.017 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 1337006139375617 : 122.778 (us)

8线
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 9.93037 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 126.559 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 136.814 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 126.309 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 1337006139375617 : 128.601 (us) 
*/

