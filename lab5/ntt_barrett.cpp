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

// 可以自行添加需要的头文件

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

void fRead(int *a, int *b, int *n, int *p, int input_id){
    std::string filename = "data/" + std::to_string(input_id) + ".in";
    std::ifstream fin(filename);
    
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }
    
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) {
        fin >> a[i];
    }
    for (int i = 0; i < *n; i++) {
        fin >> b[i];
    }
    fin.close();
}

void fCheck(int *ab, int n, int input_id){
    std::string filename = "data/" + std::to_string(input_id) + ".out";
    std::ifstream fin(filename);
    
    if (!fin.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        std::cout << "无法验证结果" << std::endl;
        return;
    }
    
    bool correct = true;
    for (int i = 0; i < n * 2 - 1; i++) {
        int expected;
        fin >> expected;
        if (ab[i] != expected) {
            correct = false;
            break;
        }
    }
    fin.close();
    
    if (correct) {
        std::cout << "多项式乘法结果正确" << std::endl;
    } else {
        std::cout << "多项式乘法结果错误" << std::endl;
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

void NTT(int *f, int Mod, int type){
    for(int i = 0; i < MAXN; ++i) if(i < idx[i]) std::swap(f[i], f[idx[i]]);

    for(int len = 2; len <= MAXN; len <<= 1){
        
        int mid = len >> 1;
        ll W = _pow(G, (Mod - 1) / len, Mod);

        for(int l = 0; l < MAXN; l += len){
            ll w = 1;
            for(int i = l; i < l + mid ; ++i){
                int a = f[i], b = barrett.mul(w, f[i + mid]); // 蝶形变换，注意需要暂存结果
                f[i] = (a + b) % Mod, f[i + mid] = (a - b + Mod) % Mod;
                w = barrett.mul(w, W);
            }
        }
    }

    if(type == 1){
        std::reverse(f + 1, f + MAXN);
        ll inv = _pow(MAXN, Mod - 2, Mod);
        for(int i = 0; i < MAXN; ++i) f[i] = barrett.mul(f[i], inv);
    }
}

int P, nm;

void poly_multiply(int *a, int *b, int *ab, int n, int p){
    barrett = BarrettReduction(p);

    nm = n + n - 1;
    MAXN = 1, P = 0;
    while(MAXN < nm) MAXN <<= 1, P++; // p表示MAXN是2的几次幂
        
    for(int i = 0; i < MAXN; ++i) idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
    
    memset(a + n, 0, sizeof(int) * (MAXN - n));
    memset(b + n, 0, sizeof(int) * (MAXN - n));

    NTT(a, p, 0), NTT(b, p, 0);
    
    for(int i = 0; i < MAXN; ++i){
        ab[i] = barrett.mul(a[i], b[i]);
    } 
    
    NTT(ab, p, 1);
}

int a[N << 2], b[N << 2], ab[N << 2];

int main(int argc, char *argv[])
{
    
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
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (ms) "<<std::endl;
    }
    return 0;
}

/*
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.010253 (ms) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 73.0194 (ms) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 71.868 (ms) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 72.0748 (ms) 
*/