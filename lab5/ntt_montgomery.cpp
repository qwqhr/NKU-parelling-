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
#include <vector>
#define ll long long

// 可以自行添加需要的头文件

const int N = 300050;
const int G = 3;

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

int Mod;

// Montgomery规约结构体
struct MontgomeryReduction {
    int mod;
    unsigned int inv_mod;      // mod * inv_mod = -1 (mod R)
    int mont_r_mod_n;          // R mod mod
    int mont_r2_mod_n;         // R^2 mod mod
    static const unsigned long long R = 1ULL << 32;
    
    MontgomeryReduction() {}
    
    MontgomeryReduction(int m) {
        mod = m;
        
        // 计算 inv_mod = -mod^{-1} mod R
        ll x, y;
        exgcd(mod, R, x, y);
        inv_mod = (unsigned int)(-x);
        
        mont_r_mod_n = (int)(R % mod);
        mont_r2_mod_n = (int)((1ULL * mont_r_mod_n * mont_r_mod_n) % mod);
    }
    
    // 扩展欧几里得算法
    ll exgcd(ll a, ll b, ll &x, ll &y) {
        if (b == 0) {
            x = 1, y = 0;
            return a; 
        }
        ll x1, y1;
        ll d = exgcd(b, a % b, x1, y1); 
        x = y1, y = x1 - (a / b) * y1;
        return d; 
    }
    
    // REDC算法
    inline int REDC(unsigned long long T) {
        unsigned int m = (unsigned int)T * inv_mod;
        unsigned long long t_long = (T + (unsigned long long)m * mod) >> 32;
        
        int result;
        if (t_long >= mod) result = (int)(t_long - mod);
        else result = (int)t_long;
        
        return result;
    }
    
    // 转换到蒙哥马利域
    inline int to_mont(int x) {
        return REDC((unsigned long long)x * mont_r2_mod_n);
    }
    
    // 从蒙哥马利域转换回普通域
    inline int from_mont(int mont_x) {
        return REDC((unsigned long long)mont_x);
    }
    
    // 蒙哥马利模乘 
    inline int mul(int mont_a, int mont_b) {
        return REDC((unsigned long long)mont_a * mont_b);
    }
    
    // 快速幂 
    int pow(int a, int p) {
        int mont_a = to_mont(a);
        int mont_ans = mont_r_mod_n; // 蒙哥马利域中的1
        int mont_mul = mont_a;
        
        while(p){
            if(p & 1) mont_ans = mul(mont_ans, mont_mul);
            mont_mul = mul(mont_mul, mont_mul);
            p >>= 1;
        }
        
        return from_mont(mont_ans);
    }
};

MontgomeryReduction montgomery;


int _pow(int a, int p){
    return montgomery.pow(a, p);
}

int idx[N << 2], MAXN;

void NTT(int *f, int type){
    for(int i = 0; i < MAXN; ++i) if(i < idx[i]) std::swap(f[i], f[idx[i]]);

    for(int len = 2; len <= MAXN; len <<= 1){
        
        int mid = len >> 1;
        int W = _pow(G, (Mod - 1) / len);
        int mont_W = montgomery.to_mont(W);
        
        for(int l = 0; l < MAXN; l += len){
            int mont_w = montgomery.mont_r_mod_n; // 蒙哥马利域中的1

            for(int i = l; i < l + mid ; ++i){
                int mont_a = f[i];
                int mont_b = montgomery.mul(mont_w, f[i + mid]); // 使用蒙哥马利乘法
                
                f[i] = (mont_a + mont_b) % Mod;      // 保持原有的模加
                f[i + mid] = (mont_a - mont_b + Mod) % Mod; // 保持原有的模减
                
                mont_w = montgomery.mul(mont_w, mont_W); // 使用蒙哥马利乘法
            }
        }
    }

    if(type == 1){
        std::reverse(f + 1, f + MAXN);
        int inv = _pow(MAXN, Mod - 2);
        int mont_inv = montgomery.to_mont(inv);
        for(int i = 0; i < MAXN; ++i) {
            f[i] = montgomery.mul(f[i], mont_inv); // 使用蒙哥马利乘法
        }
    }
}

int P, nm;

void poly_multiply(int *a, int *b, int *ab, int n, int p){
    nm = n + n - 1, Mod = p;
    MAXN = 1, P = 0;
    while(MAXN < nm) MAXN <<= 1, P++;
        
    for(int i = 0; i < MAXN; ++i) idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
    
    // 初始化Montgomery结构体
    montgomery = MontgomeryReduction(Mod);

    // 将输入转换到蒙哥马利域
    for(int i = 0; i < n; ++i) {
        a[i] = montgomery.to_mont(a[i]);
        b[i] = montgomery.to_mont(b[i]);
    }
    for(int i = n; i < MAXN; ++i) a[i] = 0, b[i] = 0;

    NTT(a, 0), NTT(b, 0);
    
    for(int i = 0; i < MAXN; ++i){
        ab[i] = montgomery.mul(a[i], b[i]); // 使用蒙哥马利乘法
    } 
    
    NTT(ab, 1);

    // 将结果转换回普通域
    for(int i = 0; i < n + n - 1; ++i) {
        ab[i] = montgomery.from_mont(ab[i]);
    }
}

int a[N << 2], b[N << 2], ab[N << 2];

int main(int argc, char *argv[])
{
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
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
average latency for n = 4 p = 7340033 : 0.009698 (ms) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 61.8339 (ms) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 60.9402 (ms) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 60.7637 (ms) 
*/

