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
#include <arm_neon.h>
#define ll long long

const int N = 300050;
const int G = 3;
ll mod;

const ll MODS[4] = {469762049, 998244353, 1004535809, 1224736769};

ll _pow(ll a, ll p, ll Mod);
ll exgcd(ll a, ll b, ll &x, ll &y);
ll inv(ll a, ll mod);

// Montgomery 规约
struct MontgomeryContext {
    ll Mod;
    unsigned int inv_Mod; // N' such that Mod * N' = -1 (mod R)
    const unsigned long long R = 1ULL << 32;
    ll mont_r;      // R mod Mod
    ll mont_r2;     // R^2 mod Mod
    int64x2_t mod_vec; // SIMD向量存储模数
    
    void init(ll m) {
        Mod = m;
        mod_vec = vdupq_n_s64(Mod);
        
        // 计算 inv_Mod: Mod * inv_Mod = -1 (mod R)
        ll x, y;
        exgcd(Mod, R, x, y);
        inv_Mod = (unsigned int)(-x);
        
        mont_r = R % Mod;
        mont_r2 = ((unsigned __int128)mont_r * mont_r) % Mod;
    }
};

MontgomeryContext mont_ctx[4];

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

inline ll REDC(unsigned __int128 T, const MontgomeryContext& ctx) {
    unsigned int m = (unsigned int)T * ctx.inv_Mod;
    unsigned __int128 t = (T + (unsigned __int128)m * ctx.Mod) >> 32;
    return t >= ctx.Mod ? t - ctx.Mod : t;
}

inline ll to_mont(ll x, const MontgomeryContext& ctx) {
    return REDC((unsigned __int128)x * ctx.mont_r2, ctx);
}

inline ll from_mont(ll mont_x, const MontgomeryContext& ctx) {
    return REDC(mont_x, ctx);
}

inline ll mont_mul(ll mont_a, ll mont_b, const MontgomeryContext& ctx) {
    return REDC((unsigned __int128)mont_a * mont_b, ctx);
}

inline ll mont_add(ll mont_a, ll mont_b, const MontgomeryContext& ctx) {
    ll sum = mont_a + mont_b;
    return sum >= ctx.Mod ? sum - ctx.Mod : sum;
}

inline ll mont_sub(ll mont_a, ll mont_b, const MontgomeryContext& ctx) {
    return mont_a >= mont_b ? mont_a - mont_b : mont_a - mont_b + ctx.Mod;
}

inline int64x2_t simd_mont_add(int64x2_t a, int64x2_t b, const MontgomeryContext& ctx) {
    int64x2_t sum = vaddq_s64(a, b);
    uint64x2_t mask = vcgeq_s64(sum, ctx.mod_vec);
    int64x2_t correction = vandq_s64(ctx.mod_vec, vreinterpretq_s64_u64(mask));
    return vsubq_s64(sum, correction);
}

inline int64x2_t simd_mont_sub(int64x2_t a, int64x2_t b, const MontgomeryContext& ctx) {
    int64x2_t diff = vsubq_s64(a, b);
    uint64x2_t mask = vcltq_s64(diff, vdupq_n_s64(0));
    int64x2_t correction = vandq_s64(ctx.mod_vec, vreinterpretq_s64_u64(mask));
    return vaddq_s64(diff, correction);
}

// SIMD版本的蒙哥马利乘法，一次处理2个元素
inline int64x2_t simd_mont_mul(int64x2_t mont_a, int64x2_t mont_b, const MontgomeryContext& ctx) {
    alignas(16) int64_t a_vals[2], b_vals[2], result[2];
    vst1q_s64(a_vals, mont_a);
    vst1q_s64(b_vals, mont_b);
    
    result[0] = mont_mul(a_vals[0], b_vals[0], ctx);
    result[1] = mont_mul(a_vals[1], b_vals[1], ctx);
    
    return vld1q_s64(result);
}

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

int idx[N << 2], MAXN;

void NTT(ll *f, ll Mod, int type, int maxn) {
    // 获取对应模数的Montgomery
    int mod_idx = 0;
    for(int i = 0; i < 4; i++) {
        if(MODS[i] == Mod) {
            mod_idx = i;
            break;
        }
    }
    const MontgomeryContext& ctx = mont_ctx[mod_idx];

    for(int i = 0; i < maxn; ++i) {
        if(i < idx[i]) std::swap(f[i], f[idx[i]]);
    }

    for(int i = 0; i < maxn; ++i) {
        f[i] = to_mont(f[i], ctx);
    }

    for(int len = 2; len <= maxn; len <<= 1){
        int mid = len >> 1;
        ll W = _pow(G, (Mod - 1) / len, Mod);
        ll mont_W = to_mont(W, ctx);
        
        // 预计算W^2用于SIMD
        ll mont_W2 = mont_mul(mont_W, mont_W, ctx);
        int64x2_t mont_W2_vec = vdupq_n_s64(mont_W2);

        #pragma omp parallel for
        for(int l = 0; l < maxn; l += len){
            ll mont_w1 = ctx.mont_r; 
            ll mont_w2 = mont_mul(mont_w1, mont_W, ctx);
            
            // SIMD一次处理2个蝶形运算
            int i;
            for(i = l; i + 1 < l + mid; i += 2) {
                alignas(16) int64_t w_vals[2] = {mont_w1, mont_w2};
                int64x2_t mont_w_vec = vld1q_s64(w_vals);
                
                int64x2_t mont_a_vec = vld1q_s64(reinterpret_cast<const int64_t*>(&f[i]));
                int64x2_t f_mid_vec = vld1q_s64(reinterpret_cast<const int64_t*>(&f[i + mid]));
                
                int64x2_t mont_b_vec = simd_mont_mul(mont_w_vec, f_mid_vec, ctx);
                
                int64x2_t res_low = simd_mont_add(mont_a_vec, mont_b_vec, ctx);
                int64x2_t res_high = simd_mont_sub(mont_a_vec, mont_b_vec, ctx);
                
                vst1q_s64(reinterpret_cast<int64_t*>(&f[i]), res_low);
                vst1q_s64(reinterpret_cast<int64_t*>(&f[i + mid]), res_high);

                mont_w_vec = simd_mont_mul(mont_w_vec, mont_W2_vec, ctx);
                
                alignas(16) int64_t updated_w[2];
                vst1q_s64(updated_w, mont_w_vec);
                mont_w1 = updated_w[0];
                mont_w2 = updated_w[1];
            }
            
            // 处理剩余的元素
            ll mont_w = mont_w1;
            for(int j = i; j < l + mid; ++j) {
                ll mont_a = f[j];
                ll mont_b = mont_mul(mont_w, f[j + mid], ctx);
                
                f[j] = mont_add(mont_a, mont_b, ctx);
                f[j + mid] = mont_sub(mont_a, mont_b, ctx);
                
                mont_w = mont_mul(mont_w, mont_W, ctx);
            }
        }
    }

    if(type == 1){
        std::reverse(f + 1, f + maxn);
        ll inv_n = _pow(maxn, Mod - 2, Mod);
        ll mont_inv_n = to_mont(inv_n, ctx);
        
        // SIMD优化的逆变换
        int i;
        int64x2_t mont_inv_vec = vdupq_n_s64(mont_inv_n);
        for(i = 0; i + 1 < maxn; i += 2) {
            int64x2_t f_vec = vld1q_s64(reinterpret_cast<const int64_t*>(&f[i]));
            int64x2_t result_vec = simd_mont_mul(f_vec, mont_inv_vec, ctx);
            vst1q_s64(reinterpret_cast<int64_t*>(&f[i]), result_vec);
        }
        for(; i < maxn; ++i) {
            f[i] = mont_mul(f[i], mont_inv_n, ctx);
        }
    }

    for(int i = 0; i < maxn; ++i) {
        f[i] = from_mont(f[i], ctx);
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
        
        // 初始化Montgomery
        for(int i = 0; i < 4; i++) {
            mont_ctx[i].init(MODS[i]);
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
ll mod_results[4][N << 2];

void parallel_crt_ntt_openmp(ll *a, ll *b, ll *result, int maxn) {
    omp_set_num_threads(4);
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        ll current_mod = MODS[thread_id];
            
        std::vector<ll> local_a(maxn);
        std::vector<ll> local_b(maxn);
            
        // 复制数据并取模
        for(int i = 0; i < maxn; ++i) {
            local_a[i] = a[i] % current_mod;
            local_b[i] = b[i] % current_mod;
        }
            
        NTT(local_a.data(), current_mod, 0, maxn);
        NTT(local_b.data(), current_mod, 0, maxn);

        // 点乘使用SIMD优化
        int i;
        for(i = 0; i + 1 < maxn; i += 2) {
            mod_results[thread_id][i] = (local_a[i] * local_b[i]) % current_mod;
            mod_results[thread_id][i + 1] = (local_a[i + 1] * local_b[i + 1]) % current_mod;
        }
        for(; i < maxn; ++i) {
            mod_results[thread_id][i] = (local_a[i] * local_b[i]) % current_mod;
        }
            
        NTT(mod_results[thread_id], current_mod, 1, maxn);
    }
    
    // 串行 CRT 重构
    for(int i = 0; i < maxn; ++i) {
        ll remainders[4];
        for(int j = 0; j < 4; ++j) {
            remainders[j] = mod_results[j][i];
        }
        result[i] = crt_solver.reconstruct(remainders, mod);
    }
}

int P, nm;

void poly_multiply(ll *a, ll *b, ll *ab, int n, ll p){
    nm = n + n - 1;
    MAXN = 1, P = 0;
    mod = p;
    while(MAXN < nm) MAXN <<= 1, P++;
        
    for(int i = 0; i < MAXN; ++i) {
        idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
    }

    memset(a + n, 0, sizeof(ll) * (MAXN - n));
    memset(b + n, 0, sizeof(ll) * (MAXN - n));

    parallel_crt_ntt_openmp(a, b, ab, MAXN);
}

ll a[N << 2], b[N << 2], ab[N << 2];

int main(int argc, char *argv[])
{
    int test_begin = 0;
    int test_end = 4;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_;
        ll p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        fWrite(ab, n_, i);
    }
    return 0;
}

/*
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 128.697 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 127.936 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 124.995 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 1337006139375617 : 126.82 (us) 
*/