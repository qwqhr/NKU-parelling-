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

// 可以自行添加需要的头文件

const int N = 300050;
const int G = 3;

void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
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
    // 判断多项式乘法结果是否正确
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

int Mod;
int idx[N << 2], MAXN;
int32x4_t mod_vec;
unsigned int inv_Mod; // N' such that Mod * N' = -1 (mod R)
const unsigned long long r= 1ll << 32;
int mont_r_mod_n;
int mont_r2_mod_n;         // R^2 mod Mod(用于转换到蒙哥马利域)


int _pow(int a, int p){
    int ans = 1, mul = a;
    while(p){
        if(p & 1) ans = (1ll * ans * mul) % Mod;
        mul = (1ll * mul * mul) % Mod;
        p >>= 1;
    }
    return ans;
}

inline int REDC(unsigned long long T) {
    // m = (T mod R * N') mod R
    unsigned int m = (unsigned int)T * inv_Mod;
    // t = (T + m * Mod) / R
    unsigned long long t_long = (T + (unsigned long long)m * Mod) >> 32;

    // 返回值应该在 [0, Mod)
    // 如果 t >= Mod, 则减去 Mod
    int result;
    if (t_long >= Mod) result = (int)(t_long - Mod); // 确保结果在 int 范围内
    else result = (int)t_long;

    return result;
}

// 转换到蒙哥马利域: x' = x * R mod Mod
inline int to_mont(int x) {
    return REDC((unsigned long long)x * mont_r2_mod_n);
}

// 从蒙哥马利域转换回普通域: x = x' * R^{-1} mod Mod
inline int from_mont(int mont_x) {
    return REDC((unsigned long long)mont_x);
}

// 蒙哥马利乘法: a_mont * b_mont * R^{-1} mod Mod
inline int MontMul(int mont_a, int mont_b) {
    return REDC((unsigned long long)mont_a * mont_b);
}


inline int32x4_t vadd(int32x4_t a, int32x4_t b) {
    int32x4_t sum = vaddq_s32(a, b);
    uint32x4_t mask = vcgeq_s32(sum, mod_vec); // 大于赋1，小于赋0
    int32x4_t correction = vandq_s32(mod_vec, vreinterpretq_s32_u32(mask)); // (sum >= Mod ? Mod : 0)
    return vsubq_s32(sum, correction);
}

inline int32x4_t vsub(int32x4_t a, int32x4_t b) {
    int32x4_t diff = vsubq_s32(a, b);
    uint32x4_t mask = vcltq_s32(diff, vdupq_n_s32(0)); 
    int32x4_t correction = vandq_s32(mod_vec, vreinterpretq_s32_u32(mask)); // (diff < 0 ? Mod : 0)
    return vaddq_s32(diff, correction);
}


inline int32x4_t vREDC(uint64x2_t T_vec_low, uint64x2_t T_vec_high) {

    // m = (T % R) * N' % R = T_low * inv_Mod
    uint32x2_t T_low_parts_low = vmovn_u64(T_vec_low);      
    uint32x2_t T_low_parts_high = vmovn_u64(T_vec_high);    
    uint32x4_t T_low = vcombine_u32(T_low_parts_low, T_low_parts_high);
    uint32x4_t m = vmulq_n_u32(T_low, inv_Mod);

    // m * N 
    unsigned int Mod_u32 = (unsigned int)Mod;
    uint64x2_t mN_low = vmull_n_u32(vget_low_u32(m), Mod_u32);
    uint64x2_t mN_high = vmull_n_u32(vget_high_u32(m), Mod_u32);

    // T + m*N
    uint64x2_t sum64_low = vaddq_u64(T_vec_low, mN_low);
    uint64x2_t sum64_high = vaddq_u64(T_vec_high, mN_high);

    // t = (T + m*N) >> 32
    uint32x2_t t_parts_low = vshrn_n_u64(sum64_low, 32);
    uint32x2_t t_parts_high = vshrn_n_u64(sum64_high, 32);
    int32x4_t t = vreinterpretq_s32_u32(vcombine_u32(t_parts_low, t_parts_high));

    // result = t - Mod if t >= Mod
    uint32x4_t mask = vcgeq_s32(t, mod_vec);
    int32x4_t correction = vandq_s32(mod_vec, vreinterpretq_s32_u32(mask));
    int32x4_t result = vsubq_s32(t, correction);

    return result;
}


inline int32x4_t vMontMul(int32x4_t mont_a, int32x4_t mont_b) {
    
    int64x2_t ab_low_s64 = vmull_s32(vget_low_s32(mont_a), vget_low_s32(mont_b));
    int64x2_t ab_high_s64 = vmull_s32(vget_high_s32(mont_a), vget_high_s32(mont_b));

    uint64x2_t ab_low_u64 = vreinterpretq_u64_s64(ab_low_s64);
    uint64x2_t ab_high_u64 = vreinterpretq_u64_s64(ab_high_s64);

    return vREDC(ab_low_u64, ab_high_u64);
}



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

void NTT(int *f, int type){
    for(int i = 0; i < MAXN; ++i) if(i < idx[i]) std::swap(f[i], f[idx[i]]);

    for(int len = 2; len <= MAXN; len <<= 1){
        int mid = len >> 1;
        int W = _pow(G, (Mod - 1) / len);
        int mont_W = to_mont(W);
        int32x4_t mont_W_vec = vdupq_n_s32(mont_W); // W向量

        int mont_W2 = MontMul(mont_W, mont_W);
        int mont_W4 = MontMul(mont_W2, mont_W2);
        int32x4_t mont_W4_vec = vdupq_n_s32(mont_W4);// W^4向量

        for(int l = 0; l < MAXN; l += len){

            int w1 = mont_r_mod_n; 
            int w2 = MontMul(w1, mont_W); 
            int w3 = MontMul(w2, mont_W); 
            int w4 = MontMul(w3, mont_W);
            int w = w1;

            alignas(16) int w_init[4] = {w1, w2, w3, w4};// 内存对齐，如果未对齐可能会出现故障
            int32x4_t mont_w_vec = vld1q_s32(w_init);

            // 一次处理4个蝶形运算
            for(int i = l; i < l + mid; i += 4, w = MontMul(w, mont_W4)) {
                if (i + 3 >= l + mid) { 
                    for (int j = i; j < l + mid; ++j) {
                        int mont_a = f[j], mont_b = MontMul(w, f[j + mid]);
                        f[j] = (mont_a + mont_b >= Mod) ? mont_a + mont_b - Mod : mont_a + mont_b;
                        f[j + mid] = (mont_a - mont_b < 0) ? mont_a - mont_b + Mod : mont_a - mont_b;
                        w = MontMul(w, mont_W);
                    }
                    break; 
                }

                int32x4_t mont_a_vec = vld1q_s32(&f[i]);
                int32x4_t f_mid_vec = vld1q_s32(&f[i + mid]);

                int32x4_t mont_b_vec = vMontMul(mont_w_vec, f_mid_vec);

                int32x4_t res_i = vadd(mont_a_vec, mont_b_vec);
                int32x4_t res_mid = vsub(mont_a_vec, mont_b_vec);

                vst1q_s32(&f[i], res_i);
                vst1q_s32(&f[i + mid], res_mid);

                mont_w_vec = vMontMul(mont_w_vec, mont_W4_vec);
            }
        }
    }

    if(type == 1){
        std::reverse(f + 1, f + MAXN);
        int inv = _pow(MAXN, Mod - 2);
        int mont_inv = to_mont(inv);
        for(int i = 0; i < MAXN; ++i) f[i] = MontMul(f[i], mont_inv);
    }
}

int P, nm;

void poly_multiply(int *a, int *b, int *ab, int n, int p){
    nm = n + n - 1, Mod = p;
    MAXN = 1, P = 0;
    mod_vec = vdupq_n_s32(Mod);//装载模数到向量寄存器中
    while(MAXN < nm) MAXN <<= 1, P++;//p表示MAXN是2的几次幂
        
    for(int i = 0; i < MAXN; ++i) idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
    
    ll tem_inv_Mod, y;
    exgcd(Mod, r, tem_inv_Mod, y);
    inv_Mod = (unsigned int)(-tem_inv_Mod);
    mont_r_mod_n = (int)(r % Mod); // R mod Mod
    mont_r2_mod_n = (int)((1ULL * mont_r_mod_n * mont_r_mod_n) % Mod);

    for(int i = 0; i < n; ++i) a[i] = to_mont(a[i]), b[i] = to_mont(b[i]);
    for(int i = n; i < MAXN; ++i) a[i] = 0, b[i] = 0;

    NTT(a, 0), NTT(b, 0);
    
    for(int i = 0; i < MAXN; ++i){
        ab[i] = MontMul(a[i], b[i]);
    } 
    
    NTT(ab, 1);

    for(int i = 0; i < n + n - 1; ++i) {
        ab[i] = from_mont(ab[i]);
    }
}

int a[N << 2], b[N << 2], ab[N << 1];

int main(int argc, char *argv[])
{
    
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
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
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
        fWrite(ab, n_, i);
    }
    return 0;
}

/*

多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.03714 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 65.2451 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 65.0911 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 65.1966 (us) 

*/
