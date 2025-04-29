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
#include <arm_neon.h>

#define ll long long

// 可以自行添加需要的头文件

const int N = 300050;
const int G = 3;

int Mod;

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

int _pow(int a, int p){
    int ans = 1, mul = a;
    while(p){
        if(p & 1) ans = (1ll * ans * mul) % Mod;
        mul = (1ll * mul * mul) % Mod;
        p >>= 1;
    }
    return ans;
}

int idx[N << 2], MAXN;

inline int add_mod(int x, int y){
    if(x + y >= Mod) return x + y - Mod;
    return x + y;
}

inline int sub_mod(int x, int y){
    if (x - y < 0) return x - y + Mod;
    return x - y;
}

void NTT(int *f, int type){

    for(int i = 0; i < MAXN; ++i) if(i < idx[i]) std::swap(f[i], f[idx[i]]);

    
    for(int len = 2; len <= MAXN; len <<= 1){
        int mid = len >> 1;
        int Wn = _pow(G, (Mod - 1) / len); 

        for(int l = 0; l < MAXN; l += len){
            int w = 1; 

            int i = l; 

            int Wn2 = (1ll * Wn * Wn) % Mod;
            int Wn3 = (1ll * Wn2 * Wn) % Mod;
            int Wn4 = (1ll * Wn3 * Wn) % Mod; // Wn^4, 四个数一组进行并行化

            int32x4_t mod_vec = vdupq_n_s32(Mod);// 模向量
            int32x4_t zero_vec = vdupq_n_s32(0);// 零向量

            for (; i < l + mid - 3; i += 4) {
                int32x4_t a_vec = vld1q_s32(&f[i]);// f向量

                int w0 = w;
                int w1 = (1ll * w0 * Wn) % Mod;
                int w2 = (1ll * w1 * Wn) % Mod;
                int w3 = (1ll * w2 * Wn) % Mod;

                int b0 = (1ll * w0 * f[i + mid + 0]) % Mod;
                int b1 = (1ll * w1 * f[i + mid + 1]) % Mod;
                int b2 = (1ll * w2 * f[i + mid + 2]) % Mod;
                int b3 = (1ll * w3 * f[i + mid + 3]) % Mod;


                int b_vals[4] = {b0, b1, b2, b3};
                int32x4_t b_vec = vld1q_s32(b_vals);

                int32x4_t sum = vaddq_s32(a_vec, b_vec);
                uint32x4_t mask_add = vcgeq_s32(sum, mod_vec);
                int32x4_t sub_val = vsubq_s32(sum, mod_vec);
                int32x4_t res_add = vbslq_s32(mask_add, sub_val, sum);

                int32x4_t diff = vsubq_s32(a_vec, b_vec);
                uint32x4_t mask_sub = vcltq_s32(diff, zero_vec);
                int32x4_t add_val = vaddq_s32(diff, mod_vec);
                int32x4_t res_sub = vbslq_s32(mask_sub, add_val, diff);

                vst1q_s32(&f[i], res_add);
                vst1q_s32(&f[i + mid], res_sub);

                w = (1ll * w * Wn4) % Mod;
            }


            // Tail Loop
            for (; i < l + mid; ++i) {
                int a_scalar = f[i];
                int b_scalar = (1ll * w * f[i + mid]) % Mod;
                f[i] = add_mod(a_scalar, b_scalar);
                f[i + mid] = sub_mod(a_scalar, b_scalar);
                w = (1ll * w * Wn) % Mod;
            }
        }
    }

    if(type == 1){
        std::reverse(f + 1, f + MAXN);
        int inv = _pow(MAXN, Mod - 2);
        for(int i = 0; i < MAXN; ++i) f[i] = (1ll * f[i] * inv) % Mod;
    }
}

int P, nm;

void poly_multiply(int *a, int *b, int *ab, int n, int p){
    nm = n + n - 1, Mod = p;
    MAXN = 1, P = 0;
    while(MAXN < nm) MAXN <<= 1, P++;//p表示MAXN是2的几次幂
        
    for(int i = 0; i < MAXN; ++i) idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
    
    memset(a + n, 0, sizeof(int) * (MAXN - n));
    memset(b + n, 0, sizeof(int) * (MAXN - n));

    NTT(a, 0), NTT(b, 0);
    
    for(int i = 0; i < MAXN; ++i){
        ab[i] = (1ll * a[i] * b[i]) % p;
    } 
    
    NTT(ab, 1);
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
average latency for n = 4 p = 7340033 : 0.01388 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 64.5372 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 71.7735 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 75.0975 (us) 
*/