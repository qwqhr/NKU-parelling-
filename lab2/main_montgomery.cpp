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

unsigned int inv_Mod; // N' such that Mod * N' = -1 (mod R)
const unsigned long long r= 1ll << 32;
int mont_r_mod_n;
int mont_r2_mod_n;         // R^2 mod Mod(用于转换到蒙哥马利域)

inline int REDC(unsigned long long T) {
    // m = (T mod R * N') mod R
    // m = (T * N')
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
        
        for(int l = 0; l < MAXN; l += len){
            int mont_w = mont_r_mod_n;

            for(int i = l; i < l + mid ; ++i){
                //int a = f[i], b = (1ll * w * f[i + mid]) % Mod;
                //f[i] = add_mod(a, b), f[i + mid] = sub_mod(a, b);
                //w = (1ll * w * W) % Mod;
                
                int mont_a = f[i];
                int mont_b = MontMul(mont_w, f[i + mid]);

                f[i] = add_mod(mont_a, mont_b);
                f[i + mid] = sub_mod(mont_a, mont_b);

                mont_w = MontMul(mont_w, mont_W);
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
average latency for n = 4 p = 7340033 : 0.04533 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 97.3844 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 97.146 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 97.0815 (us) 
*/