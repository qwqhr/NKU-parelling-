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
#define ll long long

// 可以自行添加需要的头文件

const int N = 300050;
const int G = 3;
ll mod;

const ll MODS[4] = {469762049, 998244353, 1004535809, 1224736769};

void fRead(ll *a, ll *b, int *n, ll *p, int input_id){
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

void fCheck(ll *ab, int n, int input_id){
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

// 扩展欧几里得算法
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

// 求逆元
ll inv(ll a, ll mod) {
    ll x, y;
    exgcd(a, mod, x, y);
    return (x % mod + mod) % mod;
}

int idx[N << 2], MAXN;

// 单模数 NTT
void NTT(ll *f, ll Mod, int type, int maxn) {

    for(int i = 0; i < maxn; ++i) {
        if(i < idx[i]) std::swap(f[i], f[idx[i]]);
    }

    for(int len = 2; len <= maxn; len <<= 1){
        int mid = len >> 1;
        ll W = _pow(G, (Mod - 1) / len, Mod);

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

        return (result % p + p) % p; // 确保结果非负
    }
};

// CRT线程数据结构
struct CRT_ThreadData {
    ll *a, *b, *result; // 每个线程的本地工作数组
    ll mod;                  // 当前线程负责的模数
    int maxn;                // 多项式最大长度
    bool work_ready;         // 工作准备标志
    bool should_exit;        // 退出标志
    pthread_mutex_t mutex;   // 线程锁
    pthread_cond_t cond;     // 条件变量
};

// 全局线程池变量
pthread_t crt_threads[4];
CRT_ThreadData crt_thread_data[4];
pthread_barrier_t crt_barrier;

// CRT工作线程函数
void* crt_worker_thread(void* arg) {
    CRT_ThreadData* data = (CRT_ThreadData*)arg;
    
    while (true) {
        // 等待工作分配
        pthread_mutex_lock(&data->mutex);
        while (!data->work_ready && !data->should_exit) {
            pthread_cond_wait(&data->cond, &data->mutex);
        }
        
        if (data->should_exit) {
            pthread_mutex_unlock(&data->mutex);
            break;
        }
        
        pthread_mutex_unlock(&data->mutex);
        
        ll mod = data->mod;
        int maxn = data->maxn;
        data->work_ready = false;
        
        NTT(data->a, mod, 0, maxn);  
        NTT(data->b, mod, 0, maxn);  
        
        for (int i = 0; i < maxn; ++i) {
            data->result[i] = (1ll * data->a[i] * data->b[i]) % mod;
        }
        
        NTT(data->result, mod, 1, maxn);
        
        // 等待所有线程完成
        pthread_barrier_wait(&crt_barrier);
    }
    
    return nullptr;
}
 
// 初始化静态 CRT 线程池
void init_crt_threads() {
    pthread_barrier_init(&crt_barrier, nullptr, 4 + 1);
    
    for(int i = 0; i < 4; ++i) {
        crt_thread_data[i].mod = MODS[i];
        crt_thread_data[i].work_ready = false;
        crt_thread_data[i].should_exit = false;
        pthread_mutex_init(&crt_thread_data[i].mutex, nullptr);
        pthread_cond_init(&crt_thread_data[i].cond, nullptr);
        
        // 为每个线程分配独立的工作数组
        crt_thread_data[i].a = new ll[N << 2];     
        crt_thread_data[i].b = new ll[N << 2];      
        crt_thread_data[i].result = new ll[N << 2];

        // 创建线程
        pthread_create(&crt_threads[i], nullptr, crt_worker_thread, &crt_thread_data[i]);
    }
}

// 销毁 CRT 线程池
void destroy_crt_threads() {
    // 通知所有线程退出
    for(int i = 0; i < 4; ++i) {
        pthread_mutex_lock(&crt_thread_data[i].mutex);
        crt_thread_data[i].should_exit = true;
        pthread_cond_signal(&crt_thread_data[i].cond);
        pthread_mutex_unlock(&crt_thread_data[i].mutex);
    }
    
    // 等待所有线程结束并清理资源
    for(int i = 0; i < 4; ++i) {
        pthread_join(crt_threads[i], nullptr);
        pthread_mutex_destroy(&crt_thread_data[i].mutex);
        pthread_cond_destroy(&crt_thread_data[i].cond);
        delete[] crt_thread_data[i].a;     
        delete[] crt_thread_data[i].b;      
        delete[] crt_thread_data[i].result;
    }
    pthread_barrier_destroy(&crt_barrier);
}

CRT_Solver crt_solver;

// 并行CRT NTT函数 - 使用静态线程池并行计算4个模数
void assign_work_to_threads(ll *a, ll *b, ll *result, int maxn) {
    // 分配工作给4个CRT线程（每个线程负责一个模数）
    for(int i = 0; i < 4; ++i) {
        pthread_mutex_lock(&crt_thread_data[i].mutex);
        memcpy(crt_thread_data[i].a, a, sizeof(ll) * maxn);
        memcpy(crt_thread_data[i].b, b, sizeof(ll) * maxn);
        crt_thread_data[i].maxn = maxn;
        crt_thread_data[i].work_ready = true;
        pthread_cond_signal(&crt_thread_data[i].cond);
        pthread_mutex_unlock(&crt_thread_data[i].mutex);
    }
    
    // 等待所有线程完成计算
    pthread_barrier_wait(&crt_barrier);
    
    // 主线程进行CRT重构
    for(int i = 0; i < maxn; ++i) {
        ll remainders[4];
        for(int j = 0; j < 4; ++j) {
            remainders[j] = crt_thread_data[j].result[i];
        }
        result[i] = crt_solver.reconstruct(remainders, mod);
        //std::cout<<result[i]<<" ";
    }
}

int P, nm;

void poly_multiply(ll *a, ll *b, ll *ab, int n, ll p){
    nm = n + n - 1;
    MAXN = 1, P = 0;
    mod = p;
    while(MAXN < nm) MAXN <<= 1, P++; // p表示MAXN是2的几次幂
        
    for(int i = 0; i < MAXN; ++i) {
        idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
    }

    memset(a + n, 0, sizeof(ll) * (MAXN - n));
    memset(b + n, 0, sizeof(ll) * (MAXN - n));

    assign_work_to_threads(a, b, ab, MAXN);
   // for(int i = 0; i < MAXN; ++i) {
   //     std::cout<<ab[i]<<" ";
   // }
}

ll a[N << 2], b[N << 2], ab[N << 2];

int main(int argc, char *argv[])
{
    
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    init_crt_threads();

    int test_begin = 0;
    int test_end = 4;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_;
        ll p_;
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

    destroy_crt_threads();
    return 0;
}

/*
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.143871 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 138.88 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 134.842 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 133.351 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 1337006139375617 : 135.005 (us) 
*/