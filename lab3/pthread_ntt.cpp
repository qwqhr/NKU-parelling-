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
#define ll long long

const int N = 300050;
const int G = 3;

void fRead(int *a, int *b, int *n, int *p, int input_id) {
    // 数据输入函数
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) {
        fin >> a[i];
    }
    for (int i = 0; i < *n; i++) {   
        fin >> b[i];
    }
}

void fCheck(int *ab, int n, int input_id) {
    // 判断多项式乘法结果是否正确
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++) {
        int x;
        fin >> x;
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误" << std::endl;
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id) {
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++) {
        fout << ab[i] << '\n';
    }
}

ll _pow(ll a, ll p, ll Mod) {
    ll ans = 1, mul = a;
    while (p) {
        if (p & 1) ans = (ans * mul) % Mod;
        mul = (mul * mul) % Mod;
        p >>= 1;
    }
    return ans;
}

int idx[N << 2], MAXN;

int num_threads = 4; // 默认线程数为 4

pthread_t* threads = nullptr;
pthread_barrier_t barrier;

struct ThreadData {
    int thread_id;
    int *f;
    int Mod;
    int len;
    ll W;
    bool should_exit; // 退出线程
    pthread_mutex_t mutex; // 线程锁
    pthread_cond_t cond; // 条件变量
    bool work_ready;
};

ThreadData* thread_data = nullptr;

void* ntt_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int tid = data->thread_id;
    
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
        
        // 提前解锁，后面操作部分无需加锁
        pthread_mutex_unlock(&data->mutex);
        
        
        // 执行跨步分配的工作
        int *f = data->f;
        int Mod = data->Mod;
        int len = data->len;
        int mid = len >> 1;
        ll W = data->W;
        
        data->work_ready = false;

        // 跨步分配
        for (int l = tid * len; l < MAXN; l += num_threads * len) {
            ll w = 1;
            for (int i = l; i < l + mid; ++i) {
                int a = f[i], b = (1ll * w * f[i + mid]) % Mod;
                f[i] = (a + b) % Mod;
                f[i + mid] = (a - b + Mod) % Mod;
                w = (1ll * w * W) % Mod;
            }
        }

        // 等待所有线程完成当前轮次
        pthread_barrier_wait(&barrier);
    }
    
    return nullptr;
}

// 初始化线程池
void init_thread_pool() {
    if (threads != nullptr) return;
    
    threads = new pthread_t[num_threads];
    thread_data = new ThreadData[num_threads];
    
    // 初始化barrier
    pthread_barrier_init(&barrier, nullptr, num_threads + 1); // +1是为了主线程也参与barrier
    
    // 创建线程
    for (int i = 0; i < num_threads; ++i) {
        thread_data[i].thread_id = i;
        thread_data[i].should_exit = false;
        thread_data[i].work_ready = false;
        pthread_mutex_init(&thread_data[i].mutex, nullptr);
        pthread_cond_init(&thread_data[i].cond, nullptr);
        
        pthread_create(&threads[i], nullptr, ntt_thread, &thread_data[i]);
    }
}

void destroy_thread_pool() {
    if (threads == nullptr) return;
    
    // 通知所有线程退出
    for (int i = 0; i < num_threads; ++i) {
        pthread_mutex_lock(&thread_data[i].mutex);
        thread_data[i].should_exit = true;
        pthread_cond_signal(&thread_data[i].cond);
        pthread_mutex_unlock(&thread_data[i].mutex);
    }
    
    // 等待所有线程结束
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
        pthread_mutex_destroy(&thread_data[i].mutex);
        pthread_cond_destroy(&thread_data[i].cond);
    }
    
    pthread_barrier_destroy(&barrier);
    delete[] threads;
    delete[] thread_data;
    threads = nullptr;
    thread_data = nullptr;
}

void assign_work_to_threads(int *f, int Mod, int len, ll W) {
    for (int i = 0; i < num_threads; ++i) {
        pthread_mutex_lock(&thread_data[i].mutex);
        thread_data[i].f = f;
        thread_data[i].Mod = Mod;
        thread_data[i].len = len;
        thread_data[i].W = W;
        thread_data[i].work_ready = true;
        pthread_cond_signal(&thread_data[i].cond);
        pthread_mutex_unlock(&thread_data[i].mutex);
    }
    
    // 等待所有线程完成
    pthread_barrier_wait(&barrier);
}

void NTT(int *f, int Mod, int type) {
    for (int i = 0; i < MAXN; ++i) {
        if (i < idx[i]) std::swap(f[i], f[idx[i]]);
    }
    
    for (int len = 2; len <= MAXN; len <<= 1) {
        int mid = len >> 1;
        ll W = _pow(G, (Mod - 1) / len, Mod);
        
        // 分配工作给线程池
        assign_work_to_threads(f, Mod, len, W);
    }
    
    if (type == 1) {
        std::reverse(f + 1, f + MAXN);
        ll inv = _pow(MAXN, Mod - 2, Mod);
        for (int i = 0; i < MAXN; ++i) {
            f[i] = (f[i] * inv) % Mod;
        }
    }
}

int P, nm;

void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    nm = n + n - 1;
    MAXN = 1, P = 0;
    while (MAXN < nm) {
        MAXN <<= 1;
        P++; // p表示MAXN是2的几次幂
    }
    
    for (int i = 0; i < MAXN; ++i) {
        idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
    }
    
    memset(a + n, 0, sizeof(int) * (MAXN - n));
    memset(b + n, 0, sizeof(int) * (MAXN - n));
    
    NTT(a, p, 0);
    NTT(b, p, 0);
    
    for (int i = 0; i < MAXN; ++i) {
        ab[i] = (1ll * a[i] * b[i]) % p;
    } 
    
    NTT(ab, p, 1);
}

int a[N << 2], b[N << 2], ab[N << 1];

int main(int argc, char *argv[]) {
    // 解析命令行参数设置线程数
    if (argc > 1) {
        num_threads = std::atoi(argv[1]);
        if (num_threads <= 0) {
            num_threads = 4; // 默认值
        }
    }
    
    
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1   

    init_thread_pool();

    int test_begin = 0;
    int test_end = 3;
    for (int i = test_begin; i <= test_end; ++i) {
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::ratio<1, 1000>> elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us) " << std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
        fWrite(ab, n_, i);
    }

    destroy_thread_pool();
    return 0;
}

/*
多项式乘法结果正确
average latency for n = 4 p = 7340033 : 0.643015 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 37.1163 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 34.3746 (us) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 35.3475 (us) 
*/