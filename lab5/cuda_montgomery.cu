#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>
#define ll long long

const int N = 300050;
const int G = 3;
const int BLOCK_SIZE = 256;
const int MAX_TESTS = 10;

// Montgomery约简结构体
struct MontgomeryReduction {
    int mod;
    unsigned int inv_mod;      // mod * inv_mod = -1 (mod R)
    int mont_r_mod_n;          // R mod mod
    int mont_r2_mod_n;         // R^2 mod mod
    static const unsigned long long R = 1ULL << 32;
    
    __host__ __device__ MontgomeryReduction() {}
    
    __host__ MontgomeryReduction(int m) {
        mod = m;
        
        // 计算 inv_mod = -mod^{-1} mod R
        ll x, y;
        exgcd(mod, R, x, y);
        inv_mod = (unsigned int)(-x);
        
        mont_r_mod_n = (int)(R % mod);
        mont_r2_mod_n = (int)((1ULL * mont_r_mod_n * mont_r_mod_n) % mod);
    }
    
    __host__ ll exgcd(ll a, ll b, ll &x, ll &y) {
        if (b == 0) {
            x = 1, y = 0;
            return a; 
        }
        ll x1, y1;
        ll d = exgcd(b, a % b, x1, y1); 
        x = y1, y = x1 - (a / b) * y1;
        return d; 
    }
    
    __host__ __device__ inline int REDC(unsigned long long T) {
        unsigned int m = (unsigned int)T * inv_mod;
        unsigned long long t_long = (T + (unsigned long long)m * mod) >> 32;
        
        int result;
        if (t_long >= mod) result = (int)(t_long - mod);
        else result = (int)t_long;
        
        return result;
    }
    
    __host__ __device__ inline int to_mont(int x) {
        return REDC((unsigned long long)x * mont_r2_mod_n);
    }

    __host__ __device__ inline int from_mont(int mont_x) {
        return REDC((unsigned long long)mont_x);
    }
    
    __host__ __device__ inline int mul(int mont_a, int mont_b) {
        return REDC((unsigned long long)mont_a * mont_b);
    }
};

static ll* d_roots_cache[MAX_TESTS];
static int max_n_cache[MAX_TESTS];
static MontgomeryReduction* d_montgomery_cache[MAX_TESTS];
static bool computed[MAX_TESTS];

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void fRead(int *a, int *b, int *n, int *p, int input_id){
    char filename[100];
    sprintf(filename, "data/%d.in", input_id);
    FILE *fin = fopen(filename, "r");
    
    if (!fin) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return;
    }
    
    fscanf(fin, "%d %d", n, p);
    for (int i = 0; i < *n; i++) {
        fscanf(fin, "%d", &a[i]);
    }
    for (int i = 0; i < *n; i++) {
        fscanf(fin, "%d", &b[i]);
    }
    fclose(fin);
}

void fCheck(int *ab, int n, int input_id){
    char filename[100];
    sprintf(filename, "data/%d.out", input_id);
    FILE *fin = fopen(filename, "r");
    
    if (!fin) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        printf("无法验证结果\n");
        return;
    }
    
    bool correct = true;
    for (int i = 0; i < n * 2 - 1; i++) {
        int expected;
        fscanf(fin, "%d", &expected);
        if (ab[i] != expected) {
            correct = false;
            break;
        }
    }
    fclose(fin);
    
    if (correct) {
        printf("多项式乘法结果正确\n");
    } else {
        printf("多项式乘法结果错误\n");
    }
}

// CPU版Montgomery快速幂
int montgomery_pow(int a, int p, int mod) {
    MontgomeryReduction mont(mod);
    int mont_a = mont.to_mont(a);
    int mont_ans = mont.mont_r_mod_n; // 蒙哥马利域中的1
    int mont_mul = mont_a;
    
    while(p){
        if(p & 1) mont_ans = mont.mul(mont_ans, mont_mul);
        mont_mul = mont.mul(mont_mul, mont_mul);
        p >>= 1;
    }
    
    return mont.from_mont(mont_ans);
}

void compute_roots_for_id(int id, int mod, int maxn) {
    if (computed[id] && max_n_cache[id] >= maxn) {
        return;
    }
    
    if (computed[id]) {
        CUDA_CHECK(cudaFree(d_roots_cache[id]));
        CUDA_CHECK(cudaFree(d_montgomery_cache[id]));
    }

    int total_roots_needed = 0;
    for (int len = 2; len <= maxn; len <<= 1) {
        total_roots_needed += len >> 1;
    }
    
    ll* d_roots;
    MontgomeryReduction* d_montgomery;
    CUDA_CHECK(cudaMalloc(&d_roots, total_roots_needed * sizeof(ll)));
    CUDA_CHECK(cudaMalloc(&d_montgomery, sizeof(MontgomeryReduction)));

    ll* h_roots = (ll*)malloc(total_roots_needed * sizeof(ll));
    MontgomeryReduction h_montgomery(mod);
    
    int offset = 0;
    
    // 预计算旋转因子
    for (int len = 2; len <= maxn; len <<= 1) {
        int W = montgomery_pow(G, (mod - 1) / len, mod);
        int mont_W = h_montgomery.to_mont(W);
        int mid = len >> 1;
        
        int mont_current_w = h_montgomery.mont_r_mod_n; // 蒙哥马利域中的1
        for (int k = 0; k < mid; k++) {
            h_roots[offset + k] = mont_current_w;
            mont_current_w = h_montgomery.mul(mont_current_w, mont_W);
        }
        offset += mid;
    }
    
    CUDA_CHECK(cudaMemcpy(d_roots, h_roots, total_roots_needed * sizeof(ll), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_montgomery, &h_montgomery, sizeof(MontgomeryReduction), cudaMemcpyHostToDevice));
    
    // 更新缓存
    d_roots_cache[id] = d_roots;
    d_montgomery_cache[id] = d_montgomery;
    max_n_cache[id] = maxn;
    computed[id] = true;
    
    free(h_roots);
}

void cleanup_all_roots() {
    for (int i = 0; i < MAX_TESTS; i++) {
        if (computed[i]) {
            CUDA_CHECK(cudaFree(d_roots_cache[i]));
            CUDA_CHECK(cudaFree(d_montgomery_cache[i]));
            computed[i] = false;
        }
    }
}

__global__ void bit_reverse_kernel(int* data, int* idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && i < idx[i]) {
        int temp = data[i];
        data[i] = data[idx[i]];
        data[idx[i]] = temp;
    }
}

__global__ void ntt_butterfly_kernel(int* data, int len, ll* roots, int roots_offset, 
                                     MontgomeryReduction* montgomery, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int mid = len >> 1;
    int blocks_per_len = n / len;
    
    if (tid < blocks_per_len * mid) {
        int block_id = tid / mid;
        int in_block_id = tid % mid;
        
        int base = block_id * len;
        int i = base + in_block_id;
        int j = i + mid;
        
        int mont_w = roots[roots_offset + in_block_id];
        
        int mont_a = data[i];
        int mont_b = montgomery->mul(mont_w, data[j]);  
        
        data[i] = (mont_a + mont_b) % montgomery->mod;
        data[j] = (mont_a - mont_b + montgomery->mod) % montgomery->mod;
    }
}

__global__ void pointwise_mul_kernel(int* a, int* b, int* result, int n, MontgomeryReduction* montgomery) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = montgomery->mul(a[i], b[i]);  
    }
}

__global__ void final_scale_kernel(int* data, int mont_inv, int n, MontgomeryReduction* montgomery) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = montgomery->mul(data[i], mont_inv);  
    }
}

__global__ void to_montgomery_kernel(int* data, int n, MontgomeryReduction* montgomery) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = montgomery->to_mont(data[i]);
    }
}

__global__ void from_montgomery_kernel(int* data, int n, MontgomeryReduction* montgomery) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = montgomery->from_mont(data[i]);
    }
}

void manual_reverse(int* arr, int start, int end) {
    while (start < end) {
        int temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }
}

void cuda_NTT(int* h_data, int* h_idx, int id, int mod, int type, int maxn) {
    // 确保旋转因子已计算
    if (!computed[id] || max_n_cache[id] < maxn) {
        compute_roots_for_id(id, mod, maxn);
    }
    
    int* d_data;
    int* d_idx;
    
    CUDA_CHECK(cudaMalloc(&d_data, maxn * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_idx, maxn * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, maxn * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_idx, h_idx, maxn * sizeof(int), cudaMemcpyHostToDevice));
    
    // 位逆序
    int grid_size = (maxn + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bit_reverse_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, d_idx, maxn);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 蝶形运算
    int roots_offset = 0;
    for(int len = 2; len <= maxn; len <<= 1) {
        int mid = len >> 1;
        int total_butterflies = maxn / len * mid;
        int grid = (total_butterflies + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        ntt_butterfly_kernel<<<grid, BLOCK_SIZE>>>(d_data, len, 
                                                  d_roots_cache[id], 
                                                  roots_offset,
                                                  d_montgomery_cache[id], maxn);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        roots_offset += mid;
    }
    
    if(type == 1) {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, maxn * sizeof(int), cudaMemcpyDeviceToHost));
        manual_reverse(h_data, 1, maxn - 1);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, maxn * sizeof(int), cudaMemcpyHostToDevice));
        
        int inv = montgomery_pow(maxn, mod - 2, mod);
        MontgomeryReduction temp_mont(mod);
        int mont_inv = temp_mont.to_mont(inv);
        
        final_scale_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, mont_inv, maxn, d_montgomery_cache[id]);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    CUDA_CHECK(cudaMemcpy(h_data, d_data, maxn * sizeof(int), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_idx));
}

int idx[N << 2], MAXN, P, nm;

void poly_multiply(int *a, int *b, int *ab, int n, int p, int id) {
    nm = n + n - 1;
    MAXN = 1, P = 0;
    while(MAXN < nm) MAXN <<= 1, P++;

    for(int i = 0; i < MAXN; ++i) {
        idx[i] = (idx[i >> 1] >> 1) | ((i & 1) << (P - 1));
    }
    
    // 转换到蒙哥马利域
    if (!computed[id]) {
        compute_roots_for_id(id, p, MAXN);
    }
    
    int* d_a, *d_b, *d_ab;
    CUDA_CHECK(cudaMalloc(&d_a, MAXN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, MAXN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ab, MAXN * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_a + n, 0, (MAXN - n) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_b + n, 0, (MAXN - n) * sizeof(int)));
    
    // 转换到Montgomery域
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    to_montgomery_kernel<<<grid, BLOCK_SIZE>>>(d_a, n, d_montgomery_cache[id]);
    to_montgomery_kernel<<<grid, BLOCK_SIZE>>>(d_b, n, d_montgomery_cache[id]);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(a, d_a, MAXN * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b, d_b, MAXN * sizeof(int), cudaMemcpyDeviceToHost));
    
    cuda_NTT(a, idx, id, p, 0, MAXN);
    cuda_NTT(b, idx, id, p, 0, MAXN);
    
    // 点乘
    CUDA_CHECK(cudaMemcpy(d_a, a, MAXN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, MAXN * sizeof(int), cudaMemcpyHostToDevice));
    
    grid = (MAXN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pointwise_mul_kernel<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_ab, MAXN, d_montgomery_cache[id]);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(ab, d_ab, MAXN * sizeof(int), cudaMemcpyDeviceToHost));
    
    cuda_NTT(ab, idx, id, p, 1, MAXN);
    
    // 从Montgomery域转换回普通域
    CUDA_CHECK(cudaMemcpy(d_ab, ab, (n + n - 1) * sizeof(int), cudaMemcpyHostToDevice));
    grid = (n + n - 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    from_montgomery_kernel<<<grid, BLOCK_SIZE>>>(d_ab, n + n - 1, d_montgomery_cache[id]);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ab, d_ab, (n + n - 1) * sizeof(int), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_ab));
}

int a[N << 2], b[N << 2], ab[N << 2];

int main(int argc, char *argv[])
{
    CUDA_CHECK(cudaSetDevice(0));
    
    // 初始化数组
    memset(computed, false, sizeof(computed));
    
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        
        auto Start = std::chrono::high_resolution_clock::now();
        poly_multiply(a, b, ab, n_, p_, i);
        auto End = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (ms) "<<std::endl;
    }
    
    cleanup_all_roots();
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

/*
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 15.4873 (ms) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 15.1056 (ms) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 13.886 (ms) 
*/