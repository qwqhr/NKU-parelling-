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
const int MAX_TESTS = 5;  

// Barrett约简结构体
struct BarrettReduction {
    ll mod;
    ll mu;  // μ = ⌊2^(2k)/mod⌋
    int k;  // mod的位数
    
    __host__ __device__ BarrettReduction() {}
    
    __host__ __device__ BarrettReduction(ll m) {
        mod = m;
        k = 64;  
        mu = ((__int128)1 << k) / m;
    }
    
    __host__ __device__ ll mul(ll a, ll b) {
        ll ab = a * b;
        ll q = ((__int128)mu * ab) >> k;  
        ll r = ab - q * mod;
        return r >= mod ? r - mod : r;
    }
};

static ll* d_roots_cache[MAX_TESTS];
static int max_n_cache[MAX_TESTS];
static BarrettReduction* d_barrett_cache[MAX_TESTS];
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

ll _pow(ll a, ll p, ll Mod){
    ll ans = 1, mul = a;
    while(p){
        if(p & 1) ans = (ans * mul) % Mod;
        mul = (mul * mul) % Mod;
        p >>= 1;
    }
    return ans;
}

void compute_roots_for_id(int id, int mod, int maxn) {
    if (computed[id] && max_n_cache[id] >= maxn) {
        return;
    }
    
    if (computed[id]) {
        CUDA_CHECK(cudaFree(d_roots_cache[id]));
        CUDA_CHECK(cudaFree(d_barrett_cache[id]));
    }
    
    int total_roots_needed = 0;
    for (int len = 2; len <= maxn; len <<= 1) {
        total_roots_needed += len >> 1;
    }
    
    ll* d_roots;
    BarrettReduction* d_barrett;
    CUDA_CHECK(cudaMalloc(&d_roots, total_roots_needed * sizeof(ll)));
    CUDA_CHECK(cudaMalloc(&d_barrett, sizeof(BarrettReduction)));

    ll* h_roots = (ll*)malloc(total_roots_needed * sizeof(ll));
    BarrettReduction h_barrett(mod);
    
    int offset = 0;
    
    for (int len = 2; len <= maxn; len <<= 1) {
        ll W = _pow(G, (mod - 1) / len, mod);
        int mid = len >> 1;
        
        ll current_w = 1;
        for (int k = 0; k < mid; k++) {
            h_roots[offset + k] = current_w;
            current_w = h_barrett.mul(current_w, W);  // 使用Barrett约简
        }
        offset += mid;
    }
    
    CUDA_CHECK(cudaMemcpy(d_roots, h_roots, total_roots_needed * sizeof(ll), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_barrett, &h_barrett, sizeof(BarrettReduction), cudaMemcpyHostToDevice));
    
    d_roots_cache[id] = d_roots;
    d_barrett_cache[id] = d_barrett;
    max_n_cache[id] = maxn;
    computed[id] = true;
    
    free(h_roots);
}

void cleanup_all_roots() {
    for (int i = 0; i < MAX_TESTS; i++) {
        if (computed[i]) {
            CUDA_CHECK(cudaFree(d_roots_cache[i]));
            CUDA_CHECK(cudaFree(d_barrett_cache[i]));
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
                                     BarrettReduction* barrett, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int mid = len >> 1;
    int blocks_per_len = n / len;
    
    if (tid < blocks_per_len * mid) {
        int block_id = tid / mid;
        int in_block_id = tid % mid;
        
        int base = block_id * len;
        int i = base + in_block_id;
        int j = i + mid;
        
        ll w = roots[roots_offset + in_block_id];
        
        int u = data[i];
        int v = barrett->mul(w, data[j]);  
        data[i] = (u + v) % barrett->mod;
        data[j] = (u - v + barrett->mod) % barrett->mod;
    }
}

__global__ void pointwise_mul_kernel(int* a, int* b, int* result, int n, BarrettReduction* barrett) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = barrett->mul(a[i], b[i]);  
    }
}

__global__ void final_scale_kernel(int* data, ll inv, int n, BarrettReduction* barrett) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = barrett->mul(data[i], inv);  
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
                                                  d_barrett_cache[id], maxn);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        roots_offset += mid;
    }
    
    if(type == 1) {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, maxn * sizeof(int), cudaMemcpyDeviceToHost));
        manual_reverse(h_data, 1, maxn - 1);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, maxn * sizeof(int), cudaMemcpyHostToDevice));
        
        ll inv = _pow(maxn, mod - 2, mod);
        final_scale_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, inv, maxn, d_barrett_cache[id]);
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
    
    memset(a + n, 0, sizeof(int) * (MAXN - n));
    memset(b + n, 0, sizeof(int) * (MAXN - n));
    
    cuda_NTT(a, idx, id, p, 0, MAXN);
    cuda_NTT(b, idx, id, p, 0, MAXN);
    
    int* d_a, *d_b, *d_ab;
    CUDA_CHECK(cudaMalloc(&d_a, MAXN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, MAXN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ab, MAXN * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_a, a, MAXN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, MAXN * sizeof(int), cudaMemcpyHostToDevice));
    
    int grid = (MAXN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pointwise_mul_kernel<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_ab, MAXN, d_barrett_cache[id]);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(ab, d_ab, MAXN * sizeof(int), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_ab));
    
    cuda_NTT(ab, idx, id, p, 1, MAXN);
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
        poly_multiply(a, b, ab, n_, p_, i);  // 传入测试ID
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
average latency for n = 131072 p = 7340033 : 11.7833 (ms) 
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 11.3955 (ms) 
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 12.5383 (ms) 
*/