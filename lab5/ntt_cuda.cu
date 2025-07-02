#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#define ll long long

const int N = 300050;
const int G = 3;
const int BLOCK_SIZE = 512;
const int MAX_TESTS = 5; 

static ll* d_roots_cache[MAX_TESTS];
static int max_n_cache[MAX_TESTS];
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
    }
    
    int total_roots_needed = 0;
    for (int len = 2; len <= maxn; len <<= 1) {
        total_roots_needed += len >> 1;
    }
    
    ll* d_roots;
    CUDA_CHECK(cudaMalloc(&d_roots, total_roots_needed * sizeof(ll)));
    
    ll* h_roots = (ll*)malloc(total_roots_needed * sizeof(ll));
    
    int offset = 0;
    

    for (int len = 2; len <= maxn; len <<= 1) {
        ll W = _pow(G, (mod - 1) / len, mod);
        int mid = len >> 1;
        
        ll current_w = 1;
        for (int k = 0; k < mid; k++) {
            h_roots[offset + k] = current_w;
            current_w = (current_w * W) % mod;
        }
        offset += mid;
    }
    
    CUDA_CHECK(cudaMemcpy(d_roots, h_roots, total_roots_needed * sizeof(ll), cudaMemcpyHostToDevice));
    
    d_roots_cache[id] = d_roots;
    max_n_cache[id] = maxn;
    computed[id] = true;
    
    free(h_roots);
}

void cleanup_all_roots() {
    for (int i = 0; i < MAX_TESTS; i++) {
        if (computed[i]) {
            CUDA_CHECK(cudaFree(d_roots_cache[i]));
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

__global__ void ntt_butterfly_kernel(int* data, int len, int mod, ll* roots, int roots_offset, int n) {
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
        int v = (w * data[j]) % mod;
        data[i] = (u + v) % mod;
        data[j] = (u - v + mod) % mod;
    }
}

__global__ void pointwise_mul_kernel(int* a, int* b, int* result, int n, int mod) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = ((ll)a[i] * b[i]) % mod;
    }
}

__global__ void final_scale_kernel(int* data, ll inv, int n, int mod) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = (data[i] * inv) % mod;
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
        
        ntt_butterfly_kernel<<<grid, BLOCK_SIZE>>>(d_data, len, mod, 
                                                  d_roots_cache[id], 
                                                  roots_offset, maxn);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        roots_offset += mid;
    }
    
    if(type == 1) {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, maxn * sizeof(int), cudaMemcpyDeviceToHost));
        manual_reverse(h_data, 1, maxn - 1);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, maxn * sizeof(int), cudaMemcpyHostToDevice));
        
        ll inv = _pow(maxn, mod - 2, mod);
        final_scale_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, inv, maxn, mod);
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
    pointwise_mul_kernel<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_ab, MAXN, p);
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
        printf("average latency for n = %d p = %d : %Lf (ms)\n", n_, p_, ans);
    }
    
    cleanup_all_roots();
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

/*
32
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 17.328112 (ms)
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 16.698695 (ms)
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 16.841276 (ms)

64
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 16.965059 (ms)
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 16.783552 (ms)
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 16.558547 (ms)

128
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 17.071224 (ms)
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 15.379213 (ms)
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 14.337067 (ms)

256
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 17.965363 (ms)
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 14.355205 (ms)
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 15.899327 (ms)

512
多项式乘法结果正确
average latency for n = 131072 p = 7340033 : 17.789513 (ms)
多项式乘法结果正确
average latency for n = 131072 p = 104857601 : 14.523976 (ms)
多项式乘法结果正确
average latency for n = 131072 p = 469762049 : 16.861739 (ms)
*/