// =====================================================================================
// benchmark_rmsnorm.cu
//
// Fused Residual-Add + RMSNorm for Llama-family decoders (TinyLlama-1.1B / Llama-3.2-1B,
// hidden size D = 2048, FP16), benchmarked on an RTX 4060 Laptop GPU (Ada, sm_89).
//
// In a Llama decoder layer, every attention / MLP sub-block ends with:
//
//     residual_out = x + residual                  (FP16 add, the new residual stream)
//     y            = weight * half( float(residual_out) * rsqrt(mean(residual_out^2) + eps) )
//
// (exactly HF `LlamaRMSNorm` semantics: statistics in FP32, cast back to FP16, then
//  multiply by the FP16 weight). There are 2 such sites per layer.
//
// Variants:
//   [A] naive      : baseline. Unfused. Scalar 16-bit element-wise add kernel, then a
//                    separate RMSNorm kernel that walks the row twice from global memory.
//   [B] vec-unfused: ablation. Same two kernels but with 128-bit vector accesses and the
//                    row cached in registers. Isolates the gain from *fusion alone*.
//   [C] fused      : optimized. One kernel, one pass over DRAM:
//                      * 128-bit vectorized loads/stores (float4 reinterpreted as 4 x half2)
//                      * warp-level reduction with __shfl_down_sync (32 lanes)
//                      * shared-memory reduction across the 8 warps of the block
//                      * residual_out is written once and never read back (no DRAM
//                        round-trip of the intermediate); row kept in registers.
//
// Ideal DRAM traffic per element (bytes): naive / vec-unfused = 5 x 2 = 10, fused = 4 x 2 = 8
// (+1 kernel launch saved). The fused kernel cannot avoid writing residual_out because the
// next layer needs it as its residual input.
//
// Build (Windows, from an "x64 Native Tools" prompt, or Linux):
//     nvcc -O3 -std=c++17 -arch=sm_89 -lineinfo benchmark_rmsnorm.cu -o benchmark_rmsnorm
// Run:
//     benchmark_rmsnorm                    (default sweep; buffers rotated so every launch
//                                           streams from DRAM, like a real forward pass)
//     benchmark_rmsnorm --graph            (time via CUDA Graph: removes CPU/WDDM launch cost)
//     benchmark_rmsnorm --hot-l2           (no rotation: data stays L2-resident across iters)
//     benchmark_rmsnorm --tokens 1,128,4096 --iters 500 --repeats 11 --warmup 100
// =====================================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d -> %s\n", #call, __FILE__,     \
                         __LINE__, cudaGetErrorString(err_));                             \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

// ------------------------------- problem constants ---------------------------------
constexpr int   kHidden  = 2048;   // TinyLlama-1.1B and Llama-3.2-1B hidden size
constexpr float kEps     = 1e-5f;  // rms_norm_eps for both models
constexpr int   kThreads = 256;    // 8 warps per row-block
constexpr int   kVec     = 8;      // halves per 128-bit access (float4 == 4 x half2)
constexpr double kRelL2Tol = 1e-3; // FP16 unit round-off is 4.9e-4

static_assert(kHidden % (kVec * kThreads) == 0, "D must be a multiple of 8 * threads");

// =====================================================================================
//                                    Device code
// =====================================================================================

// Tree reduction inside a warp with __shfl_down_sync. Result is valid in lane 0.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Block reduction: shuffle within each warp -> one partial per warp in shared memory ->
// warp 0 reduces the partials -> result broadcast to every thread through shared memory.
template <int THREADS>
__device__ __forceinline__ float block_reduce_sum(float v) {
    static_assert(THREADS % 32 == 0 && THREADS <= 1024, "bad block size");
    constexpr int kWarps = THREADS / 32;
    __shared__ float s_partial[kWarps];
    __shared__ float s_total;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) s_partial[warp] = v;
    __syncthreads();

    if (warp == 0) {
        v = (lane < kWarps) ? s_partial[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_total = v;
    }
    __syncthreads();
    return s_total;
}

// ----------------------------- [A] naive baseline ------------------------------------
__global__ void __launch_bounds__(kThreads)
residual_add_naive_kernel(const __half* __restrict__ x, const __half* __restrict__ residual,
                          __half* __restrict__ residual_out, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) residual_out[i] = __hadd(x[i], residual[i]);
}

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
rmsnorm_naive_kernel(const __half* __restrict__ h, const __half* __restrict__ weight,
                     __half* __restrict__ out, int D, float eps) {
    const __half* row = h + static_cast<size_t>(blockIdx.x) * D;
    __half* orow      = out + static_cast<size_t>(blockIdx.x) * D;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += THREADS) {          // pass 1: statistics
        const float v = __half2float(row[i]);
        ss += v * v;
    }
    const float rstd = rsqrtf(block_reduce_sum<THREADS>(ss) / D + eps);

    for (int i = threadIdx.x; i < D; i += THREADS) {          // pass 2: re-read & scale
        const __half n = __float2half_rn(__half2float(row[i]) * rstd);
        orow[i] = __hmul(n, weight[i]);
    }
}

// ------------------------- [B] vectorized, unfused (ablation) ------------------------
__global__ void __launch_bounds__(kThreads)
residual_add_vec_kernel(const __half* __restrict__ x, const __half* __restrict__ residual,
                        __half* __restrict__ residual_out, size_t n_vec) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_vec) return;
    const float4 xv = __ldcs(reinterpret_cast<const float4*>(x) + i);
    const float4 rv = __ldcs(reinterpret_cast<const float4*>(residual) + i);
    const __half2* xh = reinterpret_cast<const __half2*>(&xv);
    const __half2* rh = reinterpret_cast<const __half2*>(&rv);
    float4 hv;
    __half2* hh = reinterpret_cast<__half2*>(&hv);
#pragma unroll
    for (int j = 0; j < 4; ++j) hh[j] = __hadd2(xh[j], rh[j]);
    reinterpret_cast<float4*>(residual_out)[i] = hv;
}

template <int D, int THREADS>
__global__ void __launch_bounds__(THREADS)
rmsnorm_vec_kernel(const __half* __restrict__ h, const __half* __restrict__ weight,
                   __half* __restrict__ out, float eps) {
    constexpr int kVecsPerRow = D / kVec;
    constexpr int kVPT        = kVecsPerRow / THREADS;   // 128-bit vectors per thread
    const size_t base = static_cast<size_t>(blockIdx.x) * kVecsPerRow;
    const float4* __restrict__ h4 = reinterpret_cast<const float4*>(h) + base;
    const float4* __restrict__ w4 = reinterpret_cast<const float4*>(weight);
    float4* __restrict__ y4       = reinterpret_cast<float4*>(out) + base;

    float  hf[kVPT][kVec];
    float4 wv[kVPT];
    float  ss = 0.0f;
#pragma unroll
    for (int i = 0; i < kVPT; ++i) {
        const int c = threadIdx.x + i * THREADS;
        const float4 hv = __ldg(h4 + c);
        wv[i] = __ldg(w4 + c);
        const __half2* hh = reinterpret_cast<const __half2*>(&hv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(hh[j]);
            hf[i][2 * j] = f.x;
            hf[i][2 * j + 1] = f.y;
            ss = fmaf(f.x, f.x, ss);
            ss = fmaf(f.y, f.y, ss);
        }
    }
    const float rstd = rsqrtf(block_reduce_sum<THREADS>(ss) * (1.0f / D) + eps);
#pragma unroll
    for (int i = 0; i < kVPT; ++i) {
        const int c = threadIdx.x + i * THREADS;
        const __half2* wh = reinterpret_cast<const __half2*>(&wv[i]);
        float4 yv;
        __half2* yh = reinterpret_cast<__half2*>(&yv);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            yh[j] = __hmul2(__floats2half2_rn(hf[i][2 * j] * rstd, hf[i][2 * j + 1] * rstd), wh[j]);
        y4[c] = yv;
    }
}

// ------------------------------- [C] fused (optimized) -------------------------------
// One CTA per token row. For D = 2048 and 256 threads, every thread owns exactly one
// 128-bit chunk (8 halves) of the row, which stays in registers for the whole kernel.
template <int D, int THREADS>
__global__ void __launch_bounds__(THREADS)
fused_add_rmsnorm_kernel(const __half* __restrict__ x,        // sub-block output
                         const __half* __restrict__ residual, // incoming residual stream
                         const __half* __restrict__ weight,   // RMSNorm gamma [D]
                         __half* __restrict__ residual_out,   // x + residual (next residual)
                         __half* __restrict__ out,            // normalized output
                         float eps) {
    static_assert(D % (kVec * THREADS) == 0, "D must be a multiple of 8 * THREADS");
    constexpr int kVecsPerRow = D / kVec;
    constexpr int kVPT        = kVecsPerRow / THREADS;

    const size_t base = static_cast<size_t>(blockIdx.x) * kVecsPerRow;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x) + base;
    const float4* __restrict__ r4 = reinterpret_cast<const float4*>(residual) + base;
    const float4* __restrict__ w4 = reinterpret_cast<const float4*>(weight);
    float4* __restrict__ h4       = reinterpret_cast<float4*>(residual_out) + base;
    float4* __restrict__ y4       = reinterpret_cast<float4*>(out) + base;

    float  hf[kVPT][kVec];   // FP32 copy of the (FP16-rounded) residual row, in registers
    float4 wv[kVPT];
    float  ss = 0.0f;

#pragma unroll
    for (int i = 0; i < kVPT; ++i) {
        const int c = threadIdx.x + i * THREADS;
        // x and residual are consumed exactly once -> streaming loads (evict-first).
        const float4 xv = __ldcs(x4 + c);
        const float4 rv = __ldcs(r4 + c);
        // gamma (4 KB) is re-read by every CTA -> read-only path, stays L1/L2-resident.
        // Issued before the reduction so its latency is hidden.
        wv[i] = __ldg(w4 + c);

        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        const __half2* rh = reinterpret_cast<const __half2*>(&rv);
        float4 hv;
        __half2* hh = reinterpret_cast<__half2*>(&hv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            hh[j] = __hadd2(xh[j], rh[j]);              // FP16 add (matches PyTorch)
            const float2 f = __half22float2(hh[j]);    // stats on the rounded value (HF)
            hf[i][2 * j] = f.x;
            hf[i][2 * j + 1] = f.y;
            ss = fmaf(f.x, f.x, ss);
            ss = fmaf(f.y, f.y, ss);
        }
        // Store the new residual stream right away: the store drains while the block
        // reduction below is in flight. It is never read back by this kernel.
        h4[c] = hv;
    }

    const float rstd = rsqrtf(block_reduce_sum<THREADS>(ss) * (1.0f / D) + eps);

#pragma unroll
    for (int i = 0; i < kVPT; ++i) {
        const int c = threadIdx.x + i * THREADS;
        const __half2* wh = reinterpret_cast<const __half2*>(&wv[i]);
        float4 yv;
        __half2* yh = reinterpret_cast<__half2*>(&yv);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            yh[j] = __hmul2(__floats2half2_rn(hf[i][2 * j] * rstd, hf[i][2 * j + 1] * rstd), wh[j]);
        y4[c] = yv;
    }
}

// =====================================================================================
//                                     Host code
// =====================================================================================

struct Buffers {
    __half *x = nullptr, *r = nullptr, *w = nullptr;
    __half *h_naive = nullptr, *y_naive = nullptr;
    __half *h_vec = nullptr, *y_vec = nullptr;
    __half *h_fused = nullptr, *y_fused = nullptr;
};

// `row` = first token row of the slot being processed (buffer rotation, see benchmark()).
using LaunchFn = void (*)(const Buffers&, int N, size_t row, cudaStream_t);

static void launch_naive(const Buffers& b, int N, size_t row, cudaStream_t s) {
    const size_t off = row * kHidden;
    const size_t n = static_cast<size_t>(N) * kHidden;
    const unsigned blocks = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    residual_add_naive_kernel<<<blocks, kThreads, 0, s>>>(b.x + off, b.r + off, b.h_naive + off, n);
    rmsnorm_naive_kernel<kThreads>
        <<<N, kThreads, 0, s>>>(b.h_naive + off, b.w, b.y_naive + off, kHidden, kEps);
}

static void launch_vec_unfused(const Buffers& b, int N, size_t row, cudaStream_t s) {
    const size_t off = row * kHidden;
    const size_t n_vec = static_cast<size_t>(N) * kHidden / kVec;
    const unsigned blocks = static_cast<unsigned>((n_vec + kThreads - 1) / kThreads);
    residual_add_vec_kernel<<<blocks, kThreads, 0, s>>>(b.x + off, b.r + off, b.h_vec + off, n_vec);
    rmsnorm_vec_kernel<kHidden, kThreads>
        <<<N, kThreads, 0, s>>>(b.h_vec + off, b.w, b.y_vec + off, kEps);
}

static void launch_fused(const Buffers& b, int N, size_t row, cudaStream_t s) {
    const size_t off = row * kHidden;
    fused_add_rmsnorm_kernel<kHidden, kThreads>
        <<<N, kThreads, 0, s>>>(b.x + off, b.r + off, b.w, b.h_fused + off, b.y_fused + off, kEps);
}

struct Variant {
    const char* name;
    LaunchFn fn;
    double bytes_per_elem;  // ideal DRAM bytes per hidden element (for effective BW)
};

// ---------------------------------- options ------------------------------------------
struct Options {
    std::vector<int> tokens = {1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384};
    int warmup  = 50;
    int iters   = 200;
    int repeats = 9;
    bool hot_l2 = false;  // disable buffer rotation
    bool graph  = false;  // time a captured CUDA Graph instead of eager launches
};

static std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t end = str.find(',', pos);
        if (end == std::string::npos) end = str.size();
        const int t = std::atoi(str.substr(pos, end - pos).c_str());
        if (t > 0) v.push_back(t);
        pos = end + 1;
    }
    return v;
}

static Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "--tokens")       o.tokens  = parse_list(next());
        else if (a == "--warmup")  o.warmup  = std::max(1, std::atoi(next()));
        else if (a == "--iters")   o.iters   = std::max(1, std::atoi(next()));
        else if (a == "--repeats") o.repeats = std::max(1, std::atoi(next()));
        else if (a == "--hot-l2")  o.hot_l2  = true;
        else if (a == "--graph")   o.graph   = true;
        else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [--tokens a,b,c] [--warmup W] [--iters I] [--repeats R] [--hot-l2] [--graph]\n", argv[0]);
            std::exit(0);
        } else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); std::exit(1); }
    }
    if (o.tokens.empty()) { std::fprintf(stderr, "no valid --tokens\n"); std::exit(1); }
    return o;
}

// ------------------------------ host FP16 helpers ------------------------------------
static float g_h2f[65536];  // exact half->float lookup table

static void init_h2f_table() {
    for (uint32_t u = 0; u < 65536; ++u) {
        __half_raw raw; raw.x = static_cast<unsigned short>(u);
        g_h2f[u] = __half2float(__half(raw));
    }
}
static inline uint16_t f2h_bits(float f) {
    const __half h = __float2half_rn(f);
    uint16_t u; std::memcpy(&u, &h, sizeof(u));
    return u;
}

// Deterministic xorshift64* RNG in [lo, hi).
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    float uniform(float lo, float hi) {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        const uint64_t r = s * 2685821657736338717ULL;
        return lo + (hi - lo) * static_cast<float>((r >> 40) * (1.0 / 16777216.0));
    }
};

// ------------------------------- validation ------------------------------------------
struct ErrAcc {
    double num = 0, den = 0, maxabs = 0;
    void add(float test, float ref) {
        const double d = static_cast<double>(test) - ref;
        num += d * d; den += static_cast<double>(ref) * ref;
        maxabs = std::max(maxabs, std::fabs(d));
    }
    double rel() const { return den > 0 ? std::sqrt(num / den) : std::sqrt(num); }
};

struct ValResult {
    ErrAcc fused_vs_naive, vec_vs_naive, fused_vs_ref, naive_vs_ref, vec_vs_ref;
    size_t h_mismatch = 0;  // residual_out bit mismatches (any variant vs CPU reference)
    size_t nan_count  = 0;  // NaNs/unwritten outputs (outputs are pre-poisoned with NaN)
    bool pass = false;
};

static ValResult validate(const Buffers& b, int N, const std::vector<uint16_t>& hx,
                          const std::vector<uint16_t>& hr, const std::vector<uint16_t>& hw,
                          cudaStream_t s) {
    const size_t n = static_cast<size_t>(N) * kHidden;
    const size_t bytes = n * sizeof(__half);

    // Poison outputs with 0xFFFF (FP16 NaN) so any element a kernel fails to write is caught.
    for (__half* p : {b.h_naive, b.y_naive, b.h_vec, b.y_vec, b.h_fused, b.y_fused})
        CUDA_CHECK(cudaMemsetAsync(p, 0xFF, bytes, s));
    launch_naive(b, N, 0, s);
    launch_vec_unfused(b, N, 0, s);
    launch_fused(b, N, 0, s);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<uint16_t> hn(n), yn(n), hv(n), yv(n), hfu(n), yfu(n);
    CUDA_CHECK(cudaMemcpy(hn.data(),  b.h_naive, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yn.data(),  b.y_naive, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hv.data(),  b.h_vec,   bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yv.data(),  b.y_vec,   bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hfu.data(), b.h_fused, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(yfu.data(), b.y_fused, bytes, cudaMemcpyDeviceToHost));

    ValResult v;
    std::vector<float> hrow(kHidden);
    std::vector<uint16_t> hbits(kHidden);
    for (int row = 0; row < N; ++row) {
        const size_t o = static_cast<size_t>(row) * kHidden;
        // CPU reference with HF LlamaRMSNorm semantics; statistics in FP64.
        double ss = 0.0;
        for (int d = 0; d < kHidden; ++d) {
            // FP32 sum of two halves rounded once to FP16 == IEEE FP16 add (p'=24 >= 2p+2).
            hbits[d] = f2h_bits(g_h2f[hx[o + d]] + g_h2f[hr[o + d]]);
            hrow[d] = g_h2f[hbits[d]];
            ss += static_cast<double>(hrow[d]) * hrow[d];
        }
        const double rstd = 1.0 / std::sqrt(ss / kHidden + static_cast<double>(kEps));
        for (int d = 0; d < kHidden; ++d) {
            const size_t i = o + d;
            const uint16_t nb = f2h_bits(static_cast<float>(hrow[d] * rstd));
            const float ref = g_h2f[f2h_bits(g_h2f[nb] * g_h2f[hw[d]])];

            const float a = g_h2f[yn[i]], bv = g_h2f[yv[i]], c = g_h2f[yfu[i]];
            if (std::isnan(a) || std::isnan(bv) || std::isnan(c) || std::isnan(g_h2f[hn[i]]) ||
                std::isnan(g_h2f[hv[i]]) || std::isnan(g_h2f[hfu[i]])) {
                ++v.nan_count;
                continue;
            }
            if (hn[i] != hbits[d] || hv[i] != hbits[d] || hfu[i] != hbits[d]) ++v.h_mismatch;
            v.fused_vs_naive.add(c, a);
            v.vec_vs_naive.add(bv, a);
            v.fused_vs_ref.add(c, ref);
            v.naive_vs_ref.add(a, ref);
            v.vec_vs_ref.add(bv, ref);
        }
    }
    v.pass = v.nan_count == 0 && v.h_mismatch == 0 &&
             v.fused_vs_naive.rel() < kRelL2Tol && v.vec_vs_naive.rel() < kRelL2Tol &&
             v.fused_vs_ref.rel() < kRelL2Tol && v.naive_vs_ref.rel() < kRelL2Tol &&
             v.vec_vs_ref.rel() < kRelL2Tol;
    return v;
}

// --------------------------------- timing --------------------------------------------
static double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// Returns median microseconds per invocation for each variant.
//  * Each sample = `iters` back-to-back invocations bracketed by one cudaEvent pair
//    (amortises the ~0.5 us event resolution); `repeats` samples -> median.
//  * Buffer rotation: invocation `it` works on slot (it % slots). The slots span >= 3x L2,
//    so no invocation finds its x/residual in L2 from a previous one and dirty h/y lines
//    are forced back to DRAM -- the steady state of a real forward pass. Without this,
//    working sets <= 32 MiB are served from L2 and "DRAM bandwidth" exceeds the peak.
//  * Variants are interleaved and their order rotated every repeat, so slow drift in the
//    laptop GPU's boost clock / temperature does not systematically favour any variant.
//  * --graph: the `iters` invocations are captured once into a CUDA Graph and replayed,
//    removing CPU / WDDM submission overhead (what a CUDA-Graph inference engine sees).
static std::vector<double> benchmark(const std::vector<Variant>& vars, const Buffers& b, int N,
                                     int slots, const Options& o, cudaStream_t s,
                                     cudaEvent_t e0, cudaEvent_t e1) {
    const size_t nv = vars.size();
    auto run = [&](size_t k, int count) {
        for (int it = 0; it < count; ++it)
            vars[k].fn(b, N, static_cast<size_t>(it % slots) * N, s);
    };

    std::vector<cudaGraphExec_t> exec(nv, nullptr);
    if (o.graph) {
        for (size_t k = 0; k < nv; ++k) {
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            run(k, o.iters);
            CUDA_CHECK(cudaStreamEndCapture(s, &g));
            CUDA_CHECK(cudaGraphInstantiate(&exec[k], g, 0));
            CUDA_CHECK(cudaGraphDestroy(g));
        }
    }

    for (size_t k = 0; k < nv; ++k) {   // warm-up (clocks, caches, TLBs, graph upload)
        run(k, o.warmup);
        if (o.graph) CUDA_CHECK(cudaGraphLaunch(exec[k], s));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s));

    std::vector<std::vector<float>> samples(nv);
    for (int rep = 0; rep < o.repeats; ++rep) {
        for (size_t k0 = 0; k0 < nv; ++k0) {
            const size_t k = (k0 + rep) % nv;
            CUDA_CHECK(cudaEventRecord(e0, s));
            if (o.graph) CUDA_CHECK(cudaGraphLaunch(exec[k], s));
            else         run(k, o.iters);
            CUDA_CHECK(cudaEventRecord(e1, s));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            samples[k].push_back(ms * 1000.0f / o.iters);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    for (cudaGraphExec_t e : exec)
        if (e) CUDA_CHECK(cudaGraphExecDestroy(e));

    std::vector<double> med(nv);
    for (size_t k = 0; k < nv; ++k) med[k] = median(samples[k]);
    return med;
}

// =====================================================================================
int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);
    init_h2f_table();

    // ------------------------------- device info -------------------------------------
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int mem_clk_khz = 0, bus_bits = 0, l2_bytes = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev));
    const double peak_gbs = 2.0 * mem_clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;  // DDR

    std::printf("=====================================================================\n");
    std::printf(" Fused Residual-Add + RMSNorm benchmark  (D=%d, FP16, eps=%.0e)\n", kHidden, kEps);
    std::printf("=====================================================================\n");
    std::printf(" GPU            : %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf(" Memory         : %.1f GiB, %d-bit bus, peak %.1f GB/s, L2 %.1f MiB\n",
                prop.totalGlobalMem / 1073741824.0, bus_bits, peak_gbs, l2_bytes / 1048576.0);
    if (prop.major != 8 || prop.minor != 9)
        std::printf(" WARNING        : tuned for sm_89; running on sm_%d%d\n", prop.major, prop.minor);

    cudaFuncAttributes fattr{};
    CUDA_CHECK(cudaFuncGetAttributes(&fattr, fused_add_rmsnorm_kernel<kHidden, kThreads>));
    int fused_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &fused_blocks_per_sm, fused_add_rmsnorm_kernel<kHidden, kThreads>, kThreads, 0));
    std::printf(" Fused kernel   : %d thr/CTA, %d regs/thr, %zu B static smem, %d CTAs/SM resident\n",
                kThreads, fattr.numRegs, fattr.sharedSizeBytes, fused_blocks_per_sm);
    // ------------------------------- allocation --------------------------------------
    // Rotation capacity: enough rows that one sweep over all slots touches >= 3x L2
    // (4 tensors x D x 2 B per row for one variant).
    const int max_tokens = *std::max_element(opt.tokens.begin(), opt.tokens.end());
    const size_t bytes_per_row_set = 4ull * kHidden * sizeof(__half);
    const int rot_rows = static_cast<int>((3ull * l2_bytes + bytes_per_row_set - 1) / bytes_per_row_set);
    const int cap_rows = opt.hot_l2 ? max_tokens : std::max(max_tokens, rot_rows);
    const size_t max_elems = static_cast<size_t>(cap_rows) * kHidden;
    const size_t max_bytes = max_elems * sizeof(__half);

    std::printf(" Timing         : %s, %s, warmup %d, median of %d x %d (interleaved)\n",
                opt.graph ? "CUDA Graph replay" : "eager launches",
                opt.hot_l2 ? "HOT L2 (no rotation)" : "DRAM-streaming (buffer rotation >= 3x L2)",
                opt.warmup, opt.repeats, opt.iters);
    std::printf(" Device memory  : %.0f MiB allocated (%d rows x 8 tensors)\n",
                8.0 * max_bytes / 1048576.0, cap_rows);
    std::printf("---------------------------------------------------------------------\n");

    Buffers b;
    CUDA_CHECK(cudaMalloc(&b.x, max_bytes));
    CUDA_CHECK(cudaMalloc(&b.r, max_bytes));
    CUDA_CHECK(cudaMalloc(&b.w, kHidden * sizeof(__half)));
    for (__half** p : {&b.h_naive, &b.y_naive, &b.h_vec, &b.y_vec, &b.h_fused, &b.y_fused})
        CUDA_CHECK(cudaMalloc(p, max_bytes));

    // 128-bit accesses require 16-byte alignment (cudaMalloc guarantees >= 256 B).
    for (const void* p : {(const void*)b.x, (const void*)b.r, (const void*)b.w,
                          (const void*)b.h_fused, (const void*)b.y_fused}) {
        if (reinterpret_cast<uintptr_t>(p) % 16 != 0) {
            std::fprintf(stderr, "buffer not 16-byte aligned\n");
            return EXIT_FAILURE;
        }
    }

    // ------------------------------ input data ---------------------------------------
    // x ~ U(-1,1) (sub-block output), residual ~ U(-4,4) plus a few large "massive
    // activation" channels as seen in real Llama residual streams, gamma ~ U(0.5,1.5).
    std::vector<uint16_t> hx(max_elems), hr(max_elems), hw(kHidden);
    {
        Rng rng(0x5EED1234ABCDULL);
        for (size_t i = 0; i < max_elems; ++i) hx[i] = f2h_bits(rng.uniform(-1.f, 1.f));
        for (size_t i = 0; i < max_elems; ++i) {
            const int ch = static_cast<int>(i % kHidden);
            const float mag = (ch == 7 || ch == 1031) ? 120.f : 4.f;
            hr[i] = f2h_bits(rng.uniform(-mag, mag));
        }
        for (int d = 0; d < kHidden; ++d) hw[d] = f2h_bits(rng.uniform(0.5f, 1.5f));
    }
    CUDA_CHECK(cudaMemcpy(b.x, hx.data(), max_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.r, hr.data(), max_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.w, hw.data(), kHidden * sizeof(__half), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    const std::vector<Variant> vars = {
        {"naive",       launch_naive,       10.0},  // read x,r ; write h ; read h ; write y
        {"vec-unfused", launch_vec_unfused, 10.0},
        {"fused",       launch_fused,        8.0},  // read x,r ; write h,y
    };

    struct Row { int N; int slots; std::vector<double> us; ValResult val; };
    std::vector<Row> rows;
    bool all_pass = true;

    for (int N : opt.tokens) {
        const int slots = opt.hot_l2 ? 1 : std::max(1, cap_rows / N);
        std::printf(" tokens=%-6d validating...", N);
        std::fflush(stdout);
        ValResult val = validate(b, N, hx, hr, hw, stream);
        std::printf(" %s  benchmarking (%d rotation slots)...", val.pass ? "PASS" : "FAIL", slots);
        std::fflush(stdout);
        std::vector<double> us = benchmark(vars, b, N, slots, opt, stream, e0, e1);
        std::printf(" done\n");
        all_pass &= val.pass;
        rows.push_back({N, slots, us, val});
    }

    // ------------------------------- correctness -------------------------------------
    std::printf("\n=== Numerical check: relative L2 = ||test - ref||_2 / ||ref||_2  (tol %.0e) ===\n", kRelL2Tol);
    std::printf("%7s | %12s %10s | %12s | %12s %12s | %7s %5s | %s\n", "tokens", "fused~naive",
                "maxabs", "vec~naive", "fused~cpu", "naive~cpu", "h-bits", "NaN", "status");
    for (const Row& r : rows) {
        const ValResult& v = r.val;
        std::printf("%7d | %12.3e %10.3e | %12.3e | %12.3e %12.3e | %7zu %5zu | %s\n", r.N,
                    v.fused_vs_naive.rel(), v.fused_vs_naive.maxabs, v.vec_vs_naive.rel(),
                    v.fused_vs_ref.rel(), v.naive_vs_ref.rel(), v.h_mismatch, v.nan_count,
                    v.pass ? "PASS" : "FAIL");
    }
    std::printf("  (h-bits = residual_out elements not bit-identical to the reference FP16 add)\n");

    // ------------------------------- performance -------------------------------------
    std::printf("\n=== Performance (median us per Add+RMSNorm call) ===\n");
    std::printf("%7s | %9s %11s %9s | %11s %16s | %9s %9s\n", "tokens", "naive", "vec-unfused",
                "fused", "unfused GB/s", "fused GB/s (%pk)", "x naive", "x vec");
    for (const Row& r : rows) {
        const double elems = static_cast<double>(r.N) * kHidden;
        const double bw_vec   = elems * vars[1].bytes_per_elem / (r.us[1] * 1e-6) / 1e9;
        const double bw_fused = elems * vars[2].bytes_per_elem / (r.us[2] * 1e-6) / 1e9;
        std::printf("%7d | %9.2f %11.2f %9.2f | %11.1f  %8.1f (%4.1f%%) | %8.3fx %8.3fx\n", r.N,
                    r.us[0], r.us[1], r.us[2], bw_vec, bw_fused, 100.0 * bw_fused / peak_gbs,
                    r.us[0] / r.us[2], r.us[1] / r.us[2]);
    }
    std::printf("  (GB/s = ideal DRAM bytes / time: unfused 10 B/elem, fused 8 B/elem)\n");

    // --------------------------- end-to-end projection -------------------------------
    // Each decoder layer has 2 Add+RMSNorm sites (post-attn add + post_attention_layernorm,
    // post-MLP add + next input_layernorm / final norm).
    std::printf("\n=== Projected time saved per forward pass (2 sites/layer) ===\n");
    std::printf("%7s | %22s %22s | %22s %22s\n", "tokens", "TinyLlama vs naive",
                "TinyLlama vs vec", "Llama3.2-1B vs naive", "Llama3.2-1B vs vec");
    for (const Row& r : rows) {
        auto saved = [&](int layers, int base) { return 2.0 * layers * (r.us[base] - r.us[2]); };
        std::printf("%7d | %19.1f us %19.1f us | %19.1f us %19.1f us\n", r.N, saved(22, 0),
                    saved(22, 1), saved(16, 0), saved(16, 1));
    }
    std::printf("  (divide by your measured forward-pass latency to get the end-to-end gain)\n");

    std::printf("\nOVERALL: %s\n", all_pass ? "ALL NUMERICAL CHECKS PASSED" : "NUMERICAL CHECK FAILURE");

    // --------------------------------- cleanup ---------------------------------------
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (__half* p : {b.x, b.r, b.w, b.h_naive, b.y_naive, b.h_vec, b.y_vec, b.h_fused, b.y_fused})
        CUDA_CHECK(cudaFree(p));
    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
