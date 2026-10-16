#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#if defined(__CUDA_NO_HALF_OPERATORS__)
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#if defined(__CUDA_NO_HALF2_OPERATORS__)
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#if defined(__CUDA_NO_HALF_CONVERSIONS__)
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <tuple>

// Tree reduction inside a warp using __shfl_down_sync. Result valid in lane 0.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

// Block reduction across warps via shared memory.
template <int THREADS>
__device__ __forceinline__ float block_reduce_sum(float v) {
    static_assert(THREADS % 32 == 0 && THREADS <= 1024, "THREADS must be multiple of 32");
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

// Optimized Fused Residual-Add + RMSNorm Kernel for D = 2048, FP16
// Each CTA processes 1 token row.
template <int D, int THREADS>
__global__ void __launch_bounds__(THREADS)
fused_add_rmsnorm_kernel(const __half* __restrict__ x,
                         const __half* __restrict__ residual,
                         const __half* __restrict__ weight,
                         __half* __restrict__ residual_out,
                         __half* __restrict__ out,
                         float eps) {
    constexpr int kVec = 8; // 8 halves = 128-bit float4
    static_assert(D % (kVec * THREADS) == 0, "D must be a multiple of 8 * THREADS");
    constexpr int kVecsPerRow = D / kVec;
    constexpr int kVPT        = kVecsPerRow / THREADS; // 1 float4 per thread for D=2048, THREADS=256

    const size_t base = static_cast<size_t>(blockIdx.x) * kVecsPerRow;
    const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x) + base;
    const float4* __restrict__ r4 = reinterpret_cast<const float4*>(residual) + base;
    const float4* __restrict__ w4 = reinterpret_cast<const float4*>(weight);
    float4* __restrict__ h4       = reinterpret_cast<float4*>(residual_out) + base;
    float4* __restrict__ y4       = reinterpret_cast<float4*>(out) + base;

    float  hf[kVPT][kVec];
    float4 wv[kVPT];
    float  ss = 0.0f;

#pragma unroll
    for (int i = 0; i < kVPT; ++i) {
        const int c = threadIdx.x + i * THREADS;
        const float4 xv = __ldcs(x4 + c); // Stream load
        const float4 rv = __ldcs(r4 + c);
        wv[i]           = __ldg(w4 + c);  // Read-only weight cache

        const __half2* xh = reinterpret_cast<const __half2*>(&xv);
        const __half2* rh = reinterpret_cast<const __half2*>(&rv);
        float4 hv;
        __half2* hh = reinterpret_cast<__half2*>(&hv);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            hh[j] = __hadd2(xh[j], rh[j]);
            const float2 f = __half22float2(hh[j]);
            hf[i][2 * j]     = f.x;
            hf[i][2 * j + 1] = f.y;
            ss = fmaf(f.x, f.x, ss);
            ss = fmaf(f.y, f.y, ss);
        }
        // Write new residual stream once; drains during block reduction
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
        for (int j = 0; j < 4; ++j) {
            yh[j] = __hmul2(__floats2half2_rn(hf[i][2 * j] * rstd, hf[i][2 * j + 1] * rstd), wh[j]);
        }
        y4[c] = yv;
    }
}

// C++ entry point wrapped for PyTorch
std::tuple<torch::Tensor, torch::Tensor> fused_add_rmsnorm(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    TORCH_CHECK(residual.scalar_type() == torch::kHalf, "residual must be float16");
    TORCH_CHECK(weight.scalar_type() == torch::kHalf, "weight must be float16");

    const int hidden_size = weight.numel();
    TORCH_CHECK(hidden_size == 2048, "hidden_size must be 2048 (TinyLlama/Llama-3.2-1B)");
    TORCH_CHECK(x.size(-1) == hidden_size, "Last dim of x must match hidden_size");
    TORCH_CHECK(residual.size(-1) == hidden_size, "Last dim of residual must match hidden_size");

    auto x_contig   = x.contiguous();
    auto res_contig = residual.contiguous();
    auto w_contig   = weight.contiguous();

    const size_t num_tokens = x_contig.numel() / hidden_size;

    auto residual_out = torch::empty_like(x_contig);
    auto out          = torch::empty_like(x_contig);

    constexpr int kThreads = 256;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    fused_add_rmsnorm_kernel<2048, kThreads><<<static_cast<unsigned int>(num_tokens), kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(x_contig.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(res_contig.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w_contig.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(residual_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        static_cast<float>(eps)
    );

    return std::make_tuple(residual_out, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_rmsnorm", &fused_add_rmsnorm, "Fused Residual Add + RMSNorm (sm_89)",
          py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("eps") = 1e-5);
}
