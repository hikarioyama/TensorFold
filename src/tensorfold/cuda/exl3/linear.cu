// EXL3 linear, any codebook and width, 1-128 rows: a row's bits depend only on it (mma keeps rows apart, K ranges fixed by (K, N), fixed-order sums).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "decode.cuh"

using namespace tf_exl3;

namespace {

enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else if (dtype == BF16) {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(p) + i);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const half*>(p) + i);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(p) + i) = u;
    } else {
        half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(static_cast<half*>(p) + i) = u;
    }
}

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh, + bias.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, const half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
        // Speculative loads must have a valid address even when the split-K result has no bias.
        const half bv = __ldg((bias ? bias : svh) + col + j);
        if (bias) v[j] += __half2float(bv);
    }
}

// A lane's words of one k step: at 1 and 2 bits the warp loads the step together and each lane takes its words by shuffle.
template <int K2>
__host__ __device__ constexpr bool step_shuffled() {
    return K2 == 2 || K2 == 4;
}

template <int K2, bool COOPERATIVE = false, int TILES = 8>
__host__ __device__ constexpr int step_regs() {
    return (step_shuffled<K2>() || COOPERATIVE) ? TILES * tile_words<K2>() / 32 : TILES * lane_words<K2>();
}

template <int K2, bool COOPERATIVE = false, int TILES = 8>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[step_regs<K2, COOPERATIVE, TILES>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>() || COOPERATIVE) {
#pragma unroll
        for (int c = 0; c < step_regs<K2, COOPERATIVE, TILES>(); ++c) raw[c] = __ldg(step + c * 32 + lane);
    } else {
        int word, offset;
        lane_start<K2>(lane, word, offset);
#pragma unroll
        for (int j = 0; j < TILES; ++j)
#pragma unroll
            for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
    }
}

// Tile j's lane words from a loaded step; prev: the lane's first window starts in the word before its own.
template <int K2, bool COOPERATIVE = false, int TILES = 8>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[step_regs<K2, COOPERATIVE, TILES>()], int j, int lane, bool prev,
                                                uint32_t (&w)[lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    if constexpr (step_shuffled<K2>()) {
        static_assert(LW == 2, "a lane's windows span two words at 1 and 2 bits");
        constexpr int LPW = 8 / K2;                  // lanes whose windows end in the same word
        const uint32_t r = raw[j * TW / 32];
        const int base = (j * TW) % 32;
        const uint32_t own = __shfl_sync(0xffffffffu, r, base + lane / LPW);
        const uint32_t before = __shfl_sync(0xffffffffu, r, base + (lane / LPW + TW - 1) % TW);
        w[0] = prev ? before : own;
        w[1] = prev ? own : 0u;                      // a window inside one word never reads w[1]
    } else if constexpr (COOPERATIVE) {
        static_assert(K2 == 10 || K2 == 12, "cooperative words require 5/6 bits");
        int word, offset;
        lane_start<K2>(lane, word, offset);
        // Each unrolled tile spans two register-held 32-word chunks.
        const int chunk = j * TW / 32, base = (j * TW) % 32;
#pragma unroll
        for (int q = 0; q < LW; ++q) {
            const int at = base + (word + q) % TW;
            const uint32_t low = __shfl_sync(0xffffffffu, raw[chunk], at & 31);
            const uint32_t high = __shfl_sync(0xffffffffu, raw[chunk + 1], at & 31);
            w[q] = at < 32 ? low : high;
        }
    } else {
#pragma unroll
        for (int q = 0; q < LW; ++q) w[q] = raw[j * LW + q];
    }
}

// Two parts: blockIdx.x < nb is part 1, its upper half part 2 (the parts' column ranges do not overlap).
__global__ void __launch_bounds__(128) rot_in2_kernel(const void* __restrict__ x, int x_dtype,
                                                      const half* __restrict__ suh1, half* __restrict__ xh1,
                                                      const half* __restrict__ suh2, half* __restrict__ xh2,
                                                      int K, int nb) {
    const int part = blockIdx.x >= (unsigned)nb;
    const int blk = (blockIdx.x - (part ? nb : 0)) * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)row * K + k, v);
    load4(part ? suh2 : suh1, F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(part ? xh2 : xh1, F16, (size_t)row * K + k, v);
}

__global__ void __launch_bounds__(128) rot_in_kernel(const void* __restrict__ x, int x_dtype,
                                                     const half* __restrict__ suh, half* __restrict__ xh, int K) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= K) return;
    const int k = blk * 128 + 4 * lane;
    float v[4], s[4];
    load4(x, x_dtype, (size_t)row * K + k, v);
    load4(suh, F16, k, s);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= s[j];
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    store4(xh, F16, (size_t)row * K + k, v);
}

template <bool TRANSPOSE>
__device__ __forceinline__ float4 red4(const float* red, int base, int row, int col) {
    if constexpr (!TRANSPOSE) return *reinterpret_cast<const float4*>(red + base + col);
    const int sw = (row & 1) | ((row & 6) << 2);
    const float4 v = *reinterpret_cast<const float4*>(red + base + (col ^ (sw & ~3)));
    return (sw & 1) ? make_float4(v.y, v.x, v.w, v.z) : v;
}

// One projection's programs: ``nb``/``split`` are this program's N block and K split, so a pair launch can run two
// projections of the same width from one grid. y's rows may be a column range of a wider buffer (a fused projection
// writes several parts into one output), so its row stride is passed in; Z stays this projection's own contiguous
// split-K scratch, whose stride is N.
template <int K2, int CB, int WK, bool TRANSPOSE = false, bool COOPERATIVE = false, bool HALF_N = false>
__device__ __forceinline__ void linear_part(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    long long y_stride, float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK,
    int nb, int split) {
    constexpr int TW = tile_words<K2>();
    constexpr int NT = HALF_N ? 4 : 8, NC = NT * 16;
    const int half_n = HALF_N ? (nb & 1) : 0;
    if constexpr (HALF_N) nb >>= 1;
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp

    const int NB = N / 128;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb + half_n * 4 * TW;
    const int col0 = nb * 128 + half_n * 64;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);

        float acc[NT][2][4];
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * K;
        const half* x1 = xh + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2, COOPERATIVE, NT>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2, COOPERATIVE, NT>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t a[4];
            a[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
            a[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
            a[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
            a[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2, COOPERATIVE, NT>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2, COOPERATIVE, NT>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                if constexpr (TRANSPOSE) {
                    uint32_t wa[4] = {b0[0], b1[0], b0[1], b1[1]}, xb[2] = {a[0], a[2]};
                    mma16816(acc[j][0], wa, xb);
                } else {
                    mma16816(acc[j][0], a, b0);
                    mma16816(acc[j][1], a, b1);
                }
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if constexpr (TRANSPOSE) {
                const int ra = 2 * t, rb = ra + 1;
                const int sa = (ra & 1) | ((ra & 6) << 2), sb = (rb & 1) | ((rb & 6) << 2);
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    const int col = i * 16 + g;
                    if (ra < RH) {
                        red[(warp * RH + ra) * NC + (col ^ sa)] = acc[i][0][0];
                        red[(warp * RH + ra) * NC + ((col + 8) ^ sa)] = acc[i][0][2];
                    }
                    if (rb < RH) {
                        red[(warp * RH + rb) * NC + (col ^ sb)] = acc[i][0][1];
                        red[(warp * RH + rb) * NC + ((col + 8) ^ sb)] = acc[i][0][3];
                    }
                }
            } else {
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * NC + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = red4<TRANSPOSE>(red, r * NC, r, 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = red4<TRANSPOSE>(red, (w * RH + r) * NC, r, 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + rlo + r) * y_stride + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * (NC / 4); idx += WK * 32) {
                    const int r = idx / (NC / 4), c = 4 * (idx % (NC / 4));
                    float4 s = red4<TRANSPOSE>(red, r * NC, r, c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = red4<TRANSPOSE>(red, (w * RH + r) * NC, r, c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK * (HALF_N ? 2 : 1) - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + nb * 128 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, nb * 128 + 4 * lane);
                    store4(y, y_dtype, (size_t)(m0 + r) * y_stride + nb * 128 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

template <int K2, int CB, int WK, bool TRANSPOSE = false, bool HALF_N = false>
__global__ void __launch_bounds__(WK * 32) linear_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    long long y_stride, float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK) {
    linear_part<K2, CB, WK, TRANSPOSE, HALF_N, HALF_N>(xh, T, stride_k, stride_nb, svh, bias, y, y_dtype, y_stride, Z, counters, M, K, N, SK,
                            blockIdx.x, blockIdx.y);
}

// Two projections of the same input, width and codebook in one launch (plan K1-b's multi-projection): the programs
// of both parts share the SMs and the launch, and each part keeps its own weights, scales, bias, output, split-K
// scratch and counters, so a part's programs do exactly what a launch of its own would have done — a row's bits are
// unchanged. Grid: x = NB1 + NB2 over both parts, y = the longer of the two K splits.
template <int K2, int CB, int WK, bool TRANSPOSE = false, bool COOPERATIVE = false>
__global__ void __launch_bounds__(WK * 32) linear_pair_kernel(
    const half* __restrict__ xh1, const uint32_t* __restrict__ T1, long long k1, long long nb1, const half* svh1,
    const half* bias1, void* y1, int y1_dtype, long long ys1, float* Z1, int* c1, int M, int K1, int N1, int SK1,
    const half* __restrict__ xh2, const uint32_t* __restrict__ T2, long long k2, long long nb2, const half* svh2,
    const half* bias2, void* y2, int y2_dtype, long long ys2, float* Z2, int* c2, int K2n, int N2, int SK2, bool compact) {
    const int w1 = N1 / 128;
    const int a_count = w1 * SK1;
    const bool part_a = compact ? blockIdx.x < (unsigned)a_count : blockIdx.x < (unsigned)w1;
    const int nb = compact ? (part_a ? blockIdx.x / SK1 : (blockIdx.x - a_count) / SK2)
                           : (part_a ? blockIdx.x : blockIdx.x - w1);
    const int split = compact ? (part_a ? blockIdx.x % SK1 : (blockIdx.x - a_count) % SK2) : blockIdx.y;
    if (part_a) {
        if (split < SK1) {
            linear_part<K2, CB, WK, TRANSPOSE, COOPERATIVE>(xh1, T1, k1, nb1, svh1, bias1, y1, y1_dtype, ys1, Z1, c1, M, K1, N1, SK1,
                                    nb, split);
        }
    } else if (split < SK2) {
        linear_part<K2, CB, WK, TRANSPOSE, COOPERATIVE>(xh2, T2, k2, nb2, svh2, bias2, y2, y2_dtype, ys2, Z2, c2, M, K2n, N2, SK2,
                                nb, split);
    }
}

template <int K2, int CB, int WK, bool TRANSPOSE = false, bool COOPERATIVE = false, bool HALF_N = false, int GROUP = 1>
__device__ __forceinline__ void linear_part_g2(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    long long y_stride, float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK,
    int nb, int split) {
    constexpr int TW = tile_words<K2>();
    constexpr int NT = HALF_N ? 4 : 8, NC = NT * 16;
    const int half_n = HALF_N ? (nb & 1) : 0;
    if constexpr (HALF_N) nb >>= 1;
    constexpr int LW = lane_words<K2>();
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int RH = min(M, 8);                                 // rows of red a warp

    const int NB = N / 128;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = T + nb * stride_nb + half_n * 4 * TW;
    const int col0 = nb * 128 + half_n * 64;
    bool prev = false;
    if constexpr (step_shuffled<K2>()) {
        int word, offset;
        lane_start<K2>(lane, word, offset);
        prev = word != lane / (8 / K2);
    }

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16 * GROUP, pass += GROUP) {

        float acc[GROUP][NT][2][4];
#pragma unroll
        for (int rg = 0; rg < GROUP; ++rg)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[rg][i][h][c] = 0.f;

        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        // up to 6 bits the next k step's words are loaded while this one is decoded; 7 and 8 bits load per tile
        constexpr bool PF = K2 <= 12;
        constexpr int SR = step_regs<K2, COOPERATIVE, NT>();
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2, COOPERATIVE, NT>(tile, lane, cur);
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            const int kt = kt0 + i;
            uint32_t av[GROUP][4];
#pragma unroll
            for (int rg = 0; rg < GROUP; ++rg) {
                const int tile_m = m0 + rg * 16;
                const int R = max(1, min(16, M - tile_m));
                const int r0 = min(M - 1, tile_m + (g < R ? g : R - 1));
                const int r1 = min(M - 1, tile_m + (g + 8 < R ? g + 8 : R - 1));
                const half* x0 = xh + (size_t)r0 * K;
                const half* x1 = xh + (size_t)r1 * K;
                av[rg][0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t));
                av[rg][1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t));
                av[rg][2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * t + 8));
                av[rg][3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * t + 8));
            }
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2, COOPERATIVE, NT>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2, COOPERATIVE, NT>(cur, j, lane, prev, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
#pragma unroll
                for (int rg = 0; rg < GROUP; ++rg) {
                    if constexpr (TRANSPOSE) {
                        uint32_t wa[4] = {b0[0], b1[0], b0[1], b1[1]}, xb[2] = {av[rg][0], av[rg][2]};
                        mma16816(acc[rg][j][0], wa, xb);
                    } else {
                        mma16816(acc[rg][j][0], av[rg], b0);
                        mma16816(acc[rg][j][1], av[rg], b1);
                    }
                }
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

#pragma unroll
        for (int rg = 0; rg < GROUP; ++rg) {
            const int tile_m = m0 + rg * 16;
            const int R = min(16, M - tile_m);
            if (R <= 0) continue;
        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = min(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if constexpr (TRANSPOSE) {
                const int ra = 2 * t, rb = ra + 1;
                const int sa = (ra & 1) | ((ra & 6) << 2), sb = (rb & 1) | ((rb & 6) << 2);
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    const int col = i * 16 + g;
                    if (ra < RH) {
                        red[(warp * RH + ra) * NC + (col ^ sa)] = acc[rg][i][0][0];
                        red[(warp * RH + ra) * NC + ((col + 8) ^ sa)] = acc[rg][i][0][2];
                    }
                    if (rb < RH) {
                        red[(warp * RH + rb) * NC + (col ^ sb)] = acc[rg][i][0][1];
                        red[(warp * RH + rb) * NC + ((col + 8) ^ sb)] = acc[rg][i][0][3];
                    }
                }
            } else {
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * t;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * NC + col) =
                            rlo ? make_float2(acc[rg][i][h][2], acc[rg][i][h][3]) : make_float2(acc[rg][i][h][0], acc[rg][i][h][1]);
                    }
            }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = red4<TRANSPOSE>(red, r * NC, r, 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = red4<TRANSPOSE>(red, (w * RH + r) * NC, r, 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, bias, col0 + 4 * lane);
                    store4(y, y_dtype, (size_t)(tile_m + rlo + r) * y_stride + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * (NC / 4); idx += WK * 32) {
                    const int r = idx / (NC / 4), c = 4 * (idx % (NC / 4));
                    float4 s = red4<TRANSPOSE>(red, r * NC, r, c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = red4<TRANSPOSE>(red, (w * RH + r) * NC, r, c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + tile_m + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + (pass + rg) * NB + nb, 1) == SK * (HALF_N ? 2 : 1) - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)tile_m + r) * N + nb * 128 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(Z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(Z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, bias, nb * 128 + 4 * lane);
                    store4(y, y_dtype, (size_t)(tile_m + r) * y_stride + nb * 128 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[(pass + rg) * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
        }
    }
}


// Two independent 16-row tiles reuse one trellis decode. G1 is left verbatim.
template <int K2, int CB, int WK>
__global__ void __launch_bounds__(WK * 32) linear_g2_kernel(
    const half* __restrict__ xh, const uint32_t* __restrict__ T, long long stride_k, long long stride_nb,
    const half* __restrict__ svh, const half* __restrict__ bias, void* __restrict__ y, int y_dtype,
    long long y_stride, float* __restrict__ Z, int* __restrict__ counters, int M, int K, int N, int SK) {
    linear_part_g2<K2, CB, WK, false, false, false, 2>(xh,T,stride_k,stride_nb,svh,bias,y,y_dtype,y_stride,Z,counters,M,K,N,SK,blockIdx.x,blockIdx.y);
}

template <int K2, int CB, int WK>
__global__ void __launch_bounds__(WK * 32) linear_pair_g2_kernel(
    const half* __restrict__ xh1, const uint32_t* __restrict__ T1, long long k1, long long nb1, const half* svh1,
    const half* bias1, void* y1, int y1_dtype, long long ys1, float* Z1, int* c1, int M, int K1, int N1, int SK1,
    const half* __restrict__ xh2, const uint32_t* __restrict__ T2, long long k2, long long nb2, const half* svh2,
    const half* bias2, void* y2, int y2_dtype, long long ys2, float* Z2, int* c2, int K2n, int N2, int SK2, bool compact) {
    const int w1=N1/128;
    if (blockIdx.x < (unsigned)w1) {
        if (blockIdx.y < (unsigned)SK1)
            linear_part_g2<K2, CB, WK, false, false, false, 2>(xh1,T1,k1,nb1,svh1,bias1,y1,y1_dtype,ys1,Z1,c1,M,K1,N1,SK1,blockIdx.x,blockIdx.y);
    } else if (blockIdx.y < (unsigned)SK2) {
        linear_part_g2<K2, CB, WK, false, false, false, 2>(xh2,T2,k2,nb2,svh2,bias2,y2,y2_dtype,ys2,Z2,c2,M,K2n,N2,SK2,blockIdx.x-w1,blockIdx.y);
    }
}

// W_q [K, N] fp16 from the trellis words (tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb): one warp per tile.
template <int K2, int CB>
__global__ void __launch_bounds__(32) unpack_kernel(const uint32_t* __restrict__ T, half* __restrict__ W, int N,
                                                    int64_t stride_k, int64_t stride_nb) {
    const int kt = blockIdx.y, nt = blockIdx.x, lane = threadIdx.x;
    uint32_t w[lane_words<K2>()];
    ldg_lane_words<K2>(T + kt * stride_k + (nt >> 3) * stride_nb + (nt & 7) * tile_words<K2>(), lane, w);
    uint32_t b[2][2];
    decode_lane<K2, CB>(w, lane, b[0], b[1]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t v = b[j >> 2][(j >> 1) & 1];
        const unsigned short h = (j & 1) ? (unsigned short)(v >> 16) : (unsigned short)(v & 0xffffu);
        W[(size_t)(kt * 16 + value_row(lane, j)) * N + nt * 16 + value_col(lane, j)] = __ushort_as_half(h);
    }
}

int dtype_of(const at::Tensor& t) {
    return t.scalar_type() == at::kFloat ? F32 : t.scalar_type() == at::kBFloat16 ? BF16 : F16;
}

}  // namespace

#define TF_EXL3_WIDTHS(X, CB) X(2, CB) X(4, CB) X(6, CB) X(8, CB) X(10, CB) X(12, CB) X(14, CB) X(16, CB)
#define TF_EXL3_ALL(X) TF_EXL3_WIDTHS(X, 0) TF_EXL3_WIDTHS(X, 1) TF_EXL3_WIDTHS(X, 2) X(3, 2) X(5, 2) X(7, 2)

// Two rotations of the same input rows (a fused projection's parts have different suh) in one launch: grid.x covers
// both parts, so a program's columns come from its own suh and xh. Each value is the arithmetic of the single-part
// launch, so the rotated rows are unchanged.
void exl3_rot_in2_cuda(const at::Tensor& x, const at::Tensor& suh1, at::Tensor& xh1, const at::Tensor& suh2,
                       at::Tensor& xh2) {
    const int M = (int)x.size(0), K = (int)x.size(1);
    const int nb = (K / 128 + 3) / 4;
    dim3 grid((unsigned)(2 * nb), (unsigned)M);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto launch = [&](int dt) {
        rot_in2_kernel<<<grid, 128, 0, stream>>>(
            x.data_ptr(), dt, reinterpret_cast<const half*>(suh1.data_ptr()), reinterpret_cast<half*>(xh1.data_ptr()),
            reinterpret_cast<const half*>(suh2.data_ptr()), reinterpret_cast<half*>(xh2.data_ptr()), K, nb);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (x.scalar_type() == at::kFloat) launch(F32);
    else if (x.scalar_type() == at::kBFloat16) launch(BF16);
    else launch(F16);
}

void exl3_rot_in_cuda(const at::Tensor& x, const at::Tensor& suh, at::Tensor& xh) {
    const int M = (int)x.size(0), K = (int)x.size(1);
    dim3 grid((unsigned)((K / 128 + 3) / 4), (unsigned)M);
    rot_in_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr(), dtype_of(x), reinterpret_cast<const half*>(suh.data_ptr()),
        reinterpret_cast<half*>(xh.data_ptr()), K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3_linear_cuda(const at::Tensor& xh, const at::Tensor& T, int64_t stride_k, int64_t stride_nb,
                      const at::Tensor& svh, const c10::optional<at::Tensor>& bias, at::Tensor& y,
                      const c10::optional<at::Tensor>& Z, at::Tensor& counters, int64_t K2, int64_t cb, int64_t SK,
                      int64_t WK, int64_t mode) {
    const int M = (int)xh.size(0), K = (int)xh.size(1), N = (int)y.size(1);
    TORCH_CHECK(WK == 2 || WK == 4 || WK == 8, "WK must be 2, 4 or 8");
    TORCH_CHECK((K / 16) % (SK * WK) == 0, "K / 16 must split evenly over SK * WK warps");
    dim3 grid((unsigned)(N / 128), (unsigned)SK);
    auto stream = at::cuda::getCurrentCUDAStream();
    const half* bptr = bias ? reinterpret_cast<const half*>(bias->data_ptr()) : nullptr;
    float* zptr = Z ? Z->data_ptr<float>() : nullptr;
    TORCH_CHECK(SK == 1 || zptr, "Z is needed with more than one split");
#define TF_LAUNCH(K2_, CB_)                                                                                        \
    if (K2 == K2_ && cb == CB_) {                                                                               \
        auto kernel = WK == 2 ? linear_kernel<K2_, CB_, 2>                                                      \
                              : WK == 4 ? linear_kernel<K2_, CB_, 4> : linear_kernel<K2_, CB_, 8>;              \
        if constexpr (CB_ == 2 && K2_ <= 12) {                                                               \
            if (mode && M <= 8) kernel = WK == 2 ? linear_kernel<K2_, CB_, 2, true>                            \
                 : WK == 4 ? linear_kernel<K2_, CB_, 4, true> : linear_kernel<K2_, CB_, 8, true>;              \
        }                                                                                                    \
        if constexpr (CB_ == 2 && (K2_ == 8 || K2_ == 10 || K2_ == 12)) {                          \
            if (mode == 9 && M > 16) kernel = WK == 2 ? linear_g2_kernel<K2_, CB_, 2>                    \
                : WK == 4 ? linear_g2_kernel<K2_, CB_, 4> : linear_g2_kernel<K2_, CB_, 8>;             \
        }                                                                                             \
        bool half_n = false;                                                                           \
        if constexpr (K2_ == 10 && CB_ == 2) {                                                           \
            if (mode == 7 && M <= 8 && WK == 4 && SK > 1) {                                                   \
                kernel = linear_kernel<K2_, CB_, 4, true, true>; grid.x *= 2; half_n = true;              \
            }                                                                                           \
        }                                                                                               \
        const int smem = (int)(WK * std::min(M, 8) * (half_n ? 64 : 128) * sizeof(float));                                \
        if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
        kernel<<<grid, (unsigned)(WK * 32), smem, stream>>>(                                              \
            reinterpret_cast<const half*>(xh.data_ptr()), reinterpret_cast<const uint32_t*>(T.data_ptr()),      \
            stride_k, stride_nb, reinterpret_cast<const half*>(svh.data_ptr()), bptr, y.data_ptr(), dtype_of(y),\
            y.stride(0), zptr, counters.data_ptr<int>(), M, K, N, (int)SK);                                                  \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                         \
        return;                                                                                                 \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}

void exl3_linear_pair_cuda(const at::Tensor& xh1, const at::Tensor& T1, int64_t k1, int64_t nb1, const at::Tensor& svh1,
                           const c10::optional<at::Tensor>& bias1, at::Tensor& y1, const c10::optional<at::Tensor>& Z1,
                           at::Tensor& c1, const at::Tensor& xh2, const at::Tensor& T2, int64_t k2, int64_t nb2,
                           const at::Tensor& svh2, const c10::optional<at::Tensor>& bias2, at::Tensor& y2,
                           const c10::optional<at::Tensor>& Z2, at::Tensor& c2, int64_t K2v, int64_t cb, int64_t SK1,
                           int64_t SK2, int64_t WK, int64_t mode, bool compact) {
    const int M = (int)xh1.size(0), K = (int)xh1.size(1), N1 = (int)y1.size(1), N2 = (int)y2.size(1);
    TORCH_CHECK(xh2.size(0) == M && xh2.size(1) == K, "both parts take the same rows and K");
    TORCH_CHECK(WK == 2 || WK == 4 || WK == 8, "WK must be 2, 4 or 8");
    compact = compact && SK1 != SK2 && mode != 9;
    dim3 grid((unsigned)(compact ? (N1 / 128 * SK1 + N2 / 128 * SK2) : N1 / 128 + N2 / 128),
              (unsigned)(compact ? 1 : std::max(SK1, SK2)));
    auto stream = at::cuda::getCurrentCUDAStream();
    const half* b1 = bias1 ? reinterpret_cast<const half*>(bias1->data_ptr()) : nullptr;
    const half* b2 = bias2 ? reinterpret_cast<const half*>(bias2->data_ptr()) : nullptr;
    float* z1 = Z1 ? Z1->data_ptr<float>() : nullptr;
    float* z2 = Z2 ? Z2->data_ptr<float>() : nullptr;
    TORCH_CHECK((SK1 == 1 || z1) && (SK2 == 1 || z2), "Z is needed with more than one split");
#define TF_LAUNCH(K2_, CB_)                                                                                    \
    if (K2v == K2_ && cb == CB_) {                                                                             \
        auto kernel = WK == 2 ? linear_pair_kernel<K2_, CB_, 2>                                                \
                              : WK == 4 ? linear_pair_kernel<K2_, CB_, 4> : linear_pair_kernel<K2_, CB_, 8>;    \
        if (mode && M <= 8) kernel = WK == 2 ? linear_pair_kernel<K2_, CB_, 2, true>                            \
             : WK == 4 ? linear_pair_kernel<K2_, CB_, 4, true> : linear_pair_kernel<K2_, CB_, 8, true>;          \
        /* Mode 3: measured small-M 5-bit WK8 pair only; all other paths stay unchanged. */               \
        if constexpr (K2_ == 10 && CB_ == 2) {                                                            \
            if ((mode == 3 || mode == 7) && M <= 8 && WK == 8) kernel = linear_pair_kernel<K2_, CB_, 8, true, true>;      \
        }                                                                                                \
        if constexpr (CB_ == 2 && (K2_ == 8 || K2_ == 10 || K2_ == 12)) {                          \
            if (mode == 9 && M > 16) kernel = WK == 2 ? linear_pair_g2_kernel<K2_, CB_, 2>               \
                : WK == 4 ? linear_pair_g2_kernel<K2_, CB_, 4> : linear_pair_g2_kernel<K2_, CB_, 8>;  \
        }                                                                                             \
        const int smem = (int)(WK * std::min(M, 8) * 128 * sizeof(float));                                     \
        if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
        kernel<<<grid, (unsigned)(WK * 32), smem, stream>>>(                                                   \
            reinterpret_cast<const half*>(xh1.data_ptr()), reinterpret_cast<const uint32_t*>(T1.data_ptr()),  \
            k1, nb1, reinterpret_cast<const half*>(svh1.data_ptr()), b1, y1.data_ptr(), dtype_of(y1),          \
            y1.stride(0), z1, c1.data_ptr<int>(), M, K, N1, (int)SK1,                                          \
            reinterpret_cast<const half*>(xh2.data_ptr()), reinterpret_cast<const uint32_t*>(T2.data_ptr()),  \
            k2, nb2, reinterpret_cast<const half*>(svh2.data_ptr()), b2, y2.data_ptr(), dtype_of(y2),          \
            y2.stride(0), z2, c2.data_ptr<int>(), K, N2, (int)SK2, compact);                                            \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                        \
        return;                                                                                                \
    }
    // only the widths this checkpoint family actually pairs (3, 4, 5, 6 bits, mul1): every other case the caller's
    // compatibility test refuses, so the compile stays small
    TF_LAUNCH(6, 2) TF_LAUNCH(8, 2) TF_LAUNCH(10, 2) TF_LAUNCH(12, 2)
#undef TF_LAUNCH
    TORCH_CHECK(false, "unsupported EXL3 width/codebook for a pair launch: K2=", K2v, " codebook=", cb);
}

void exl3_unpack_cuda(const at::Tensor& T, at::Tensor& W, int64_t stride_k, int64_t stride_nb, int64_t K2,
                      int64_t cb) {
    const int K = (int)W.size(0), N = (int)W.size(1);
    dim3 grid((unsigned)(N / 16), (unsigned)(K / 16));
    auto stream = at::cuda::getCurrentCUDAStream();
#define TF_LAUNCH(K2_, CB_)                                                                                         \
    if (K2 == K2_ && cb == CB_) {                                                                                \
        unpack_kernel<K2_, CB_><<<grid, 32, 0, stream>>>(reinterpret_cast<const uint32_t*>(T.data_ptr()),         \
                                                        reinterpret_cast<half*>(W.data_ptr()), N, stride_k,      \
                                                        stride_nb);                                              \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                          \
        return;                                                                                                  \
    }
    TF_EXL3_ALL(TF_LAUNCH)
#undef TF_LAUNCH
    TORCH_CHECK(false, "unsupported EXL3 width/codebook: K2=", K2, " codebook=", cb);
}
