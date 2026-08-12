// AnimaXS Metal kernels — A12 / Apple5 targets only.
// No Metal 3 APIs, no simdgroup_matrix, no bfloat16, no MTLIOCommandQueue.
// All norm/softmax reductions accumulate in float (fp32). Residual stream stays fp32.
//
// W4 layout (see AnimapkFile / HANDOFF.md §12): along K (input dim), group=64,
// q unsigned 0..15, even K → low nibble, odd K → high nibble; scale/zero fp16 per group.
// value = q * scale + zero.

#include <metal_stdlib>
using namespace metal;

constant uint W4_GROUP = 64;

// ---------------------------------------------------------------------------
// Dequantization: packed W4 → fp16 buffer.
// data: packed nibbles (K/2 bytes), scale/zero: fp16 arrays of ceil(K/64) per row.
// For a row-major [R, K] tensor, thread computes one output element (r, k).
// ---------------------------------------------------------------------------
kernel void dequant_w4_to_half(
    device const uchar  *packed    [[buffer(0)]],
    device const half   *scale     [[buffer(1)]],
    device const half   *zero      [[buffer(2)]],
    device half         *out       [[buffer(3)]],
    constant uint       &K         [[buffer(4)]],
    constant uint       &rowStride [[buffer(5)]],   // packed bytes per row = K/2 (may include pad)
    constant uint       &rows      [[buffer(6)]],
    constant uint       &outStride [[buffer(7)]],   // fp16 elements per output row
    uint2 gid [[thread_position_in_grid]])          // gid.x = k, gid.y = row
{
    uint k = gid.x;
    uint r = gid.y;
    if (k >= K || r >= rows) return;
    uint byteIdx = r * rowStride + (k >> 1);
    uchar b = packed[byteIdx];
    uint q = (k & 1) == 0 ? uint(b & 0x0F) : uint(b >> 4);
    uint g = k / W4_GROUP;
    uint groupsPerRow = (K + W4_GROUP - 1) / W4_GROUP;
    half sc = scale[r * groupsPerRow + g];
    half ze = zero[r * groupsPerRow + g];
    out[r * outStride + k] = half(float(q) * float(sc) + float(ze));
}

// ---------------------------------------------------------------------------
// Dequantization: packed W8 → fp16 buffer.
// ---------------------------------------------------------------------------
kernel void dequant_w8_to_half(
    device const uchar  *packed    [[buffer(0)]],
    device const half   *scale     [[buffer(1)]],
    device const half   *zero      [[buffer(2)]],
    device half         *out       [[buffer(3)]],
    constant uint       &K         [[buffer(4)]],
    constant uint       &rowStride [[buffer(5)]],
    constant uint       &rows      [[buffer(6)]],
    constant uint       &outStride [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint k = gid.x;
    uint r = gid.y;
    if (k >= K || r >= rows) return;
    uint q = uint(packed[r * rowStride + k]);
    uint g = k / W4_GROUP;
    uint groupsPerRow = (K + W4_GROUP - 1) / W4_GROUP;
    half sc = scale[r * groupsPerRow + g];
    half ze = zero[r * groupsPerRow + g];
    out[r * outStride + k] = half(float(q) * float(sc) + float(ze));
}

kernel void copy_half_rows(
    device const half *source      [[buffer(0)]],
    device half       *destination [[buffer(1)]],
    constant uint     &columns     [[buffer(2)]],
    constant uint     &rows        [[buffer(3)]],
    constant uint     &sourceStride [[buffer(4)]],
    constant uint     &destinationStride [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= columns || gid.y >= rows) return;
    destination[gid.y * destinationStride + gid.x] = source[gid.y * sourceStride + gid.x];
}

kernel void float_to_half(
    device const float *source [[buffer(0)]],
    device half *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) destination[gid] = half(source[gid]);
}

kernel void half_to_float(
    device const half *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) destination[gid] = float(source[gid]);
}

// Convert between projection layout [tokens,heads,headDim] and the head-major
// [heads,tokens,headDim] layout consumed by AttentionExecutor. The same kernel
// handles both directions so the layouts cannot drift independently.
kernel void transpose_token_head_half(
    device const half *source [[buffer(0)]],
    device half *destination [[buffer(1)]],
    constant uint &tokens [[buffer(2)]],
    constant uint &heads [[buffer(3)]],
    constant uint &headDim [[buffer(4)]],
    constant uint &toHeadMajor [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint count = tokens * heads * headDim;
    if (gid >= count) return;
    uint dim = gid % headDim;
    uint row = gid / headDim;
    uint head = row % heads;
    uint token = row / heads;
    uint headMajor = (head * tokens + token) * headDim + dim;
    if (toHeadMajor != 0) destination[headMajor] = source[gid];
    else destination[gid] = source[headMajor];
}

// ---------------------------------------------------------------------------
// RMSNorm (fp32 input/output and statistics; fp16 packed weight).
// Dispatch exactly one threadgroup per row, with at most 256 threads.
// ---------------------------------------------------------------------------
kernel void rmsnorm_f32_to_f32(
    device const float *in      [[buffer(0)]],
    device float       *out     [[buffer(1)]],
    device const half  *weight  [[buffer(2)]],
    constant uint      &N       [[buffer(3)]],
    constant float     &eps     [[buffer(4)]],
    constant uint      &useWt   [[buffer(5)]],
    constant uint      &rows    [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    uint threadCount = threads.x;
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint i = tid; i < N; i += threadCount) {
        float v = in[row * N + i];
        sum = fma(v, v, sum);
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(N) + eps);
    for (uint i = tid; i < N; i += threadCount) {
        float w = useWt != 0 ? float(weight[i]) : 1.0f;
        out[row * N + i] = in[row * N + i] * inv * w;
    }
}

// ---------------------------------------------------------------------------
// Mean-centered LayerNorm (no affine), fp32 throughout. One group per row.
// ---------------------------------------------------------------------------
kernel void layernorm_f32_to_f32(
    device const float *in   [[buffer(0)]],
    device float       *out  [[buffer(1)]],
    constant uint      &N    [[buffer(2)]],
    constant float     &eps  [[buffer(3)]],
    constant uint      &rows [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    uint threadCount = threads.x;
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint i = tid; i < N; i += threadCount) sum += in[row * N + i];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial[0] / float(N);
    float squareSum = 0.0f;
    for (uint i = tid; i < N; i += threadCount) {
        float centered = in[row * N + i] - mean;
        squareSum = fma(centered, centered, squareSum);
    }
    partial[tid] = squareSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(N) + eps);
    for (uint i = tid; i < N; i += threadCount) {
        out[row * N + i] = (in[row * N + i] - mean) * inv;
    }
}

// ---------------------------------------------------------------------------
// Exact reference activations, fp32 input/output with explicit bounds.
// ---------------------------------------------------------------------------
inline float erf_f32(float x)
{
    // Numerical Recipes complementary-error-function form. This follows the
    // source's erf GELU equation (not the tanh GELU approximation) and stays
    // within a few float32 ulps where Metal's Apple5 library lacks erf().
    float absolute = abs(x);
    float t = 1.0f / (1.0f + 0.5f * absolute);
    float polynomial = t * exp(
        -absolute * absolute - 1.26551223f +
        t * (1.00002368f +
        t * (0.37409196f +
        t * (0.09678418f +
        t * (-0.18628806f +
        t * (0.27886807f +
        t * (-1.13520398f +
        t * (1.48851587f +
        t * (-0.82215223f +
        t * 0.17087277f)))))))));
    float value = 1.0f - polynomial;
    return x < 0.0f ? -value : value;
}

kernel void gelu(
    device const float *in  [[buffer(0)]],
    device float       *out [[buffer(1)]],
    constant uint      &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float x = in[gid];
    out[gid] = 0.5f * x * (1.0f + erf_f32(x * 0.7071067811865475f));
}

kernel void silu(
    device const float *in  [[buffer(0)]],
    device float       *out [[buffer(1)]],
    constant uint      &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float x = in[gid];
    out[gid] = x / (1.0f + exp(-x));
}

// Qwen SwiGLU boundary: both MPS projections are fp16 and the down projection
// consumes fp16. Evaluate SiLU and multiplication in fp32, then round once.
kernel void gated_silu_half(
    device const half *gate [[buffer(0)]],
    device const half *up   [[buffer(1)]],
    device half       *out  [[buffer(2)]],
    constant uint &count    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float x = float(gate[gid]);
    out[gid] = half((x / (1.0f + exp(-x))) * float(up[gid]));
}

// Adapter MLP boundary: Linear and bias are fp16 model parameters/activations,
// while exact GELU is evaluated in fp32 before rounding back to fp16.
kernel void bias_gelu_half(
    device half *values [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    constant uint &columns [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float x = float(values[gid]) + float(bias[gid % columns]);
    values[gid] = half(0.5f * x * (1.0f + erf_f32(x * 0.7071067811865475f)));
}

kernel void add_bias_half(
    device half *values [[buffer(0)]],
    device const half *bias [[buffer(1)]],
    constant uint &columns [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) values[gid] += bias[gid % columns];
}

kernel void add_bias_half_into_float(
    device float *residual [[buffer(0)]],
    device const half *branch [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) residual[gid] += float(branch[gid]) + float(bias[gid % columns]);
}

// AdaLN: normalized * (1 + scale) + shift, all fp32; vectors broadcast by N.
kernel void modulate_f32(
    device const float *normalized [[buffer(0)]],
    device const float *scale      [[buffer(1)]],
    device const float *shift      [[buffer(2)]],
    device float       *out        [[buffer(3)]],
    constant uint      &N          [[buffer(4)]],
    constant uint      &count      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    uint column = gid % N;
    out[gid] = normalized[gid] * (1.0f + scale[column]) + shift[column];
}

// MPS branches are fp16, but gates and residual arithmetic remain fp32.
kernel void gate_add_half_f32(
    device float       *residual [[buffer(0)]],
    device const half  *branch   [[buffer(1)]],
    device const float *gate     [[buffer(2)]],
    constant uint      &N        [[buffer(3)]],
    constant uint      &count    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    residual[gid] += float(branch[gid]) * gate[gid % N];
}

// ---------------------------------------------------------------------------
// Add half branch into fp32 residual (gate == 1).
// ---------------------------------------------------------------------------
kernel void add_half_into_float(
    device float       *residual [[buffer(0)]],
    device const half  *branch   [[buffer(1)]],
    constant uint      &count    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    residual[gid] += float(branch[gid]);
}

kernel void add_f32(
    device float *destination [[buffer(0)]],
    device const float *source [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) destination[gid] += source[gid];
}

// Per-token/per-head RMSNorm at an fp16 MPS boundary, without RoPE (cross attention).
// Input/output layout is [tokens,heads,128], and the [128] fp16 weight is shared.
kernel void rmsnorm_heads_half(
    device const half *in [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    uint threadCount = threads.x;
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint d = tid; d < 128; d += threadCount) {
        float value = float(in[row * 128 + d]);
        sum = fma(value, value, sum);
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / 128.0f + eps);
    for (uint d = tid; d < 128; d += threadCount) {
        out[row * 128 + d] = half(float(in[row * 128 + d]) * inv * float(weight[d]));
    }
}

// ---------------------------------------------------------------------------
// DiT Q/K fused RMSNorm + split-half RoPE. Input/output [tokens,heads,128]
// fp16; statistics and rotation arithmetic fp32. One 64-thread group per
// token/head. The shared [128] weight is reused across every head.
// ---------------------------------------------------------------------------
kernel void rms_rope_split_half(
    device const half  *in         [[buffer(0)]],
    device const half  *weight     [[buffer(1)]],
    device const float *rope       [[buffer(2)]], // [tokens,64,2,2]
    device half        *out        [[buffer(3)]],
    constant uint      &tokens     [[buffer(4)]],
    constant uint      &heads      [[buffer(5)]],
    constant float     &eps        [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint p [[thread_index_in_threadgroup]])
{
    uint row = group.x;
    if (row >= tokens * heads || p >= 64) return;
    uint token = row / heads;
    uint base = row * 128;
    float first = float(in[base + p]);
    float second = float(in[base + p + 64]);
    threadgroup float partial[64];
    partial[p] = fma(first, first, second * second);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (p < stride) partial[p] += partial[p + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / 128.0f + eps);
    float a = first * inv * float(weight[p]);
    float b = second * inv * float(weight[p + 64]);
    uint ropeBase = (token * 64 + p) * 4;
    float c = rope[ropeBase];
    float negativeSine = rope[ropeBase + 1];
    float sine = rope[ropeBase + 2];
    out[base + p] = half(c * a + negativeSine * b);
    out[base + p + 64] = half(sine * a + c * b);
}

// LLM adapter Q/K norm + rotate_half RoPE. Adapter heads are 64 wide and its
// theta-10000 rope therefore contains 32 two-dimensional rotation blocks/token.
kernel void rms_rope_adapter64(
    device const half *in [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const float *rope [[buffer(2)]], // [tokens,32,2,2]
    device half *out [[buffer(3)]],
    constant uint &tokens [[buffer(4)]],
    constant uint &heads [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint p [[thread_index_in_threadgroup]])
{
    uint row = group.x;
    if (row >= tokens * heads || p >= 32) return;
    uint token = row / heads;
    uint base = row * 64;
    float first = float(in[base + p]);
    float second = float(in[base + p + 32]);
    threadgroup float partial[32];
    partial[p] = fma(first, first, second * second);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 16; stride > 0; stride >>= 1) {
        if (p < stride) partial[p] += partial[p + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / 64.0f + eps);
    float a = first * inv * float(weight[p]);
    float b = second * inv * float(weight[p + 32]);
    uint ropeBase = (token * 32 + p) * 4;
    out[base + p] = half(rope[ropeBase] * a + rope[ropeBase + 1] * b);
    out[base + p + 32] = half(rope[ropeBase + 2] * a + rope[ropeBase + 3] * b);
}

// Final adapter RMSNorm and per-token T5 weighting. Padding is zeroed by the
// caller before dispatch, so only the real target rows are written.
kernel void rmsnorm_half_to_weighted_f32(
    device const half *in [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const float *tokenWeight [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint d = tid; d < 1024; d += threads.x) {
        float value = float(in[row * 1024 + d]);
        sum = fma(value, value, sum);
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / 1024.0f + eps);
    float multiplier = tokenWeight[row];
    for (uint d = tid; d < 1024; d += threads.x) {
        out[row * 1024 + d] = float(in[row * 1024 + d]) * inv * float(weight[d]) * multiplier;
    }
}

// ---------------------------------------------------------------------------
// Direct packed W4 matrix-vector product for precision-sensitive M=1 linears.
// Weight is [rows,K], input [K], output [rows]. Quant groups reset per row.
// Dispatch one power-of-two threadgroup (up to 256 threads) per output row.
// ---------------------------------------------------------------------------
kernel void w4_matvec_f32(
    device const uchar *packed     [[buffer(0)]],
    device const half  *scale      [[buffer(1)]],
    device const half  *zero       [[buffer(2)]],
    device const float *input      [[buffer(3)]],
    device float       *out        [[buffer(4)]],
    constant uint      &K          [[buffer(5)]],
    constant uint      &rows       [[buffer(6)]],
    constant uint      &rowStride  [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    uint threadCount = threads.x;
    uint groupsPerRow = (K + W4_GROUP - 1) / W4_GROUP;
    float sum = 0.0f;
    for (uint k = tid; k < K; k += threadCount) {
        uchar byte = packed[row * rowStride + (k >> 1)];
        uint q = (k & 1) == 0 ? uint(byte & 0x0F) : uint(byte >> 4);
        uint quantGroup = row * groupsPerRow + k / W4_GROUP;
        float value = float(q) * float(scale[quantGroup]) + float(zero[quantGroup]);
        sum = fma(input[k], value, sum);
    }
    threadgroup float partial[256];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[row] = partial[0];
}

// Stable row softmax. Scores/probabilities are fp16 at MPS boundaries, while
// max, exp, and sum are evaluated and reduced in fp32.
kernel void attention_softmax_rows(
    device half       *scores    [[buffer(0)]],
    constant uint     &rows      [[buffer(1)]],
    constant uint     &columns   [[buffer(2)]],
    constant uint     &queryBase [[buffer(3)]],
    constant uint     &causal    [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    uint threadCount = threads.x;
    threadgroup float partial[256];
    float localMax = -INFINITY;
    for (uint column = tid; column < columns; column += threadCount) {
        if (causal == 0 || column <= queryBase + row) {
            localMax = max(localMax, float(scores[row * columns + column]));
        }
    }
    partial[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rowMax = partial[0];
    float localSum = 0.0f;
    for (uint column = tid; column < columns; column += threadCount) {
        float probability = 0.0f;
        if (causal == 0 || column <= queryBase + row) {
            probability = exp(float(scores[row * columns + column]) - rowMax);
        }
        localSum += probability;
    }
    partial[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverseSum = 1.0f / partial[0];
    for (uint column = tid; column < columns; column += threadCount) {
        float probability = 0.0f;
        if (causal == 0 || column <= queryBase + row) {
            probability = exp(float(scores[row * columns + column]) - rowMax) * inverseSum;
        }
        scores[row * columns + column] = half(probability);
    }
}

// ---------------------------------------------------------------------------
// FLOW model-sampling conversion (fp32): denoised = x - sigma * velocity.
// Kept separate from Euler so the production operation order matches ComfyUI.
// ---------------------------------------------------------------------------
kernel void flow_velocity_to_denoised_f32(
    device const float *x        [[buffer(0)]],
    device const float *velocity [[buffer(1)]],
    device float       *denoised [[buffer(2)]],
    constant float     &sigma    [[buffer(3)]],
    constant uint      &count    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    denoised[gid] = x[gid] - sigma * velocity[gid];
}

// ---------------------------------------------------------------------------
// Euler flow step (fp32): x_next = x + dSigma * (x - denoised) / sigma
// ---------------------------------------------------------------------------
kernel void euler_step_f32(
    device const float *x        [[buffer(0)]],
    device const float *denoised [[buffer(1)]],
    device float       *xNext    [[buffer(2)]],
    constant float     &sigma    [[buffer(3)]],
    constant float     &dSigma   [[buffer(4)]],
    constant uint      &count    [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float inv = 1.0f / sigma;
    xNext[gid] = x[gid] + dSigma * (x[gid] - denoised[gid]) * inv;
}

// ---------------------------------------------------------------------------
// Patchify: input [17, 64, 64] (C,H,W fp32) → tokens [32*32, 68].
// For each 2x2 spatial patch at (i,j), token = concat of 17 channels × 4 pixels
// in (c, i0, j0, i1, j1) order matching the reference rearrange.
// ---------------------------------------------------------------------------
kernel void patchify17(
    device const float *input  [[buffer(0)]],  // [C=17, H=64, W=64]
    device float       *tokens [[buffer(1)]],  // [T=1024, 68]
    constant uint      &H      [[buffer(2)]],
    constant uint      &W      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])     // gid.x = token index, gid.y = channel
{
    uint t = gid.x;
    uint c = gid.y;
    uint patchW = W >> 1;
    uint patchH = H >> 1;
    if (t >= patchH * patchW || c >= 17) return;
    uint i = t / patchW;         // patch row
    uint j = t % patchW;         // patch col
    // gather 4 pixels for channel c
    for (uint di = 0; di < 2; ++di) {
        for (uint dj = 0; dj < 2; ++dj) {
            float v = input[c * H * W + (i * 2 + di) * W + (j * 2 + dj)];
            uint outIdx = t * 68 + c * 4 + di * 2 + dj;
            tokens[outIdx] = v;
        }
    }
}


// Reverse the first 16 channels of patchify output [tokens,68] to [16,H,W].
kernel void unpatchify16(
    device const float *tokens [[buffer(0)]],
    device float       *output [[buffer(1)]],
    constant uint      &H      [[buffer(2)]],
    constant uint      &W      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.x;
    uint c = gid.y;
    uint patchW = W >> 1;
    uint patchH = H >> 1;
    if (t >= patchH * patchW || c >= 16) return;
    uint i = t / patchW;
    uint j = t % patchW;
    for (uint di = 0; di < 2; ++di) {
        for (uint dj = 0; dj < 2; ++dj) {
            uint tokenIndex = t * 68 + c * 4 + di * 2 + dj;
            output[c * H * W + (i * 2 + di) * W + (j * 2 + dj)] = tokens[tokenIndex];
        }
    }
}

// FinalLayer projects each 2x2 patch to exactly 16 * 2 * 2 = 64 values.
// Keep this separate from unpatchify16, whose 68-wide input includes four
// patchified mask channels.
kernel void unpatchify_velocity16(
    device const float *tokens [[buffer(0)]],
    device float       *output [[buffer(1)]],
    constant uint      &H      [[buffer(2)]],
    constant uint      &W      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.x;
    uint c = gid.y;
    uint patchW = W >> 1;
    uint patchH = H >> 1;
    if (t >= patchH * patchW || c >= 16) return;
    uint i = t / patchW;
    uint j = t % patchW;
    for (uint di = 0; di < 2; ++di) {
        for (uint dj = 0; dj < 2; ++dj) {
            // einops: (p1 p2 t C), with temporal patch size one and C fastest.
            uint tokenIndex = t * 64 + (di * 2 + dj) * 16 + c;
            output[c * H * W + (i * 2 + di) * W + (j * 2 + dj)] = tokens[tokenIndex];
        }
    }
}

// ---------------------------------------------------------------------------
// Wan VAE T=1 kernels (D052/D053/D060).
// Activations are position-major [height*width, channels] fp16 (channels
// contiguous), matching the decoder's internal HWC layout. All reductions
// accumulate in fp32. Convolution is implemented by the host as tiled
// im2col + MPS GEMM; these kernels cover norm/activation/upsample/elementwise
// work only.
// ---------------------------------------------------------------------------

// Channel-wise Wan RMS norm: F.normalize over C at each position, x sqrt(C),
// x learned gamma. Input/output HWC fp16 [positions, channels].
kernel void vae_channel_rmsnorm_half(
    device const half  *input     [[buffer(0)]],
    device const half  *gamma     [[buffer(1)]],
    device half        *output    [[buffer(2)]],
    constant uint      &positions [[buffer(3)]],
    constant uint      &channels  [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= positions) return;
    const uint base = gid * channels;
    float sum = 0.0f;
    for (uint c = 0; c < channels; ++c) {
        float v = float(input[base + c]);
        sum += v * v;
    }
    float inverse = 1.0f / max(sqrt(sum), 1e-12f);
    float scale = sqrt(float(channels));
    for (uint c = 0; c < channels; ++c) {
        float v = float(input[base + c]);
        output[base + c] = half(v * inverse * scale * float(gamma[c]));
    }
}

// SiLU on fp16 element buffer (identical math to the fp32 `silu` kernel).
kernel void silu_half(
    device const half *input  [[buffer(0)]],
    device half       *output [[buffer(1)]],
    constant uint     &count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float x = float(input[gid]);
    output[gid] = half(x / (1.0f + exp(-x)));
}

// Nearest-exact 2x upsample of a position-major HWC fp16 tensor.
// [H,W,C] -> [2H,2W,C]; out(y,x,c) = in(y/2, x/2, c).
kernel void vae_nearest_exact_2x_half(
    device const half  *input  [[buffer(0)]],
    device half        *output [[buffer(1)]],
    constant uint      &height [[buffer(2)]],
    constant uint      &width  [[buffer(3)]],
    constant uint      &channels [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])  // gid.x = output pixel, gid.y = channel
{
    const uint outPixels = height * 2 * width * 2;
    const uint pixel = gid.x;
    if (pixel >= outPixels || gid.y >= channels) return;
    const uint outWidth = width * 2;
    const uint outY = pixel / outWidth;
    const uint outX = pixel % outWidth;
    const uint inY = outY / 2;
    const uint inX = outX / 2;
    output[pixel * channels + gid.y] = input[(inY * width + inX) * channels + gid.y];
}

// Fused nearest-exact 2x + 3x3 im2col gather: builds one output-position tile
// for a padded 3x3 convolution whose INPUT is the nearest-exact upsample of a
// source tensor. Reads source directly (no enlarged temporary), with zero
// padding applied in the upsample coordinate space.
//   source  HWC fp16 [srcH*srcW, channels]
//   output  fp16 [tileRows, channels*9] in PyTorch im2col order
//           (channel-major: (c*3+ky)*3+kx)
// Each thread produces one row (one output pixel) fully: loop over 9 taps.
kernel void vae_im2col_upsample3x3_half(
    device const half  *source   [[buffer(0)]],
    device half        *output   [[buffer(1)]],
    constant uint      &srcH     [[buffer(2)]],
    constant uint      &srcW     [[buffer(3)]],
    constant uint      &channels [[buffer(4)]],
    constant uint      &outH     [[buffer(5)]],
    constant uint      &outW     [[buffer(6)]],
    constant uint      &tileRows [[buffer(7)]],
    constant uint      &tileBase [[buffer(8)]],
    constant uint      &outStride [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= tileRows) return;
    const uint outPixel = tileBase + gid;
    if (outPixel >= outH * outW) return;
    const uint outY = outPixel / outW;
    const uint outX = outPixel % outW;
    const uint rowBase = gid * outStride;
    for (uint ky = 0; ky < 3; ++ky) {
        for (uint kx = 0; kx < 3; ++kx) {
            // Convolution input coordinate in the UPSCALED space.
            const int inY = int(outY) + int(ky) - 1;
            const int inX = int(outX) + int(kx) - 1;
            // Zero padding at the upscaled-image border.
            if (inY < 0 || inX < 0 || inY >= int(outH) || inX >= int(outW)) {
                for (uint c = 0; c < channels; ++c) {
                    output[rowBase + (c * 3 + ky) * 3 + kx] = 0.0h;
                }
            } else {
                const uint srcY = uint(inY) / 2;
                const uint srcX = uint(inX) / 2;
                const uint srcBase = (srcY * srcW + srcX) * channels;
                for (uint c = 0; c < channels; ++c) {
                    output[rowBase + (c * 3 + ky) * 3 + kx] = source[srcBase + c];
                }
            }
        }
    }
}

// Plain 3x3 im2col gather (no upsample) for position-major HWC fp16 input.
kernel void vae_im2col3x3_half(
    device const half  *input    [[buffer(0)]],
    device half        *output   [[buffer(1)]],
    constant uint      &height   [[buffer(2)]],
    constant uint      &width    [[buffer(3)]],
    constant uint      &channels [[buffer(4)]],
    constant uint      &tileRows [[buffer(5)]],
    constant uint      &tileBase [[buffer(6)]],
    constant uint      &outStride [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= tileRows) return;
    const uint pixel = tileBase + gid;
    if (pixel >= height * width) return;
    const uint y = pixel / width;
    const uint x = pixel % width;
    const uint rowBase = gid * outStride;
    for (uint ky = 0; ky < 3; ++ky) {
        for (uint kx = 0; kx < 3; ++kx) {
            const int inY = int(y) + int(ky) - 1;
            const int inX = int(x) + int(kx) - 1;
            if (inY < 0 || inX < 0 || inY >= int(height) || inX >= int(width)) {
                for (uint c = 0; c < channels; ++c) {
                    output[rowBase + (c * 3 + ky) * 3 + kx] = 0.0h;
                }
            } else {
                const uint srcBase = (uint(inY) * width + uint(inX)) * channels;
                for (uint c = 0; c < channels; ++c) {
                    output[rowBase + (c * 3 + ky) * 3 + kx] = input[srcBase + c];
                }
            }
        }
    }
}

// Elementwise add of two HWC fp16 buffers into a third (residual add / identity).
kernel void vae_add_half(
    device const half *lhs  [[buffer(0)]],
    device const half *rhs  [[buffer(1)]],
    device half       *out  [[buffer(2)]],
    constant uint     &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = half(float(lhs[gid]) + float(rhs[gid]));
}

// Copy fp16 buffer (used when a stage output must survive a scratch swap).
kernel void copy_half(
    device const half *input  [[buffer(0)]],
    device half       *output [[buffer(1)]],
    constant uint     &count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    output[gid] = input[gid];
}

// Fold a conv weight into row-padded fp16 scratch for the MPS GEMM path.
// Source is PyTorch layout [Cout, Cin, (KT,) KH, KW] fp16 contiguous. For rank-5
// causal weights, only the FINAL temporal slice (index KT-1) is used (D052).
// Output row o,c is written at scratch[(o*outStride) + (c*KH+ky)*KW+kx], so the
// im2col column order (c-major, then ky, then kx) matches exactly.
kernel void vae_fold_weight_half(
    device const half  *source     [[buffer(0)]],
    device half        *output     [[buffer(1)]],
    constant uint      &channelsIn [[buffer(2)]],
    constant uint      &channelsOut [[buffer(3)]],
    constant uint      &kh         [[buffer(4)]],
    constant uint      &kw         [[buffer(5)]],
    constant uint      &kt         [[buffer(6)]],   // temporal kernel size (1 = rank-4)
    constant uint      &outStride  [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    const uint kernelElements = kh * kw;
    const uint total = channelsOut * channelsIn * kernelElements;
    if (gid >= total) return;
    const uint k = gid % kernelElements;          // (ky*kw + kx)
    const uint c = (gid / kernelElements) % channelsIn;
    const uint o = gid / (kernelElements * channelsIn);
    const uint ky = k / kw, kx = k % kw;
    // Final temporal slice offset within each (o,c) plane: (KT-1)*KH*KW.
    const uint temporalSkip = (kt > 1) ? (kt - 1) * kernelElements : 0;
    const uint srcIndex = ((o * channelsIn + c) * kt * kernelElements) + temporalSkip + k;
    const uint dstIndex = o * outStride + (c * kh + ky) * kw + kx;
    output[dstIndex] = source[srcIndex];
}

// Latent [C,H,W] fp32 (channel-major) -> position-major fp16 [H*W, C].
// D060: the sampler's final latent is consumed unchanged (no mean/std denorm).
kernel void vae_latent_to_position_half(
    device const float *latent  [[buffer(0)]],
    device half        *output  [[buffer(1)]],
    constant uint      &height  [[buffer(2)]],
    constant uint      &width   [[buffer(3)]],
    constant uint      &channels [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])  // gid.x = pixel, gid.y = channel
{
    const uint pixels = height * width;
    if (gid.x >= pixels || gid.y >= channels) return;
    const uint y = gid.x / width, x = gid.x % width;
    output[gid.x * channels + gid.y] = half(latent[gid.y * pixels + y * width + x]);
}

// Position-major fp16 [H*W, C] -> fp32 channel-major [C,H,W] (final RGB).
kernel void vae_position_to_rgb_f32(
    device const half *positioned [[buffer(0)]],
    device float      *output     [[buffer(1)]],
    constant uint     &height     [[buffer(2)]],
    constant uint     &width      [[buffer(3)]],
    constant uint     &channels   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])  // gid.x = pixel, gid.y = channel
{
    const uint pixels = height * width;
    if (gid.x >= pixels || gid.y >= channels) return;
    const uint y = gid.x / width, x = gid.x % width;
    output[gid.y * pixels + y * width + x] = float(positioned[gid.x * channels + gid.y]);
}

// Split an interleaved position-major [positions, 3C] fp16 buffer (the 1x1
// to_qkv output) into three contiguous [positions, C] blocks so the
// AttentionExecutor's [heads, rows, headDim] layout can address Q/K/V directly.
// gid.x = position, gid.y = channel; output block b starts at b*positions*C.
kernel void vae_split_qkv_half(
    device const half *qkv   [[buffer(0)]],
    device half       *split [[buffer(1)]],
    constant uint     &positions [[buffer(2)]],
    constant uint     &channels  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= positions || gid.y >= channels) return;
    const half q = qkv[gid.x * channels * 3 + gid.y];
    const half k = qkv[gid.x * channels * 3 + channels + gid.y];
    const half v = qkv[gid.x * channels * 3 + 2 * channels + gid.y];
    split[gid.x * channels + gid.y] = q;
    split[positions * channels + gid.x * channels + gid.y] = k;
    split[2 * positions * channels + gid.x * channels + gid.y] = v;
}

// Decoder's fp16 position-major [H*W,3] RGB (~[-1,1]) -> RGBA8 interleaved
// (r,g,b,255), fusing the (rgb+1)/2 clamp. One thread per pixel. This is the
// J004 final image path: it avoids materializing a full [Float] RGB copy.
kernel void vae_position_to_rgba8(
    device const half *positioned [[buffer(0)]],
    device uchar4     *output     [[buffer(1)]],
    constant uint     &pixels     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= pixels) return;
    float r = float(positioned[gid * 3 + 0]);
    float g = float(positioned[gid * 3 + 1]);
    float b = float(positioned[gid * 3 + 2]);
    r = clamp((r + 1.0) * 0.5, 0.0, 1.0);
    g = clamp((g + 1.0) * 0.5, 0.0, 1.0);
    b = clamp((b + 1.0) * 0.5, 0.0, 1.0);
    output[gid] = uchar4(
        (uchar)(r * 255.0 + 0.5),
        (uchar)(g * 255.0 + 0.5),
        (uchar)(b * 255.0 + 0.5),
        255);
}
