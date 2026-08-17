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

// Round-to-nearest-even BF16 boundary while retaining fp32 storage. Preserve
// infinities and NaNs rather than allowing the integer bias to alter them.
// Shared by round_f32_to_bf16 and every fused kernel that must reproduce the
// exact same boundary (P3-A/P3-B): one definition, no duplicated logic.
inline float round_f32_to_bf16_value(float value)
{
    uint bits = as_type<uint>(value);
    uint exponent = bits & 0x7f800000u;
    if (exponent != 0x7f800000u) {
        bits += 0x00007fffu + ((bits >> 16) & 1u);
        bits &= 0xffff0000u;
    }
    return as_type<float>(bits);
}

kernel void round_f32_to_bf16(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    destination[gid] = round_f32_to_bf16_value(source[gid]);
}

// Round a fp16 value through BF16 while retaining the Apple5-friendly fp16
// storage format. Every BF16 value in the finite fp16 range is exactly
// representable as fp16, so this is a lossless storage conversion after the
// intended BF16 mantissa truncation. The kernel is safe in-place.
kernel void round_half_to_bf16(
    device const half *source [[buffer(0)]],
    device half *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float value = float(source[gid]);
    uint bits = as_type<uint>(value);
    uint exponent = bits & 0x7f800000u;
    if (exponent != 0x7f800000u) {
        bits += 0x00007fffu + ((bits >> 16) & 1u);
        bits &= 0xffff0000u;
    }
    destination[gid] = half(as_type<float>(bits));
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

// Exact GELU expression shared by every GELU kernel (fp32 in/out) so the
// fused half kernels cannot drift from the reference `gelu` math.
inline float gelu_f32(float x)
{
    return 0.5f * x * (1.0f + erf_f32(x * 0.7071067811865475f));
}

// Stats ABI shared by every probe kernel in this file (see the probe section
// for the full layout). Declared here, before their first use, so all fused
// P3-A/P3-B probe kernels and the classic probe kernels can reference them.
constant uint NUM_FLAG_NAN          = 1u << 0;
constant uint NUM_FLAG_POS_INF      = 1u << 1;
constant uint NUM_FLAG_NEG_INF      = 1u << 2;
constant uint NUM_FLAG_HALF_OVERFLOW = 1u << 3;
constant uint NUM_FLAG_RESULT_NAN   = 1u << 4;
constant uint NUM_FLAG_RESULT_INF   = 1u << 5;
constant uint NUM_SLOT_UINTS        = 4u;

// ---------------------------------------------------------------------------
// P3-B: in-place GELU on fp16 MLP hidden activations with fp32 register
// arithmetic. Replaces the legacy sequence half_to_float → gelu(f32) →
// round_f32_to_bf16 → float_to_half with a single pass:
//   y = gelu_f32(float(values[i]))
//   y = round_f32_to_bf16_value(y)          (only when bf16Boundary)
//   values[i] = half(y)
// Safe in-place: each thread reads exactly one element before writing it.
// ---------------------------------------------------------------------------
kernel void dit_gelu_half_inplace(
    device half *values [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    constant uint &bf16Boundary [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float y = gelu_f32(float(values[gid]));
    if (bf16Boundary != 0) y = round_f32_to_bf16_value(y);
    values[gid] = half(y);
}

// P3-B probe variant: identical math plus float_to_half_probe health stats on
// the value fed to half() (i.e. after the BF16 boundary, matching the legacy
// probed conversion path).
kernel void dit_gelu_half_inplace_probe(
    device half *values [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    constant uint &bf16Boundary [[buffer(2)]],
    device uint *stats [[buffer(3)]],
    constant uint &probeSlot [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint partialFlags[256];
    threadgroup uint partialMax[256];
    threadgroup uint partialIndex[256];
    uint base = group.x * 256;
    uint i = base + tid;
    uint localFlags = 0;
    uint localMax = 0;
    uint localIndex = 0xFFFFFFFFu;
    if (i < count) {
        float y = gelu_f32(float(values[i]));
        if (bf16Boundary != 0) y = round_f32_to_bf16_value(y);
        values[i] = half(y);
        if (isnan(y)) {
            localFlags |= NUM_FLAG_NAN;
            localIndex = min(localIndex, i);
        } else if (isinf(y)) {
            localFlags |= (y > 0.0f) ? NUM_FLAG_POS_INF : NUM_FLAG_NEG_INF;
            localIndex = min(localIndex, i);
        } else {
            float a = fabs(y);
            if (a > 65504.0f) {
                localFlags |= NUM_FLAG_HALF_OVERFLOW;
                localIndex = min(localIndex, i);
            }
            uint bits = as_type<uint>(a);
            if (bits > localMax) localMax = bits;
        }
    }
    partialFlags[tid] = localFlags;
    partialMax[tid] = localMax;
    partialIndex[tid] = localIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialFlags[tid] |= partialFlags[tid + stride];
            partialMax[tid] = max(partialMax[tid], partialMax[tid + stride]);
            partialIndex[tid] = min(partialIndex[tid], partialIndex[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && partialFlags[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        uint slotBase = probeSlot * NUM_SLOT_UINTS;
        atomic_fetch_or_explicit(&atom[slotBase + 0], partialFlags[0], memory_order_relaxed);
        atomic_fetch_min_explicit(&atom[slotBase + 2], partialIndex[0], memory_order_relaxed);
    }
    if (tid == 0 && partialMax[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        atomic_fetch_max_explicit(&atom[probeSlot * NUM_SLOT_UINTS + 1], partialMax[0], memory_order_relaxed);
    }
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

// ---------------------------------------------------------------------------
// P3-A: fused LayerNorm + AdaLN modulation + optional BF16 compute boundary +
// fp16 conversion in ONE pass. Replaces the legacy three-pass sequence
// layernorm_f32_to_f32 → modulate_f32 → (round_f32_to_bf16 + float_to_half)
// while preserving its exact reduction structure, modulation indexing
// (scale at offset modulationOffset, shift at offset 0), boundary placement,
// and conversion semantics.
//
// Legacy equivalence (all fp32):
//   norm    = (residual - mean) * rsqrt(var + eps)
//   mod     = norm * (1 + scale) + shift
//   bf16    = round_f32_to_bf16_value(mod)            (only when bf16Boundary)
//   output  = half(bf16)
// ---------------------------------------------------------------------------
kernel void dit_layernorm_modulate_to_half(
    device const float *residual      [[buffer(0)]],
    device const float *modulation    [[buffer(1)]],
    device half       *output         [[buffer(2)]],
    constant uint     &N              [[buffer(3)]],
    constant float    &eps            [[buffer(4)]],
    constant uint     &rows           [[buffer(5)]],
    constant uint     &modulationOffset [[buffer(6)]],
    constant uint     &bf16Boundary   [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    uint threadCount = threads.x;
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint i = tid; i < N; i += threadCount) sum += residual[row * N + i];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial[0] / float(N);
    float squareSum = 0.0f;
    for (uint i = tid; i < N; i += threadCount) {
        float centered = residual[row * N + i] - mean;
        squareSum = fma(centered, centered, squareSum);
    }
    partial[tid] = squareSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(N) + eps);
    // EXACT modulate_f32 indexing: scale at modulationOffset, shift at 0.
    device const float *scale = modulation + modulationOffset;
    device const float *shift = modulation;
    for (uint i = tid; i < N; i += threadCount) {
        float normalized = (residual[row * N + i] - mean) * inv;
        float modulated = normalized * (1.0f + scale[i]) + shift[i];
        if (bf16Boundary != 0) modulated = round_f32_to_bf16_value(modulated);
        output[row * N + i] = half(modulated);
    }
}

// P3-A probe variant: identical math plus the float_to_half_probe health
// stats (NaN/±Inf/|v| > 65504) accumulated into the shared stats buffer.
// The health check runs on the value fed to half(), i.e. after the BF16
// boundary, matching the legacy probed path.
kernel void dit_layernorm_modulate_to_half_probe(
    device const float *residual      [[buffer(0)]],
    device const float *modulation    [[buffer(1)]],
    device half       *output         [[buffer(2)]],
    constant uint     &N              [[buffer(3)]],
    constant float    &eps            [[buffer(4)]],
    constant uint     &rows           [[buffer(5)]],
    constant uint     &modulationOffset [[buffer(6)]],
    constant uint     &bf16Boundary   [[buffer(7)]],
    device uint       *stats          [[buffer(8)]],
    constant uint     &probeSlot      [[buffer(9)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    uint threadCount = threads.x;
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint i = tid; i < N; i += threadCount) sum += residual[row * N + i];
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = partial[0] / float(N);
    float squareSum = 0.0f;
    for (uint i = tid; i < N; i += threadCount) {
        float centered = residual[row * N + i] - mean;
        squareSum = fma(centered, centered, squareSum);
    }
    partial[tid] = squareSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(partial[0] / float(N) + eps);
    threadgroup uint partialFlags[256];
    threadgroup uint partialMax[256];
    threadgroup uint partialIndex[256];
    device const float *scale = modulation + modulationOffset;
    device const float *shift = modulation;
    uint localFlags = 0;
    uint localMax = 0;
    uint localIndex = 0xFFFFFFFFu;
    for (uint i = tid; i < N; i += threadCount) {
        float normalized = (residual[row * N + i] - mean) * inv;
        float modulated = normalized * (1.0f + scale[i]) + shift[i];
        if (bf16Boundary != 0) modulated = round_f32_to_bf16_value(modulated);
        output[row * N + i] = half(modulated);
        if (isnan(modulated)) {
            localFlags |= NUM_FLAG_NAN;
            localIndex = min(localIndex, row * N + i);
        } else if (isinf(modulated)) {
            localFlags |= (modulated > 0.0f) ? NUM_FLAG_POS_INF : NUM_FLAG_NEG_INF;
            localIndex = min(localIndex, row * N + i);
        } else {
            float a = fabs(modulated);
            if (a > 65504.0f) {
                localFlags |= NUM_FLAG_HALF_OVERFLOW;
                localIndex = min(localIndex, row * N + i);
            }
            uint bits = as_type<uint>(a);
            if (bits > localMax) localMax = bits;
        }
    }
    partialFlags[tid] = localFlags;
    partialMax[tid] = localMax;
    partialIndex[tid] = localIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialFlags[tid] |= partialFlags[tid + stride];
            partialMax[tid] = max(partialMax[tid], partialMax[tid + stride]);
            partialIndex[tid] = min(partialIndex[tid], partialIndex[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && partialFlags[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        uint slotBase = probeSlot * NUM_SLOT_UINTS;
        atomic_fetch_or_explicit(&atom[slotBase + 0], partialFlags[0], memory_order_relaxed);
        atomic_fetch_min_explicit(&atom[slotBase + 2], partialIndex[0], memory_order_relaxed);
    }
    if (tid == 0 && partialMax[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        atomic_fetch_max_explicit(&atom[probeSlot * NUM_SLOT_UINTS + 1], partialMax[0], memory_order_relaxed);
    }
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

// Direct packed W8 matrix-vector product. Keep the same thread assignment,
// reduction order, and fp32 accumulation as w4_matvec_f32; only the packed
// weight decode changes from nibbles to one byte per K element.
kernel void w8_matvec_f32(
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
        uint q = uint(packed[row * rowStride + k]);
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

kernel void fp16_matvec_f32(
    device const half  *packed     [[buffer(0)]],
    device const half  *unusedScale [[buffer(1)]],
    device const half  *unusedZero [[buffer(2)]],
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
    float sum = 0.0f;
    for (uint k = tid; k < K; k += threadCount) {
        sum = fma(input[k], float(packed[row * rowStride + k]), sum);
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

// ---------------------------------------------------------------------------
// P8: direct packed quantized GEMM (runbook §13). The packed W4/W8 weight
// [N,K] is consumed DIRECTLY — the W tile for one K group (K=64) is decoded
// into threadgroup memory and reused across the TM activation rows, so the
// full [N,K] fp16 weight scratch of the dequantized-MPS path is never
// allocated. Decode semantics are EXACTLY those of dequant_w4_to_half /
// dequant_w8_to_half (group K = 64, q unsigned, even-K low nibble / odd-K
// high nibble for W4, value = q*scale + zero, scale/zero fp16 per (row,
// group)). Input is [M,K] fp16 row-major; output is [M,N] fp16 row-major
// written as half(FP32 accumulator). Threads are laid out TM×TN — one thread
// per output element — with the K reduction vectorized 4 halves at a time.
// Profiles: qgemm_8x8x64 (TM=8, TN=8, 64 threads) and qgemm_8x16x64 (TM=8,
// TN=16, 128 threads).
// ---------------------------------------------------------------------------
// Direct packed W4/W8 GEMM (P8). Quant group K = 64, so TK = 64. Each K
// group's [TN][64] W tile is decoded ONCE into threadgroup memory and reused
// across TM activation rows; accumulation is FP32; output is half. Two tile
// profiles: 8x8 (64 threads) and 8x16 (128 threads). Generated as concrete
// kernels via a macro because MSL does not allow template-parameter structs.
// The W4/W8 decode matches dequant_w4_to_half / dequant_w8_to_half exactly.
#define QGEMM_KERNEL(KNAME, TM, TN, ISW4)                                             \
kernel void KNAME(                                                                    \
    device const half   *input       [[buffer(0)]],  /* [M, K] fp16 */                \
    device const uchar  *packed      [[buffer(1)]],  /* [N, packedRowStride] */       \
    device const half   *scale       [[buffer(2)]],  /* [N, groupsPerRow] */          \
    device const half   *zero        [[buffer(3)]],  /* [N, groupsPerRow] */          \
    device half         *output      [[buffer(4)]],  /* [M, N] fp16 */                \
    constant uint       &M           [[buffer(5)]],                                  \
    constant uint       &N           [[buffer(6)]],                                  \
    constant uint       &K           [[buffer(7)]],                                  \
    constant uint       &rowStride   [[buffer(8)]],                                  \
    constant uint       &inputStride [[buffer(9)]],                                  \
    constant uint       &outputStride [[buffer(10)]],                                \
    uint3 group [[threadgroup_position_in_grid]],                                    \
    uint tid [[thread_index_in_threadgroup]])                                        \
{                                                                                     \
    constexpr uint TK = 64;                                                          \
    threadgroup half aTile[TM * TK];                                                 \
    threadgroup half wTile[TN * TK];                                                 \
    uint mBase = group.x * TM;                                                       \
    uint nBase = group.y * TN;                                                       \
    if (mBase >= M || nBase >= N) return;                                            \
    uint mCount = min((uint)TM, M - mBase);                                                \
    uint nCount = min((uint)TN, N - nBase);                                                \
    uint groupsPerRow = (K + W4_GROUP - 1) / W4_GROUP;                               \
    uint mRow = tid / TN;                                                            \
    uint nCol = tid % TN;                                                            \
    uint row = mBase + mRow;                                                         \
    uint wRow = nBase + nCol;                                                        \
    float acc = 0.0f;                                                                \
    uint kBase = 0;                                                                  \
    while (kBase < K) {                                                              \
        uint kCount = min((uint)TK, K - kBase);                                            \
        for (uint k = nCol; k < TK; k += TN) {                                       \
            if (k < kCount) {                                                        \
                aTile[mRow * TK + k] = input[row * inputStride + kBase + k];         \
            } else {                                                                 \
                aTile[mRow * TK + k] = 0;                                            \
            }                                                                        \
        }                                                                            \
        {                                                                            \
            uint packedBase = wRow * rowStride;                                      \
            uint paramBase = wRow * groupsPerRow;                                    \
            uint g = kBase / W4_GROUP;                                               \
            half sc = scale[paramBase + g];                                          \
            half ze = zero[paramBase + g];                                           \
            if (ISW4) {                                                              \
                for (uint i = 0; i < TK / TM; ++i) {                                 \
                    uint k = mRow * (TK / TM) + i;                                   \
                    if (k < kCount) {                                                \
                        uchar byte = packed[packedBase + ((kBase + k) >> 1)];        \
                        uint q = ((kBase + k) & 1) == 0 ? uint(byte & 0x0F) : uint(byte >> 4); \
                        wTile[nCol * TK + k] = half(float(q) * float(sc) + float(ze)); \
                    } else {                                                         \
                        wTile[nCol * TK + k] = 0;                                    \
                    }                                                                \
                }                                                                    \
            } else {                                                                 \
                for (uint i = 0; i < TK / TM; ++i) {                                 \
                    uint k = mRow * (TK / TM) + i;                                   \
                    if (k < kCount) {                                                \
                        uint q = uint(packed[packedBase + kBase + k]);               \
                        wTile[nCol * TK + k] = half(float(q) * float(sc) + float(ze)); \
                    } else {                                                         \
                        wTile[nCol * TK + k] = 0;                                    \
                    }                                                                \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                             \
        for (uint k = 0; k < TK; k += 4) {                                           \
            float a0 = float(aTile[mRow * TK + k + 0]);                              \
            float a1 = float(aTile[mRow * TK + k + 1]);                              \
            float a2 = float(aTile[mRow * TK + k + 2]);                              \
            float a3 = float(aTile[mRow * TK + k + 3]);                              \
            uint wBase = nCol * TK + k;                                              \
            acc = fma(a0, float(wTile[wBase + 0]), acc);                             \
            acc = fma(a1, float(wTile[wBase + 1]), acc);                             \
            acc = fma(a2, float(wTile[wBase + 2]), acc);                             \
            acc = fma(a3, float(wTile[wBase + 3]), acc);                             \
        }                                                                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                             \
        kBase += TK;                                                                 \
    }                                                                                \
    if (mRow < mCount && nCol < nCount) {                                            \
        output[row * outputStride + nBase + nCol] = half(acc);                       \
    }                                                                                \
}

QGEMM_KERNEL(qgemm_8x8x64, 8, 8, true)
QGEMM_KERNEL(qgemm_8x16x64, 8, 16, true)
QGEMM_KERNEL(qgemm_w8_8x8x64, 8, 8, false)
QGEMM_KERNEL(qgemm_w8_8x16x64, 8, 16, false)

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

// Diagnostic high-precision tiled attention. Q/K/V and the final branch boundary
// remain fp16, but scores, softmax probabilities, and both dot-product
// accumulations stay fp32. This deliberately favors numerical evidence over speed.
kernel void attention_qk_f16_to_f32(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    constant uint &headDim [[buffer(5)]],
    constant float &scale [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= columns || gid.y >= rows) return;
    float sum = 0.0f;
    for (uint d = 0; d < headDim; ++d) {
        sum = fma(float(query[gid.y * headDim + d]),
                  float(key[gid.x * headDim + d]), sum);
    }
    scores[gid.y * columns + gid.x] = sum * scale;
}

kernel void attention_softmax_rows_f32(
    device float *scores [[buffer(0)]],
    constant uint &rows [[buffer(1)]],
    constant uint &columns [[buffer(2)]],
    constant uint &queryBase [[buffer(3)]],
    constant uint &causal [[buffer(4)]],
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
        if (causal == 0 || column <= queryBase + row)
            localMax = max(localMax, scores[row * columns + column]);
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
        if (causal == 0 || column <= queryBase + row)
            localSum += exp(scores[row * columns + column] - rowMax);
    }
    partial[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverseSum = 1.0f / partial[0];
    for (uint column = tid; column < columns; column += threadCount) {
        scores[row * columns + column] =
            (causal == 0 || column <= queryBase + row)
            ? exp(scores[row * columns + column] - rowMax) * inverseSum : 0.0f;
    }
}

kernel void attention_pv_f32_f16_to_f16(
    device const float *probabilities [[buffer(0)]],
    device const half *value [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &keyCount [[buffer(4)]],
    constant uint &headDim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= headDim || gid.y >= rows) return;
    float sum = 0.0f;
    for (uint keyRow = 0; keyRow < keyCount; ++keyRow) {
        sum = fma(probabilities[gid.y * keyCount + keyRow],
                  float(value[keyRow * headDim + gid.x]), sum);
    }
    output[gid.y * headDim + gid.x] = half(sum);
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

// ---------------------------------------------------------------------------
// Numerical-health probes (optimization phase, D201).
//
// Stats ABI shared by every probe kernel below. `stats` is an array of
// uint32 quads, one per probe slot:
//   [slot*4+0] flags   (bit meanings per kernel, see below)
//   [slot*4+1] maxAbs  float bit pattern of max |value| observed (0 if none)
//   [slot*4+2] firstIndex  first offending element index (0xFFFFFFFFu if none)
//   [slot*4+3] reserved
// Writes are relaxed atomics from thread 0 of each threadgroup, so multiple
// concurrent kernels/tiles can safely accumulate into one slot.
//
// Common flag bits (float_to_half / standalone probes):
//   bit0 NaN observed
//   bit1 +Inf observed
//   bit2 -Inf observed
//   bit3 |value| >= 65504 (would overflow/round at an fp16 storage boundary)
// gate_add probe adds:
//   bit4 result (residual) became NaN
//   bit5 result (residual) became +/-Inf
// (The NUM_FLAG_* constants themselves are declared near the P3-A kernels,
// before their first use.)

// fp32 -> fp16 conversion that also records input-health stats. Replaces
// float_to_half at probed call sites; identical conversion semantics.
kernel void float_to_half_probe(
    device const float *source    [[buffer(0)]],
    device half       *destination [[buffer(1)]],
    constant uint     &count      [[buffer(2)]],
    device uint       *stats      [[buffer(3)]],
    constant uint     &probeSlot  [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint partialFlags[256];
    threadgroup uint partialMax[256];
    threadgroup uint partialIndex[256];
    uint base = group.x * 256;
    uint i = base + tid;
    uint localFlags = 0;
    uint localMax = 0;
    uint localIndex = 0xFFFFFFFFu;
    if (i < count) {
        float v = source[i];
        destination[i] = half(v);
        if (isnan(v)) {
            localFlags |= NUM_FLAG_NAN;
            localIndex = min(localIndex, i);
        } else if (isinf(v)) {
            localFlags |= (v > 0.0f) ? NUM_FLAG_POS_INF : NUM_FLAG_NEG_INF;
            localIndex = min(localIndex, i);
        } else {
            float a = fabs(v);
            if (a > 65504.0f) {
                localFlags |= NUM_FLAG_HALF_OVERFLOW;
                localIndex = min(localIndex, i);
            }
            uint bits = as_type<uint>(a);
            if (bits > localMax) localMax = bits;
        }
    }
    partialFlags[tid] = localFlags;
    partialMax[tid] = localMax;
    partialIndex[tid] = localIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialFlags[tid] |= partialFlags[tid + stride];
            partialMax[tid] = max(partialMax[tid], partialMax[tid + stride]);
            partialIndex[tid] = min(partialIndex[tid], partialIndex[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && partialFlags[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        uint slotBase = probeSlot * NUM_SLOT_UINTS;
        atomic_fetch_or_explicit(&atom[slotBase + 0], partialFlags[0], memory_order_relaxed);
        atomic_fetch_min_explicit(&atom[slotBase + 2], partialIndex[0], memory_order_relaxed);
    }
    if (tid == 0 && partialMax[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        atomic_fetch_max_explicit(&atom[probeSlot * NUM_SLOT_UINTS + 1], partialMax[0], memory_order_relaxed);
    }
}

// Gated fp16-branch add into fp32 residual that also records branch/residual
// health. Replaces gate_add_half_f32 at probed call sites.
kernel void gate_add_half_f32_probe(
    device float       *residual [[buffer(0)]],
    device const half  *branch   [[buffer(1)]],
    device const float *gate     [[buffer(2)]],
    constant uint      &N        [[buffer(3)]],
    constant uint      &count    [[buffer(4)]],
    device uint        *stats    [[buffer(5)]],
    constant uint      &probeSlot [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint partialFlags[256];
    threadgroup uint partialMax[256];
    threadgroup uint partialIndex[256];
    uint base = group.x * 256;
    uint i = base + tid;
    uint localFlags = 0;
    uint localMax = 0;
    uint localIndex = 0xFFFFFFFFu;
    if (i < count) {
        float b = float(branch[i]);
        float g = gate[i % N];
        float result = residual[i] + b * g;
        residual[i] = result;
        if (isnan(b)) {
            localFlags |= NUM_FLAG_NAN;
            localIndex = min(localIndex, i);
        } else if (isinf(b)) {
            localFlags |= (b > 0.0f) ? NUM_FLAG_POS_INF : NUM_FLAG_NEG_INF;
            localIndex = min(localIndex, i);
        } else {
            float a = fabs(b);
            if (a > 65504.0f) {
                localFlags |= NUM_FLAG_HALF_OVERFLOW;
                localIndex = min(localIndex, i);
            }
            uint bits = as_type<uint>(a);
            if (bits > localMax) localMax = bits;
        }
        if (isnan(result)) {
            localFlags |= NUM_FLAG_RESULT_NAN;
            localIndex = min(localIndex, i);
        } else if (isinf(result)) {
            localFlags |= NUM_FLAG_RESULT_INF;
            localIndex = min(localIndex, i);
        }
    }
    partialFlags[tid] = localFlags;
    partialMax[tid] = localMax;
    partialIndex[tid] = localIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialFlags[tid] |= partialFlags[tid + stride];
            partialMax[tid] = max(partialMax[tid], partialMax[tid + stride]);
            partialIndex[tid] = min(partialIndex[tid], partialIndex[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && partialFlags[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        uint slotBase = probeSlot * NUM_SLOT_UINTS;
        atomic_fetch_or_explicit(&atom[slotBase + 0], partialFlags[0], memory_order_relaxed);
        atomic_fetch_min_explicit(&atom[slotBase + 2], partialIndex[0], memory_order_relaxed);
    }
    if (tid == 0 && partialMax[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        atomic_fetch_max_explicit(&atom[probeSlot * NUM_SLOT_UINTS + 1], partialMax[0], memory_order_relaxed);
    }
}

// Row softmax over fp16 scores that also records input-score health. Input
// semantics identical to attention_softmax_rows; stats record the raw scores
// as read during the max pass.
kernel void attention_softmax_rows_probe(
    device half       *scores    [[buffer(0)]],
    constant uint     &rows      [[buffer(1)]],
    constant uint     &columns   [[buffer(2)]],
    constant uint     &queryBase [[buffer(3)]],
    constant uint     &causal    [[buffer(4)]],
    device uint       *stats     [[buffer(5)]],
    constant uint     &probeSlot [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    uint threadCount = threads.x;
    threadgroup float partial[256];
    threadgroup uint localFlags[256];
    threadgroup uint localMax[256];
    threadgroup uint localIndex[256];
    float localMaxScore = -INFINITY;
    uint flags = 0;
    uint maxBits = 0;
    uint firstIndex = 0xFFFFFFFFu;
    for (uint column = tid; column < columns; column += threadCount) {
        if (causal == 0 || column <= queryBase + row) {
            float s = float(scores[row * columns + column]);
            localMaxScore = max(localMaxScore, s);
            if (isnan(s)) {
                flags |= NUM_FLAG_NAN;
                firstIndex = min(firstIndex, column);
            } else if (isinf(s)) {
                flags |= (s > 0.0f) ? NUM_FLAG_POS_INF : NUM_FLAG_NEG_INF;
                firstIndex = min(firstIndex, column);
            } else {
                float a = fabs(s);
                if (a > 65504.0f) {
                    flags |= NUM_FLAG_HALF_OVERFLOW;
                    firstIndex = min(firstIndex, column);
                }
                uint bits = as_type<uint>(a);
                if (bits > maxBits) maxBits = bits;
            }
        }
    }
    partial[tid] = localMaxScore;
    localFlags[tid] = flags;
    localMax[tid] = maxBits;
    localIndex[tid] = firstIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] = max(partial[tid], partial[tid + stride]);
            localFlags[tid] |= localFlags[tid + stride];
            localMax[tid] = max(localMax[tid], localMax[tid + stride]);
            localIndex[tid] = min(localIndex[tid], localIndex[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rowMax = partial[0];
    if (tid == 0 && localFlags[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        uint slotBase = probeSlot * NUM_SLOT_UINTS;
        atomic_fetch_or_explicit(&atom[slotBase + 0], localFlags[0], memory_order_relaxed);
        atomic_fetch_min_explicit(&atom[slotBase + 2], localIndex[0], memory_order_relaxed);
    }
    if (tid == 0 && localMax[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        atomic_fetch_max_explicit(&atom[probeSlot * NUM_SLOT_UINTS + 1], localMax[0], memory_order_relaxed);
    }
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

// Row softmax over fp32 scores (fp32-scores attention mode) with stats.
kernel void attention_softmax_rows_f32_probe(
    device float *scores    [[buffer(0)]],
    constant uint &rows     [[buffer(1)]],
    constant uint &columns  [[buffer(2)]],
    constant uint &queryBase [[buffer(3)]],
    constant uint &causal   [[buffer(4)]],
    device uint   *stats    [[buffer(5)]],
    constant uint &probeSlot [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    uint threadCount = threads.x;
    threadgroup float partial[256];
    threadgroup uint localFlags[256];
    threadgroup uint localMax[256];
    threadgroup uint localIndex[256];
    float localMaxScore = -INFINITY;
    uint flags = 0;
    uint maxBits = 0;
    uint firstIndex = 0xFFFFFFFFu;
    for (uint column = tid; column < columns; column += threadCount) {
        if (causal == 0 || column <= queryBase + row) {
            float s = scores[row * columns + column];
            localMaxScore = max(localMaxScore, s);
            if (isnan(s)) {
                flags |= NUM_FLAG_NAN;
                firstIndex = min(firstIndex, column);
            } else if (isinf(s)) {
                flags |= (s > 0.0f) ? NUM_FLAG_POS_INF : NUM_FLAG_NEG_INF;
                firstIndex = min(firstIndex, column);
            } else {
                float a = fabs(s);
                if (a > 65504.0f) {
                    flags |= NUM_FLAG_HALF_OVERFLOW;
                    firstIndex = min(firstIndex, column);
                }
                uint bits = as_type<uint>(a);
                if (bits > maxBits) maxBits = bits;
            }
        }
    }
    partial[tid] = localMaxScore;
    localFlags[tid] = flags;
    localMax[tid] = maxBits;
    localIndex[tid] = firstIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] = max(partial[tid], partial[tid + stride]);
            localFlags[tid] |= localFlags[tid + stride];
            localMax[tid] = max(localMax[tid], localMax[tid + stride]);
            localIndex[tid] = min(localIndex[tid], localIndex[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rowMax = partial[0];
    if (tid == 0 && localFlags[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        uint slotBase = probeSlot * NUM_SLOT_UINTS;
        atomic_fetch_or_explicit(&atom[slotBase + 0], localFlags[0], memory_order_relaxed);
        atomic_fetch_min_explicit(&atom[slotBase + 2], localIndex[0], memory_order_relaxed);
    }
    if (tid == 0 && localMax[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        atomic_fetch_max_explicit(&atom[probeSlot * NUM_SLOT_UINTS + 1], localMax[0], memory_order_relaxed);
    }
    float localSum = 0.0f;
    for (uint column = tid; column < columns; column += threadCount) {
        if (causal == 0 || column <= queryBase + row)
            localSum += exp(scores[row * columns + column] - rowMax);
    }
    partial[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverseSum = 1.0f / partial[0];
    for (uint column = tid; column < columns; column += threadCount) {
        scores[row * columns + column] =
            (causal == 0 || column <= queryBase + row)
            ? exp(scores[row * columns + column] - rowMax) * inverseSum : 0.0f;
    }
}

// Standalone fp16 stats pass over an arbitrary fp16 buffer (MPS outputs:
// q/k/v tokens, attended, branch, projected). Flag bits as the common set.
kernel void probe_f16_stats(
    device const half *values [[buffer(0)]],
    device uint       *stats  [[buffer(1)]],
    constant uint     &count  [[buffer(2)]],
    constant uint     &probeSlot [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint partialFlags[256];
    threadgroup uint partialMax[256];
    threadgroup uint partialIndex[256];
    uint base = group.x * 256;
    uint i = base + tid;
    uint localFlags = 0;
    uint localMax = 0;
    uint localIndex = 0xFFFFFFFFu;
    if (i < count) {
        float v = float(values[i]);
        if (isnan(v)) {
            localFlags |= NUM_FLAG_NAN;
            localIndex = min(localIndex, i);
        } else if (isinf(v)) {
            localFlags |= (v > 0.0f) ? NUM_FLAG_POS_INF : NUM_FLAG_NEG_INF;
            localIndex = min(localIndex, i);
        } else {
            float a = fabs(v);
            if (a > 65504.0f) {
                localFlags |= NUM_FLAG_HALF_OVERFLOW;
                localIndex = min(localIndex, i);
            }
            uint bits = as_type<uint>(a);
            if (bits > localMax) localMax = bits;
        }
    }
    partialFlags[tid] = localFlags;
    partialMax[tid] = localMax;
    partialIndex[tid] = localIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialFlags[tid] |= partialFlags[tid + stride];
            partialMax[tid] = max(partialMax[tid], partialMax[tid + stride]);
            partialIndex[tid] = min(partialIndex[tid], partialIndex[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && partialFlags[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        uint slotBase = probeSlot * NUM_SLOT_UINTS;
        atomic_fetch_or_explicit(&atom[slotBase + 0], partialFlags[0], memory_order_relaxed);
        atomic_fetch_min_explicit(&atom[slotBase + 2], partialIndex[0], memory_order_relaxed);
    }
    if (tid == 0 && partialMax[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        atomic_fetch_max_explicit(&atom[probeSlot * NUM_SLOT_UINTS + 1], partialMax[0], memory_order_relaxed);
    }
}

// Standalone fp32 stats pass (velocity, denoised, Euler output, fp32 residual
// after each branch). Flag bit3 means "|v| >= 65504 (would overflow fp16)".
kernel void probe_f32_stats(
    device const float *values [[buffer(0)]],
    device uint        *stats  [[buffer(1)]],
    constant uint      &count  [[buffer(2)]],
    constant uint      &probeSlot [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup uint partialFlags[256];
    threadgroup uint partialMax[256];
    threadgroup uint partialIndex[256];
    uint base = group.x * 256;
    uint i = base + tid;
    uint localFlags = 0;
    uint localMax = 0;
    uint localIndex = 0xFFFFFFFFu;
    if (i < count) {
        float v = values[i];
        if (isnan(v)) {
            localFlags |= NUM_FLAG_NAN;
            localIndex = min(localIndex, i);
        } else if (isinf(v)) {
            localFlags |= (v > 0.0f) ? NUM_FLAG_POS_INF : NUM_FLAG_NEG_INF;
            localIndex = min(localIndex, i);
        } else {
            float a = fabs(v);
            if (a > 65504.0f) {
                localFlags |= NUM_FLAG_HALF_OVERFLOW;
                localIndex = min(localIndex, i);
            }
            uint bits = as_type<uint>(a);
            if (bits > localMax) localMax = bits;
        }
    }
    partialFlags[tid] = localFlags;
    partialMax[tid] = localMax;
    partialIndex[tid] = localIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialFlags[tid] |= partialFlags[tid + stride];
            partialMax[tid] = max(partialMax[tid], partialMax[tid + stride]);
            partialIndex[tid] = min(partialIndex[tid], partialIndex[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && partialFlags[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        uint slotBase = probeSlot * NUM_SLOT_UINTS;
        atomic_fetch_or_explicit(&atom[slotBase + 0], partialFlags[0], memory_order_relaxed);
        atomic_fetch_min_explicit(&atom[slotBase + 2], partialIndex[0], memory_order_relaxed);
    }
    if (tid == 0 && partialMax[0] != 0) {
        device atomic_uint *atom = (device atomic_uint *)stats;
        atomic_fetch_max_explicit(&atom[probeSlot * NUM_SLOT_UINTS + 1], partialMax[0], memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// P7-A: streaming/online-softmax attention kernels.
//
// The strided token-major MPS path (P4) materializes a full fp16
// [queryTile, keyCount] score tile. P7-A keeps MPS for QK and PV but
// processes keys in CHUNKS so only a [queryTile, keyChunk] score tile is
// ever live, and carries the online-softmax state between chunks. All chunks
// of a query tile encode into the SAME block command buffer: chunk (k+1)
// reads state written by chunk k purely via GPU ordering (each chunk is a
// separate encoder; encoder boundaries act as barriers), so there is NO
// per-chunk wait and NO extra command-buffer completion.
//
// State buffer layout (per query row; chunkCount rows, e.g. 128; all FP32):
//   runningMax[row]     — running max of scaled scores
//   runningSum[row]     — running sum of exp(score - runningMax)
//   runningAlpha[row]   — rescale factor exp(oldMax - newMax) of the LAST
//                         prepare pass (1.0 on the first chunk)
//   accumulator[row * headDim + d] — running output (alpha-rescaled)
// The PV result of every chunk is accumulated into the FP32 accumulator —
// NEVER into an fp16 accumulator — and only the final chunk divides by the
// running sum and rounds to half.
// ---------------------------------------------------------------------------

// P7-A: given the fp16 scores of ONE key chunk (row-major [rows, rowStride]
// halves; rowStride = chunkColumns for the tight MPS score tile), update the
// running online-softmax state and rewrite the scores tile in place to the
// RESCALED chunk probabilities (exp(score - newMax) * alpha). One threadgroup
// per query row; threads reduce over the chunk columns.
kernel void streaming_softmax_prepare(
    device half       *scores       [[buffer(0)]],   // in/out [rows, rowStride]
    device float      *runningMax   [[buffer(1)]],   // in/out [rows]
    device float      *runningSum   [[buffer(2)]],   // in/out [rows]
    device float      *runningAlpha [[buffer(3)]],   // out [rows]
    constant uint     &rows         [[buffer(4)]],
    constant uint     &columns      [[buffer(5)]],   // chunkColumns (Bk)
    constant uint     &rowStride    [[buffer(6)]],   // padded row stride (halves)
    constant uint     &firstChunk   [[buffer(7)]],   // 1 on the first chunk
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    uint row = group.x;
    if (row >= rows) return;
    uint threadCount = threads.x;
    threadgroup float partial[256];

    // Chunk-local max (fp32).
    float localMax = -INFINITY;
    for (uint column = tid; column < columns; column += threadCount) {
        localMax = max(localMax, float(scores[row * rowStride + column]));
    }
    partial[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = max(partial[tid], partial[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float chunkMax = partial[0];

    // Merge with the running max (online-softmax rescale).
    float oldMax = runningMax[row];
    float newMax = firstChunk != 0 ? chunkMax : max(oldMax, chunkMax);
    float alpha = firstChunk != 0 ? 1.0f : exp(oldMax - newMax);
    runningMax[row] = newMax;
    if (tid == 0) runningAlpha[row] = alpha;

    // Rewrite this chunk's scores to exp(score - newMax) * alpha and add the
    // chunk's row sum (fp32) into the running sum.
    float localSum = 0.0f;
    for (uint column = tid; column < columns; column += threadCount) {
        float p = exp(float(scores[row * rowStride + column]) - newMax) * alpha;
        scores[row * rowStride + column] = half(p);
        localSum += p;
    }
    partial[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threadCount >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) runningSum[row] = firstChunk != 0 ? partial[0] : runningSum[row] + partial[0];
}

// P7-A: rescale the running FP32 output accumulator by this chunk's alpha
// (stored by the prepare pass) and accumulate the fp16 chunk PV result:
// acc = alpha * acc + float(chunkOut). The PV result is row-major
// [rows, headDim] with rowBytes = headDim * 2 (tight). Threads iterate over
// the [rows, headDim] tile.
kernel void streaming_chunk_accumulate(
    device const half *chunkOut     [[buffer(0)]],   // MPS PV result [rows, headDim]
    device float      *accumulator  [[buffer(1)]],   // in/out [rows, headDim]
    device const float *runningAlpha [[buffer(2)]],  // [rows] alpha of this chunk
    constant uint     &rows         [[buffer(3)]],
    constant uint     &headDim      [[buffer(4)]],
    constant uint     &firstChunk   [[buffer(5)]],   // 1 on the first chunk
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= headDim || gid.y >= rows) return;
    uint index = gid.y * headDim + gid.x;
    accumulator[index] = (firstChunk != 0)
        ? float(chunkOut[index])
        : fma(runningAlpha[gid.y], accumulator[index], float(chunkOut[index]));
}

// P7-A: finalize one query tile of streaming attention:
// output = half(acc / runningSum) into a TOKEN-MAJOR [rows, tokenStride]
// buffer (each head owns headDim contiguous columns; column base head*headDim
// is included in the buffer offset chosen by the caller). One thread per
// [row, headDim] element.
kernel void streaming_softmax_finalize(
    device const float *accumulator  [[buffer(0)]],
    device const float *runningSum   [[buffer(1)]],
    device half        *output       [[buffer(2)]],   // token-major [rows, tokenStride]
    constant uint      &rows         [[buffer(3)]],
    constant uint      &headDim      [[buffer(4)]],
    constant uint      &tokenStride  [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= headDim || gid.y >= rows) return;
    float sum = runningSum[gid.y];
    float value = sum > 0.0f ? accumulator[gid.y * headDim + gid.x] / sum : 0.0f;
    output[gid.y * tokenStride + gid.x] = half(value);
}

// ---------------------------------------------------------------------------
// P7-B: DiT-specialized pure-Metal Flash-style online attention.
//
// STRICTLY DiT-specialized (NOT a generic transformer library):
//   - headDim == 128 only, heads == 16, non-causal
//   - token-major Q/K/V `[rows, tokenStride]` half buffers (tokenStride 2048)
//   - token-major output, FP32 score/softmax/output accumulation
// Any other shape must be rejected by the Swift side BEFORE this kernel runs.
//
// A12-safe: NO simdgroup_matrix, NO Metal 3. Score dots use `simd_sum` over
// the 32 lanes of a SIMD group (threadExecutionWidth == 32 is required); each
// lane owns 4 of the 128 headDim dims. Online-softmax running max/sum and the
// output accumulator are FP32; running-max/rescale is MANDATORY (never a raw
// exp(score), never an fp16 denominator/accumulator).
//
// Profile h128_q4_k32: 4 query rows per threadgroup, 4 SIMD groups (128
// threads), K/V tile 32 keys. Each SIMD group computes ONE query row; every
// lane of the group maintains IDENTICAL running max/sum scalars (simd_max
// broadcasts, and the per-lane chunk sums are identical because every lane
// computes the full score via simd_sum), so no lane shuffles are needed for
// the final normalization. Lane d of the SIMD group owns output dim
// (d * 32 + lane), matching the dot layout (dim d*32+lane), so the PV
// accumulation needs no lane data movement either.
// ---------------------------------------------------------------------------

// P7-B: DiT Flash attention, headDim=128, 4 query rows/threadgroup, K=32.
// Output is token-major [queryCount, tokenStride]; each head owns headDim
// contiguous columns of every row (column base = head * 128).
kernel void dit_flash_attention_h128_q4_k32(
    device const half *query    [[buffer(0)]],  // [queryCount, tokenStride]
    device const half *key      [[buffer(1)]],  // [keyCount, tokenStride]
    device const half *value    [[buffer(2)]],  // [keyCount, tokenStride]
    device half       *output   [[buffer(3)]],  // [queryCount, tokenStride]
    constant uint     &queryCount [[buffer(4)]],
    constant uint     &keyCount   [[buffer(5)]],
    constant uint     &tokenStride [[buffer(6)]],
    constant float    &scale      [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]])
{
    constexpr uint HEAD_DIM = 128;
    constexpr uint K_TILE = 32;
    constexpr uint SIMD_GROUPS = 4;      // 128 threads / 32 lanes
    constexpr uint THREADS = 128;

    uint lane = simdLane;                 // 0..31
    // `simd` is 0..3 (simdgroup_index_in_threadgroup) → query row within group.
    uint head = group.z;                  // head index (grid.z = heads)
    uint queryRow = group.x * SIMD_GROUPS + simd;
    if (queryRow >= queryCount) return;

    // Threadgroup K/V tile: [K_TILE][HEAD_DIM] halves, 8 KiB each (16 KiB total).
    threadgroup half kTile[K_TILE * HEAD_DIM];
    threadgroup half vTile[K_TILE * HEAD_DIM];

    // `tid` is 0..127 (thread_index_in_threadgroup).
    uint headColBase = head * HEAD_DIM;   // column base of this head in the token row

    // Running online-softmax state (FP32, IDENTICAL on every lane).
    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    // FP32 output accumulator: lane d of the SIMD group owns output dim
    // (d * 32 + lane), 4 dims per lane.
    float acc[HEAD_DIM / 32];             // 4 floats per lane

    #pragma unroll
    for (uint d = 0; d < HEAD_DIM / 32; ++d) {
        acc[d] = 0.0f;
    }

    uint keyBase = 0;
    while (keyBase < keyCount) {
        // Cooperative load of the K/V tile [K_TILE][HEAD_DIM] into threadgroup
        // memory (16 KiB total for the K=32 profile).
        #pragma unroll
        for (uint i = 0; i < (K_TILE * HEAD_DIM) / THREADS; ++i) {
            uint index = tid + i * THREADS;
            uint kRow = keyBase + index / HEAD_DIM;
            uint kCol = index % HEAD_DIM;
            if (kRow < keyCount) {
                kTile[index] = key[kRow * tokenStride + headColBase + kCol];
                vTile[index] = value[kRow * tokenStride + headColBase + kCol];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint kCount = min(K_TILE, keyCount - keyBase);

        // Pass 1: chunk-local max of the scaled scores for this query row.
        // Each lane computes a 4-dim partial dot; simd_max reduces (and
        // broadcasts) the chunk max over the 32 lanes.
        float localMax = -INFINITY;
                for (uint k = 0; k < K_TILE; ++k) {
            if (k >= kCount) break;
            float dot = 0.0f;
            #pragma unroll
            for (uint d = 0; d < HEAD_DIM / 32; ++d) {
                dot = fma(float(kTile[k * HEAD_DIM + d * 32 + lane]),
                          float(query[queryRow * tokenStride + headColBase + d * 32 + lane]), dot);
            }
            localMax = max(localMax, dot * scale);
        }
        float chunkMax = simd_max(localMax);   // identical on all lanes

        // Online-softmax rescale (running max/sum in FP32; MANDATORY).
        float newMax = (keyBase == 0) ? chunkMax : max(rowMax, chunkMax);
        float alpha = (keyBase == 0) ? 1.0f : exp(rowMax - newMax);
        rowMax = newMax;

        // Rescale the OLD accumulator BEFORE adding this chunk's contribution.
        #pragma unroll
        for (uint d = 0; d < HEAD_DIM / 32; ++d) {
            acc[d] = acc[d] * alpha;
        }

        // Pass 2: chunk probabilities + PV accumulation. The full score is
        // recomputed per lane as a partial dot, then `simd_sum` broadcasts the
        // FULL dot to every lane, so p = exp(score*scale - newMax) is
        // identical on all lanes and no score shuffle is needed.
        float chunkSum = 0.0f;
                for (uint k = 0; k < K_TILE; ++k) {
            if (k >= kCount) break;
            float dot = 0.0f;
            #pragma unroll
            for (uint d = 0; d < HEAD_DIM / 32; ++d) {
                dot = fma(float(kTile[k * HEAD_DIM + d * 32 + lane]),
                          float(query[queryRow * tokenStride + headColBase + d * 32 + lane]), dot);
            }
            float p = exp(simd_sum(dot) * scale - newMax);
            chunkSum += p;
            #pragma unroll
            for (uint d = 0; d < HEAD_DIM / 32; ++d) {
                acc[d] = fma(p, float(vTile[k * HEAD_DIM + d * 32 + lane]), acc[d]);
            }
        }
        // chunkSum is identical on every lane (identical p sequence), so every
        // lane updates rowSum identically — no broadcast required.
        rowSum = rowSum * alpha + chunkSum;

        // The next chunk's cooperative load must not overwrite the tile while
        // any lane is still reading it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        keyBase += K_TILE;
    }

    // Finalize: output = half(acc / rowSum), token-major write.
    float inverseSum = 1.0f / rowSum;
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM / 32; ++d) {
        output[queryRow * tokenStride + headColBase + d * 32 + lane] = half(acc[d] * inverseSum);
    }
}

// P7-B: K=16 profile of the same DiT Flash attention. Same threadgroup shape
// (4 query rows, 128 threads) with a smaller K/V tile (4 KiB K + 4 KiB V) for
// devices/pipelines whose threadgroup memory or occupancy prefers it. The math
// is byte-identical to the K=32 profile — only the key chunk size changes.
kernel void dit_flash_attention_h128_q4_k16(
    device const half *query    [[buffer(0)]],
    device const half *key      [[buffer(1)]],
    device const half *value    [[buffer(2)]],
    device half       *output   [[buffer(3)]],
    constant uint     &queryCount [[buffer(4)]],
    constant uint     &keyCount   [[buffer(5)]],
    constant uint     &tokenStride [[buffer(6)]],
    constant float    &scale      [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simdLane [[thread_index_in_simdgroup]])
{
    constexpr uint HEAD_DIM = 128;
    constexpr uint K_TILE = 16;
    constexpr uint SIMD_GROUPS = 4;
    constexpr uint THREADS = 128;

    uint lane = simdLane;
    uint head = group.z;                  // head index (grid.z = heads)
    uint queryRow = group.x * SIMD_GROUPS + simd;
    if (queryRow >= queryCount) return;

    threadgroup half kTile[K_TILE * HEAD_DIM];
    threadgroup half vTile[K_TILE * HEAD_DIM];

    uint headColBase = head * HEAD_DIM;

    float rowMax = -INFINITY;
    float rowSum = 0.0f;
    float acc[HEAD_DIM / 32];

    #pragma unroll
    for (uint d = 0; d < HEAD_DIM / 32; ++d) {
        acc[d] = 0.0f;
    }

    uint keyBase = 0;
    while (keyBase < keyCount) {
        #pragma unroll
        for (uint i = 0; i < (K_TILE * HEAD_DIM) / THREADS; ++i) {
            uint index = tid + i * THREADS;
            uint kRow = keyBase + index / HEAD_DIM;
            uint kCol = index % HEAD_DIM;
            if (kRow < keyCount) {
                kTile[index] = key[kRow * tokenStride + headColBase + kCol];
                vTile[index] = value[kRow * tokenStride + headColBase + kCol];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint kCount = min(K_TILE, keyCount - keyBase);

        float localMax = -INFINITY;
                for (uint k = 0; k < K_TILE; ++k) {
            if (k >= kCount) break;
            float dot = 0.0f;
            #pragma unroll
            for (uint d = 0; d < HEAD_DIM / 32; ++d) {
                dot = fma(float(kTile[k * HEAD_DIM + d * 32 + lane]),
                          float(query[queryRow * tokenStride + headColBase + d * 32 + lane]), dot);
            }
            localMax = max(localMax, dot * scale);
        }
        float chunkMax = simd_max(localMax);

        float newMax = (keyBase == 0) ? chunkMax : max(rowMax, chunkMax);
        float alpha = (keyBase == 0) ? 1.0f : exp(rowMax - newMax);
        rowMax = newMax;

        #pragma unroll
        for (uint d = 0; d < HEAD_DIM / 32; ++d) {
            acc[d] = acc[d] * alpha;
        }

        float chunkSum = 0.0f;
                for (uint k = 0; k < K_TILE; ++k) {
            if (k >= kCount) break;
            float dot = 0.0f;
            #pragma unroll
            for (uint d = 0; d < HEAD_DIM / 32; ++d) {
                dot = fma(float(kTile[k * HEAD_DIM + d * 32 + lane]),
                          float(query[queryRow * tokenStride + headColBase + d * 32 + lane]), dot);
            }
            float p = exp(simd_sum(dot) * scale - newMax);
            chunkSum += p;
            #pragma unroll
            for (uint d = 0; d < HEAD_DIM / 32; ++d) {
                acc[d] = fma(p, float(vTile[k * HEAD_DIM + d * 32 + lane]), acc[d]);
            }
        }
        rowSum = rowSum * alpha + chunkSum;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        keyBase += K_TILE;
    }

    float inverseSum = 1.0f / rowSum;
    #pragma unroll
    for (uint d = 0; d < HEAD_DIM / 32; ++d) {
        output[queryRow * tokenStride + headColBase + d * 32 + lane] = half(acc[d] * inverseSum);
    }
}

// ---------------------------------------------------------------------------
// A12/H11 ANE shared-IOSurface layout bridges.
//
// AnimaXS activations are tight token-major [rows, channels]. The proven H11
// Espresso ABI is channel-major, with every channel plane padded to a 64-byte
// stride. Both buffers are fp16. These kernels are the only layout copies in
// the ANE hybrid path; the underlying ANE surfaces are simultaneously shared
// MTLBuffers (no CPU staging/readback).
// ---------------------------------------------------------------------------
kernel void dit_token_to_ane_f16(
    device const half *tokenMajor [[buffer(0)]],
    device half *aneMajor [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    constant uint &planeStrideElements [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint row = gid.y;
    if (channel >= channels || row >= rows) return;
    aneMajor[channel * planeStrideElements + row] = tokenMajor[row * channels + channel];
}

kernel void dit_ane_to_token_f16(
    device const half *aneMajor [[buffer(0)]],
    device half *tokenMajor [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    constant uint &planeStrideElements [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint row = gid.y;
    if (channel >= channels || row >= rows) return;
    tokenMajor[row * channels + channel] = aneMajor[channel * planeStrideElements + row];
}
