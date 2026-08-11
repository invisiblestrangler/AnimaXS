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
    uint2 gid [[thread_position_in_grid]])          // gid.x = k, gid.y = row
{
    uint k = gid.x;
    uint r = gid.y;
    if (k >= K || r >= rows) return;
    uint byteIdx = r * rowStride + (k >> 1);
    uchar b = packed[byteIdx];
    uint q = (k & 1) == 0 ? uint(b & 0x0F) : uint(b >> 4);
    uint g = k / W4_GROUP;
    half sc = scale[r * (K + W4_GROUP - 1) / W4_GROUP + g];
    half ze = zero[r * (K + W4_GROUP - 1) / W4_GROUP + g];
    out[r * K + k] = half(float(q) * float(sc) + float(ze));
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
    uint2 gid [[thread_position_in_grid]])
{
    uint k = gid.x;
    uint r = gid.y;
    if (k >= K || r >= rows) return;
    uint q = uint(packed[r * rowStride + k]);
    uint g = k / W4_GROUP;
    half sc = scale[r * (K + W4_GROUP - 1) / W4_GROUP + g];
    half ze = zero[r * (K + W4_GROUP - 1) / W4_GROUP + g];
    out[r * K + k] = half(float(q) * float(sc) + float(ze));
}

// ---------------------------------------------------------------------------
// RMSNorm (fp32 input → fp32 output; stats in fp32). eps is buffer constant.
// vector length N per row; rows independent.
// ---------------------------------------------------------------------------
kernel void rmsnorm_f32_to_f32(
    device const float *in      [[buffer(0)]],
    device float       *out     [[buffer(1)]],
    device const float *weight  [[buffer(2)]],   // optional per-element scale; null → identity
    constant uint      &N       [[buffer(3)]],
    constant float     &eps     [[buffer(4)]],
    constant uint      &useWt   [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint n = gid.x;
    uint r = gid.y;
    // sum of squares
    float sum = 0.0f;
    for (uint i = 0; i < N; ++i) {
        float v = in[r * N + i];
        sum += v * v;
    }
    float mean = sum / float(N);
    float inv = rsqrt(mean + eps);
    float v = in[r * N + n];
    float w = useWt != 0 ? weight[n] : 1.0f;
    out[r * N + n] = v * inv * w;
}

// ---------------------------------------------------------------------------
// SCAFFOLD ONLY: this tanh approximation is NOT the adapter/DiT reference GELU.
// E003 must replace it with exact erf GELU and add hosted Metal parity tests.
// ---------------------------------------------------------------------------
kernel void gelu(
    device const half *in  [[buffer(0)]],
    device half       *out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float x = float(in[gid]);
    // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    float t = tanh(0.7978845608f * (x + 0.044715f * x * x * x));
    out[gid] = half(0.5f * x * (1.0f + t));
}

// ---------------------------------------------------------------------------
// SiLU / Swish.
// ---------------------------------------------------------------------------
kernel void silu(
    device const half *in  [[buffer(0)]],
    device half       *out [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float x = float(in[gid]);
    out[gid] = half(x / (1.0f + exp(-x)));
}

// ---------------------------------------------------------------------------
// SCAFFOLD ONLY: H005 requires a fp32 gate and fp32 residual update.
// E003 must replace this half-gate signature and add hosted Metal parity tests.
// ---------------------------------------------------------------------------
kernel void gate_add_half_into_float(
    device float       *residual [[buffer(0)]],
    device const half  *branch   [[buffer(1)]],
    device const half  *gate     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    residual[gid] += float(branch[gid]) * float(gate[gid]);
}

// ---------------------------------------------------------------------------
// Add half branch into fp32 residual (gate == 1).
// ---------------------------------------------------------------------------
kernel void add_half_into_float(
    device float       *residual [[buffer(0)]],
    device const half  *branch   [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    residual[gid] += float(branch[gid]);
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
    uint gid [[thread_position_in_grid]])
{
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
