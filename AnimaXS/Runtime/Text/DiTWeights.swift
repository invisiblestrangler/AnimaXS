import Foundation

/// Shared W4/fp16 weight loader for the DiT pack (`anima-turbo-v1.0-xsmax-w4.animapk`).
///
/// DiT storage rules (DECISIONS D011/D017, runbook §18):
///   - projection/embedding matrices are W4 (group 64, even K → low nibble)
///   - norm vectors are stored fp16
///   - dispatch on `t.storage == .fp16`; everything else is W4
///
/// All dequant helpers are `Data`-based (scope `withUnsafeBytes` internally) per D017 —
/// never pass a `baseAddress` across a closure boundary.
enum DiTWeights {

    /// Dequantize a full [rows, cols] matrix. `weight` is stored [out, in] = [rows, cols];
    /// rows = output dim, cols = input dim (K).
    static func dequantMatrix(_ t: AnimapkTensor, pack: AnimapkFile, rows: Int, cols: Int) -> [[Float]] {
        if t.storage == .fp16 {
            let flat = QuantDecoders.fp16ToFloat32(pack.dataBytes(t).data, count: rows * cols)
            return (0..<rows).map { r in Array(flat[(r * cols)..<((r + 1) * cols)]) }
        }
        // W4 (group 64, even K → low nibble)
        let scale = pack.scaleBytes(t).map { $0.data } ?? Data()
        let zero = pack.zeroBytes(t).map { $0.data } ?? Data()
        let flat = QuantDecoders.dequantW4(data: pack.dataBytes(t).data, scale: scale, zero: zero, k: rows * cols)
        return (0..<rows).map { r in Array(flat[(r * cols)..<((r + 1) * cols)]) }
    }

    /// Dequantize a length-`n` vector (norm/scale/bias).
    static func dequantVector(_ t: AnimapkTensor, pack: AnimapkFile, n: Int) -> [Float] {
        if t.storage == .fp16 {
            return QuantDecoders.fp16ToFloat32(pack.dataBytes(t).data, count: n)
        }
        let scale = pack.scaleBytes(t).map { $0.data } ?? Data()
        let zero = pack.zeroBytes(t).map { $0.data } ?? Data()
        return QuantDecoders.dequantW4(data: pack.dataBytes(t).data, scale: scale, zero: zero, k: n)
    }

    /// out[m,n] = a[m,k] × w[n,k]ᵀ  (weight stored rows=output, cols=input). Float32 accum.
    static func matmul(_ a: [[Float]], _ w: [[Float]], m: Int, k: Int, n: Int) -> [[Float]] {
        var out = [[Float]](repeating: [Float](repeating: 0, count: n), count: m)
        for i in 0..<m {
            let aRow = a[i]
            for j in 0..<n {
                var acc: Float = 0
                let wRow = w[j]
                for p in 0..<k { acc += aRow[p] * wRow[p] }
                out[i][j] = acc
            }
        }
        return out
    }

    /// Exact GELU (torch.nn.GELU default, approximate='none') — shared by DiT MLP and adapter.
    static func gelu(_ x: Float) -> Float {
        0.5 * x * (1 + erf(x / sqrt(2.0)))
    }

    /// RMSNorm over the last dimension, Float32 accumulation. eps from model metadata (1e-6).
    static func rmsNorm(_ x: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(x.count == weight.count)
        var sum: Float = 0
        for v in x { sum += v * v }
        let inv = 1.0 / sqrt(sum / Float(x.count) + eps)
        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count { out[i] = x[i] * inv * weight[i] }
        return out
    }

    /// Row-wise RMSNorm over a [rows, cols] matrix.
    static func rmsNormRows(_ x: [[Float]], weight: [Float], eps: Float) -> [[Float]] {
        x.map { rmsNorm($0, weight: weight, eps: eps) }
    }
}
