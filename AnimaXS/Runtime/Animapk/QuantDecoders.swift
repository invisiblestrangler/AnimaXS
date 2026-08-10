import Foundation

/// CPU reference dequantizers — byte-exact reference for the Metal kernels (runbook §18).
///
/// W4 (per ANIMAPK_SPEC §5 and runbook §18):
///   group axis: K (input) dimension, group size 64
///   q: unsigned 0...15, scale: fp16, zero: fp16
///   even K index → low nibble of packed byte, odd K index → high nibble
///   value = q * scale + zero
///
/// W8:
///   unsigned uint8, group size 64, fp16 scale, fp16 zero
///   value = q * scale + zero
///
/// All methods take `Data` and scope `withUnsafeBytes` internally so callers never
/// handle a potentially-dangling buffer pointer.
enum QuantDecoders {

    /// Dequantize a W4 tensor into a Float32 output array.
    /// - Parameters:
    ///   - k: number of K elements (output length)
    ///   - groupSize: quantization group size along K (64)
    static func dequantW4(data: Data, scale: Data, zero: Data,
                          k: Int, groupSize: Int = 64) -> [Float] {
        var out = [Float](repeating: 0, count: k)
        guard k > 0 else { return out }
        data.withUnsafeBytes { dBuf in
            scale.withUnsafeBytes { sBuf in
                zero.withUnsafeBytes { zBuf in
                    let d8 = dBuf.baseAddress!.assumingMemoryBound(to: UInt8.self)
                    let s16 = sBuf.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    let z16 = zBuf.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    out.withUnsafeMutableBytes { ob in
                        let oF = ob.baseAddress!.assumingMemoryBound(to: Float.self)
                        for i in 0..<k {
                            let b = d8[i >> 1]
                            let q = (i & 1) == 0 ? UInt16(b & 0x0F) : UInt16(b >> 4)
                            let g = i / groupSize
                            let sc = Float(Float16(bitPattern: s16[g]))
                            let ze = Float(Float16(bitPattern: z16[g]))
                            oF[i] = Float(q) * sc + ze
                        }
                    }
                }
            }
        }
        return out
    }

    /// Dequantize a W8 tensor into a Float32 output array.
    static func dequantW8(data: Data, scale: Data, zero: Data,
                          k: Int, groupSize: Int = 64) -> [Float] {
        var out = [Float](repeating: 0, count: k)
        guard k > 0 else { return out }
        data.withUnsafeBytes { dBuf in
            scale.withUnsafeBytes { sBuf in
                zero.withUnsafeBytes { zBuf in
                    let d8 = dBuf.baseAddress!.assumingMemoryBound(to: UInt8.self)
                    let s16 = sBuf.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    let z16 = zBuf.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    out.withUnsafeMutableBytes { ob in
                        let oF = ob.baseAddress!.assumingMemoryBound(to: Float.self)
                        for i in 0..<k {
                            let q = UInt16(d8[i])
                            let g = i / groupSize
                            let sc = Float(Float16(bitPattern: s16[g]))
                            let ze = Float(Float16(bitPattern: z16[g]))
                            oF[i] = Float(q) * sc + ze
                        }
                    }
                }
            }
        }
        return out
    }

    /// Read a raw little-endian fp16 array into Float32.
    static func fp16ToFloat32(_ bytes: Data, count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        guard count > 0 else { return out }
        bytes.withUnsafeBytes { b in
            let b16 = b.baseAddress!.assumingMemoryBound(to: UInt16.self)
            out.withUnsafeMutableBytes { ob in
                let oF = ob.baseAddress!.assumingMemoryBound(to: Float.self)
                for i in 0..<count {
                    oF[i] = Float(Float16(bitPattern: b16[i]))
                }
            }
        }
        return out
    }
}
