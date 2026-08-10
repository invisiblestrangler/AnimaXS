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
enum QuantDecoders {

    /// Dequantize a W4 tensor into Float32 output.
    /// - Parameters:
    ///   - k: number of K elements (output length)
    ///   - groupSize: quantization group size along K (64)
    static func dequantW4(data: UnsafeRawBufferPointer,
                          scale: UnsafeRawBufferPointer,
                          zero: UnsafeRawBufferPointer,
                          k: Int,
                          groupSize: Int = 64,
                          into out: UnsafeMutableRawBufferPointer) {
        precondition(out.count >= k * MemoryLayout<Float>.stride)
        guard k > 0, let d = data.baseAddress, let s = scale.baseAddress,
              let z = zero.baseAddress, let o = out.baseAddress else { return }
        let d8 = d.assumingMemoryBound(to: UInt8.self)
        let s16 = s.assumingMemoryBound(to: UInt16.self)
        let z16 = z.assumingMemoryBound(to: UInt16.self)
        let oF = o.assumingMemoryBound(to: Float.self)
        for i in 0..<k {
            let b = d8[i >> 1]
            let q = (i & 1) == 0 ? UInt16(b & 0x0F) : UInt16(b >> 4)
            let g = i / groupSize
            let sc = Float(Float16(bitPattern: s16[g]))
            let ze = Float(Float16(bitPattern: z16[g]))
            oF[i] = Float(q) * sc + ze
        }
    }

    /// Dequantize a W8 tensor into Float32 output.
    static func dequantW8(data: UnsafeRawBufferPointer,
                          scale: UnsafeRawBufferPointer,
                          zero: UnsafeRawBufferPointer,
                          k: Int,
                          groupSize: Int = 64,
                          into out: UnsafeMutableRawBufferPointer) {
        precondition(out.count >= k * MemoryLayout<Float>.stride)
        guard k > 0, let d = data.baseAddress, let s = scale.baseAddress,
              let z = zero.baseAddress, let o = out.baseAddress else { return }
        let d8 = d.assumingMemoryBound(to: UInt8.self)
        let s16 = s.assumingMemoryBound(to: UInt16.self)
        let z16 = z.assumingMemoryBound(to: UInt16.self)
        let oF = o.assumingMemoryBound(to: Float.self)
        for i in 0..<k {
            let q = UInt16(d8[i])
            let g = i / groupSize
            let sc = Float(Float16(bitPattern: s16[g]))
            let ze = Float(Float16(bitPattern: z16[g]))
            oF[i] = Float(q) * sc + ze
        }
    }

    /// Read a raw fp16 (little-endian) array into Float32.
    static func fp16ToFloat32(_ bytes: UnsafeRawBufferPointer, count: Int, into out: UnsafeMutableRawBufferPointer) {
        precondition(out.count >= count * MemoryLayout<Float>.stride)
        guard count > 0, let b = bytes.baseAddress, let o = out.baseAddress else { return }
        let b16 = b.assumingMemoryBound(to: UInt16.self)
        let oF = o.assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            oF[i] = Float(Float16(bitPattern: b16[i]))
        }
    }
}
