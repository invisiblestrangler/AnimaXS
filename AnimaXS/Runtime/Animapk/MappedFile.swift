import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Errors thrown by the ANMA parser and model validation.
///
/// `LocalizedError` conformance ensures `error.localizedDescription` surfaces
/// the actual reason (e.g. "model SHA-256 does not match manifest") instead of
/// a generic "The operation couldn't be completed. (AnimaXS.AnimapkError
/// error 3.)" message.
enum AnimapkError: Error, CustomStringConvertible, LocalizedError {
    case io(String)
    case header(String)
    case json(String)
    case validation(String)

    var description: String {
        switch self {
        case .io(let m), .header(let m), .json(let m), .validation(let m): return m
        }
    }

    var errorDescription: String? {
        description
    }
}

/// Read-only POSIX mmap wrapper. Owns the fd and mapping; unmaps/closes on deinit.
final class MappedFile {
    let fileSize: Int
    private let fd: Int32
    private let base: UnsafeMutableRawPointer

    init(url: URL) throws {
        let path = url.path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw AnimapkError.io("open failed for \(path): \(String(cString: strerror(errno)))") }
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            close(fd)
            throw AnimapkError.io("fstat failed for \(path)")
        }
        let size = Int(st.st_size)
        guard size > 0 else {
            close(fd)
            throw AnimapkError.io("file is empty: \(path)")
        }
        guard let base = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0),
              base != UnsafeMutableRawPointer(bitPattern: -1) else {
            close(fd)
            throw AnimapkError.io("mmap failed for \(path)")
        }
        self.fd = fd
        self.base = base
        self.fileSize = size
    }

    /// Entire mapped range.
    func bytes() -> UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: base, count: fileSize)
    }

    /// Pointer at a byte offset.
    func pointer(offset: Int) -> UnsafeRawPointer {
        UnsafeRawPointer(base.advanced(by: offset))
    }

    deinit {
        munmap(base, fileSize)
        close(fd)
    }
}

// MARK: - Anima DiT LoRA safetensors

/// The deliberately narrow v1 external-LoRA target set. These names match the
/// ten large per-block Linear modules in the production Anima DiT and the
/// default `networks.lora_anima` training surface in sd-scripts. Built-in
/// AdaLN-LoRA/modulation tensors are intentionally NOT external adapter targets.
enum DiTLoRATarget: String, CaseIterable, Hashable {
    case selfQ = "self_attn_q_proj"
    case selfK = "self_attn_k_proj"
    case selfV = "self_attn_v_proj"
    case selfO = "self_attn_output_proj"
    case crossQ = "cross_attn_q_proj"
    case crossK = "cross_attn_k_proj"
    case crossV = "cross_attn_v_proj"
    case crossO = "cross_attn_output_proj"
    case mlpUp = "mlp_layer1"
    case mlpDown = "mlp_layer2"

    var inputFeatures: Int {
        switch self {
        case .crossK, .crossV: return 1_024
        case .mlpDown: return 8_192
        default: return 2_048
        }
    }

    var outputFeatures: Int {
        switch self {
        case .mlpUp: return 8_192
        default: return 2_048
        }
    }
}

struct DiTLoRAKey: Hashable {
    let block: Int
    let target: DiTLoRATarget
}

enum LoRATensorDType: String {
    case f16 = "F16"
    case bf16 = "BF16"
    case f32 = "F32"
    case i64 = "I64"

    var byteWidth: Int {
        switch self {
        case .f16, .bf16: return 2
        case .f32: return 4
        case .i64: return 8
        }
    }
}

struct LoRATensorSlice {
    let dtype: LoRATensorDType
    let shape: [Int]
    let fileOffset: Int
    let byteCount: Int
}

struct DiTLoRAModuleDescriptor {
    let key: DiTLoRAKey
    let down: LoRATensorSlice
    let up: LoRATensorSlice
    let rank: Int
    let alpha: Float

    var scale: Float { alpha / Float(rank) }
}

/// Mmap-backed parser for native Anima sd-scripts `.safetensors` LoRAs.
///
/// Accepted module names are exactly the Comfy-compatible names emitted by
/// `networks.lora_anima`, for example:
/// `lora_unet_blocks_0_self_attn_q_proj.lora_down.weight`
/// `lora_unet_blocks_0_self_attn_q_proj.lora_up.weight`
/// `lora_unet_blocks_0_self_attn_q_proj.alpha`
///
/// The file mapping remains alive for the descriptor lifetime. Only the JSON
/// header is copied; tensor payloads are never loaded via `Data(contentsOf:)`.
final class DiTLoRAFile {
    static let blockCount = 28
    static let maximumHeaderBytes = 64 * 1_024 * 1_024

    let url: URL
    let modules: [DiTLoRAKey: DiTLoRAModuleDescriptor]
    private let mapped: MappedFile

    private struct PartialModule {
        var down: LoRATensorSlice?
        var up: LoRATensorSlice?
        var alpha: LoRATensorSlice?
    }

    init(url: URL) throws {
        self.url = url
        let mapped = try MappedFile(url: url)
        self.mapped = mapped
        let bytes = mapped.bytes()
        guard bytes.count >= 8, let base = bytes.baseAddress else {
            throw AnimapkError.header("LoRA safetensors file is too small")
        }

        var rawHeaderLength: UInt64 = 0
        memcpy(&rawHeaderLength, base, MemoryLayout<UInt64>.size)
        let headerLength64 = UInt64(littleEndian: rawHeaderLength)
        guard headerLength64 > 0,
              headerLength64 <= UInt64(Self.maximumHeaderBytes),
              headerLength64 <= UInt64(Int.max) else {
            throw AnimapkError.header("invalid LoRA safetensors header length \(headerLength64)")
        }
        let headerLength = Int(headerLength64)
        let dataStart = 8 + headerLength
        guard dataStart >= 8, dataStart <= bytes.count else {
            throw AnimapkError.header("LoRA safetensors header exceeds file size")
        }

        let headerData = Data(bytes: base.advanced(by: 8), count: headerLength)
        let jsonObject: Any
        do {
            jsonObject = try JSONSerialization.jsonObject(with: headerData)
        } catch {
            throw AnimapkError.json("invalid LoRA safetensors JSON header: \(error.localizedDescription)")
        }
        guard let header = jsonObject as? [String: Any] else {
            throw AnimapkError.json("LoRA safetensors header is not a JSON object")
        }

        var partials: [DiTLoRAKey: PartialModule] = [:]
        var unsupportedAnimaModules = Set<String>()

        for (name, rawValue) in header where name != "__metadata__" {
            guard let object = rawValue as? [String: Any] else {
                throw AnimapkError.json("invalid safetensors entry for \(name)")
            }
            let slice = try Self.parseTensor(
                name: name, object: object, dataStart: dataStart, fileSize: bytes.count)

            guard let parsedName = try Self.parseAnimaTensorName(name) else {
                if name.hasPrefix("lora_unet_") || name.hasPrefix("lora_te") {
                    let stem = name.split(separator: ".", maxSplits: 1).first.map(String.init) ?? name
                    unsupportedAnimaModules.insert(stem)
                }
                continue
            }
            var partial = partials[parsedName.key] ?? PartialModule()
            switch parsedName.kind {
            case .down:
                guard partial.down == nil else {
                    throw AnimapkError.validation("duplicate LoRA down tensor for \(name)")
                }
                partial.down = slice
            case .up:
                guard partial.up == nil else {
                    throw AnimapkError.validation("duplicate LoRA up tensor for \(name)")
                }
                partial.up = slice
            case .alpha:
                guard partial.alpha == nil else {
                    throw AnimapkError.validation("duplicate LoRA alpha tensor for \(name)")
                }
                partial.alpha = slice
            }
            partials[parsedName.key] = partial
        }

        // v1 must never silently partially apply an adapter trained for Anima
        // modules we do not support yet (text encoder, modulation, embedders,
        // final layer, etc.). A clean rejection is safer than a plausible but
        // semantically incomplete image.
        if let first = unsupportedAnimaModules.sorted().first {
            throw AnimapkError.validation(
                "unsupported Anima LoRA module \(first); v1 supports DiT attention Q/K/V/O and MLP layer1/layer2 only")
        }
        guard !partials.isEmpty else {
            throw AnimapkError.validation("LoRA contains no supported Anima DiT projection modules")
        }

        var complete: [DiTLoRAKey: DiTLoRAModuleDescriptor] = [:]
        complete.reserveCapacity(partials.count)
        for (key, partial) in partials {
            guard let down = partial.down, let up = partial.up, let alphaTensor = partial.alpha else {
                throw AnimapkError.validation(
                    "incomplete LoRA module blocks.\(key.block).\(key.target.rawValue): down/up/alpha are all required")
            }
            guard down.shape.count == 2, up.shape.count == 2 else {
                throw AnimapkError.validation("LoRA matrices must be rank-2 tensors")
            }
            let rank = down.shape[0]
            guard rank > 0,
                  down.shape == [rank, key.target.inputFeatures],
                  up.shape == [key.target.outputFeatures, rank] else {
                throw AnimapkError.validation(
                    "LoRA shape mismatch for block \(key.block) \(key.target.rawValue): down=\(down.shape), up=\(up.shape)")
            }
            let alpha = try Self.readScalar(alphaTensor, mapped: mapped)
            guard alpha.isFinite, alpha > 0 else {
                throw AnimapkError.validation(
                    "invalid LoRA alpha \(alpha) for block \(key.block) \(key.target.rawValue)")
            }
            complete[key] = DiTLoRAModuleDescriptor(
                key: key, down: down, up: up, rank: rank, alpha: alpha)
        }
        self.modules = complete
    }

    func module(block: Int, target: DiTLoRATarget) -> DiTLoRAModuleDescriptor? {
        modules[DiTLoRAKey(block: block, target: target)]
    }

    /// Copies one rank-2 tensor into an fp16 row-padded destination. This is
    /// used once at generation setup to build MPS-friendly adapter buffers.
    /// F16 takes the memcpy fast path; BF16/F32 are converted without creating
    /// an unbounded intermediate `Data` allocation.
    func copyMatrixToFP16(
        _ tensor: LoRATensorSlice,
        destination: UnsafeMutableRawPointer,
        destinationRowBytes: Int
    ) throws {
        guard tensor.shape.count == 2 else {
            throw AnimapkError.validation("LoRA matrix copy requires rank-2 tensor")
        }
        let rows = tensor.shape[0], columns = tensor.shape[1]
        let tightDestinationBytes = columns * MemoryLayout<Float16>.stride
        guard rows > 0, columns > 0, destinationRowBytes >= tightDestinationBytes else {
            throw AnimapkError.validation("invalid LoRA matrix destination stride")
        }
        let source = mapped.pointer(offset: tensor.fileOffset)
        switch tensor.dtype {
        case .f16:
            let sourceRowBytes = columns * 2
            for row in 0..<rows {
                memcpy(
                    destination.advanced(by: row * destinationRowBytes),
                    source.advanced(by: row * sourceRowBytes),
                    sourceRowBytes)
            }
        case .bf16:
            for row in 0..<rows {
                let destinationRow = destination.advanced(by: row * destinationRowBytes)
                    .bindMemory(to: Float16.self, capacity: columns)
                for column in 0..<columns {
                    var bits: UInt16 = 0
                    memcpy(&bits, source.advanced(by: (row * columns + column) * 2), 2)
                    let value = Float(bitPattern: UInt32(UInt16(littleEndian: bits)) << 16)
                    destinationRow[column] = Float16(value)
                }
            }
        case .f32:
            for row in 0..<rows {
                let destinationRow = destination.advanced(by: row * destinationRowBytes)
                    .bindMemory(to: Float16.self, capacity: columns)
                for column in 0..<columns {
                    var bits: UInt32 = 0
                    memcpy(&bits, source.advanced(by: (row * columns + column) * 4), 4)
                    destinationRow[column] = Float16(Float(bitPattern: UInt32(littleEndian: bits)))
                }
            }
        case .i64:
            throw AnimapkError.validation("I64 is valid only for a LoRA alpha scalar, not matrix weights")
        }
    }

    private enum TensorKind { case down, up, alpha }

    private static func parseAnimaTensorName(
        _ name: String
    ) throws -> (key: DiTLoRAKey, kind: TensorKind)? {
        let downSuffix = ".lora_down.weight"
        let upSuffix = ".lora_up.weight"
        let alphaSuffix = ".alpha"
        let stem: String
        let kind: TensorKind
        if name.hasSuffix(downSuffix) {
            stem = String(name.dropLast(downSuffix.count)); kind = .down
        } else if name.hasSuffix(upSuffix) {
            stem = String(name.dropLast(upSuffix.count)); kind = .up
        } else if name.hasSuffix(alphaSuffix) {
            stem = String(name.dropLast(alphaSuffix.count)); kind = .alpha
        } else {
            return nil
        }

        let prefix = "lora_unet_blocks_"
        guard stem.hasPrefix(prefix) else { return nil }
        let tail = stem.dropFirst(prefix.count)
        guard let separator = tail.firstIndex(of: "_") else {
            throw AnimapkError.validation("invalid Anima LoRA module name \(stem)")
        }
        guard let block = Int(tail[..<separator]), (0..<Self.blockCount).contains(block) else {
            throw AnimapkError.validation("invalid Anima LoRA block index in \(stem)")
        }
        let targetName = String(tail[tail.index(after: separator)...])
        guard let target = DiTLoRATarget(rawValue: targetName) else {
            return nil
        }
        return (DiTLoRAKey(block: block, target: target), kind)
    }

    private static func parseTensor(
        name: String, object: [String: Any], dataStart: Int, fileSize: Int
    ) throws -> LoRATensorSlice {
        guard let dtypeName = object["dtype"] as? String,
              let dtype = LoRATensorDType(rawValue: dtypeName),
              let rawShape = object["shape"] as? [Any],
              let rawOffsets = object["data_offsets"] as? [Any], rawOffsets.count == 2 else {
            throw AnimapkError.json("invalid tensor metadata for \(name)")
        }
        let shape = try rawShape.map { value -> Int in
            guard let number = value as? NSNumber else {
                throw AnimapkError.json("invalid shape in \(name)")
            }
            let v = number.intValue
            guard v >= 0 else { throw AnimapkError.json("negative shape in \(name)") }
            return v
        }
        guard let startNumber = rawOffsets[0] as? NSNumber,
              let endNumber = rawOffsets[1] as? NSNumber else {
            throw AnimapkError.json("invalid data offsets in \(name)")
        }
        let relativeStart = startNumber.intValue, relativeEnd = endNumber.intValue
        guard relativeStart >= 0, relativeEnd >= relativeStart,
              dataStart <= Int.max - relativeEnd else {
            throw AnimapkError.validation("invalid data range for \(name)")
        }
        let absoluteStart = dataStart + relativeStart
        let absoluteEnd = dataStart + relativeEnd
        guard absoluteEnd <= fileSize else {
            throw AnimapkError.validation("tensor \(name) exceeds LoRA file size")
        }
        let elementCount = try checkedElementCount(shape, name: name)
        guard elementCount <= Int.max / dtype.byteWidth,
              elementCount * dtype.byteWidth == relativeEnd - relativeStart else {
            throw AnimapkError.validation("tensor byte length does not match dtype/shape for \(name)")
        }
        return LoRATensorSlice(
            dtype: dtype, shape: shape, fileOffset: absoluteStart,
            byteCount: relativeEnd - relativeStart)
    }

    private static func checkedElementCount(_ shape: [Int], name: String) throws -> Int {
        if shape.isEmpty { return 1 }
        var count = 1
        for dimension in shape {
            guard dimension > 0, count <= Int.max / dimension else {
                throw AnimapkError.validation("invalid/overflowing tensor shape for \(name)")
            }
            count *= dimension
        }
        return count
    }

    private static func readScalar(_ tensor: LoRATensorSlice, mapped: MappedFile) throws -> Float {
        let count = try checkedElementCount(tensor.shape, name: "alpha")
        guard count == 1 else {
            throw AnimapkError.validation("LoRA alpha must contain exactly one value")
        }
        let source = mapped.pointer(offset: tensor.fileOffset)
        switch tensor.dtype {
        case .f16:
            var bits: UInt16 = 0; memcpy(&bits, source, 2)
            return Float(Float16(bitPattern: UInt16(littleEndian: bits)))
        case .bf16:
            var bits: UInt16 = 0; memcpy(&bits, source, 2)
            return Float(bitPattern: UInt32(UInt16(littleEndian: bits)) << 16)
        case .f32:
            var bits: UInt32 = 0; memcpy(&bits, source, 4)
            return Float(bitPattern: UInt32(littleEndian: bits))
        case .i64:
            var bits: UInt64 = 0; memcpy(&bits, source, 8)
            return Float(Int64(bitPattern: UInt64(littleEndian: bits)))
        }
    }
}
