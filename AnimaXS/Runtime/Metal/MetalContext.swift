import Foundation
import Metal
import MetalPerformanceShaders
import os

/// Owns the Metal device, command queue, default library, and compute pipeline cache.
/// Probes and records A12/Apple5-relevant capabilities for diagnostics (runbook §21).
///
/// A12 (Apple5) constraints honored:
///   - no simdgroup_matrix, no Metal 3, no MTLIOCommandQueue, no bfloat16
///   - workloads are submitted async; never block the UI thread
final class MetalContext {

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelineCache: [String: MTLComputePipelineState] = [:]

    // Runtime capability probe (runbook §21)
    private(set) var supportsApple5: Bool = false
    private(set) var maxBufferLength: Int = 0
    private(set) var maxThreadgroupMemoryLength: Int = 0
    private(set) var recommendedMaxWorkingSetSize: UInt64 = 0
    private(set) var currentAllocatedSize: UInt64 = 0
    private(set) var physicalMemory: UInt64 = 0
    private(set) var thermalState: ProcessInfo.ThermalState = .nominal

    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        self.device = device
        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        do {
            self.library = try device.makeDefaultLibrary(bundle: .main)
        } catch {
            os_log(.error, "failed to load default Metal library: %@", error.localizedDescription)
            return nil
        }
        self.supportsApple5 = device.supportsFamily(.apple5)
        self.maxBufferLength = device.maxBufferLength
        self.maxThreadgroupMemoryLength = device.maxThreadgroupMemoryLength
        self.recommendedMaxWorkingSetSize = device.recommendedMaxWorkingSetSize
        refreshDiagnostics()
    }

    /// Refresh values that can change while a generation is running.
    func refreshDiagnostics() {
        currentAllocatedSize = UInt64(device.currentAllocatedSize)
        physicalMemory = ProcessInfo.processInfo.physicalMemory
        thermalState = ProcessInfo.processInfo.thermalState
    }

    /// Fetch (or build) a compute pipeline state for a named kernel.
    func pipeline(named name: String) throws -> MTLComputePipelineState {
        if let p = pipelineCache[name] { return p }
        guard let fn = library.makeFunction(name: name) else {
            throw AnimapkError.validation("Metal kernel '\(name)' not found in library")
        }
        let p = try device.makeComputePipelineState(function: fn)
        pipelineCache[name] = p
        return p
    }

    // MARK: - Async execution helper

    /// Run a set of compute encoder blocks on a background command buffer without blocking the
    /// caller thread. Completion is delivered via a checked continuation.
    func runAsync(_ body: @escaping (MTLCommandBuffer) -> Void) async throws {
        let buffer = commandQueue.makeCommandBuffer()!
        body(buffer)
        buffer.commit()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            buffer.addCompletedHandler { cb in
                if let err = cb.error {
                    cont.resume(throwing: err)
                } else {
                    cont.resume()
                }
            }
        }
    }
}

/// A reusable Metal buffer allocation pool. All buffers are created once and reused to
/// avoid per-call allocation pressure (runbook §19).
final class BufferPool {
    private let device: MTLDevice
    private var buffers: [String: MTLBuffer] = [:]

    init(device: MTLDevice) {
        self.device = device
    }

    func buffer(key: String, bytes: Int, options: MTLResourceOptions = .storageModeShared) -> MTLBuffer {
        if let b = buffers[key], b.length >= bytes { return b }
        let b = device.makeBuffer(length: bytes, options: options)!
        buffers[key] = b
        return b
    }

    func removeAll() {
        buffers.removeAll()
    }
}
