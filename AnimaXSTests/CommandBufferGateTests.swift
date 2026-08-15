import XCTest
@testable import AnimaXS

/// Regression: the ping-pong restructure must register the MTLCommandBuffer
/// completed handler BEFORE commit (Metal asserts on late handlers), and the
/// await gate must be correct in both ordering directions.
final class CommandBufferGateTests: XCTestCase {

    func testWaitBeforeResume() async throws {
        let gate = CommandBufferGate()
        async let waiter: Void = { try await gate.wait() }()
        // Give the waiter a beat to store its continuation, then resume.
        try await Task.sleep(nanoseconds: 20_000_000)
        gate.resume()
        try await waiter
    }

    func testResumeBeforeWait() async throws {
        let gate = CommandBufferGate()
        gate.resume()
        try await gate.wait() // must not hang
    }

    func testResumeWithErrorBeforeWait() async {
        let gate = CommandBufferGate()
        struct Probe: Error {}
        gate.resume(throwing: Probe())
        do {
            try await gate.wait()
            XCTFail("expected the stored error to propagate")
        } catch is Probe {
            // expected
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }

    func testWaitWithErrorAfterResume() async {
        let gate = CommandBufferGate()
        struct Probe: Error {}
        async let waiter: Void = { try await gate.wait() }()
        try? await Task.sleep(nanoseconds: 20_000_000)
        gate.resume(throwing: Probe())
        do {
            try await waiter
            XCTFail("expected the error to propagate")
        } catch is Probe {
            // expected
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }
}
