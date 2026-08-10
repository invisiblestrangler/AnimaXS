import XCTest
@testable import AnimaXS

/// Placeholder for the pure-reference test suites named in ci.yml job `reference-tests`.
/// Real implementations land with D001 (parser), D003/D004 (W4/W8), I001 (sampler),
/// H002/H003/H004 (CPU refs), D005 (manifest), I004 (checkpoint).
final class ReferenceTestsPlaceholder: XCTestCase {
    func testPlaceholder() {
        XCTAssertTrue(ModelConstants.sigma8Step.count == 9)
    }
}
