import XCTest
@testable import AnimaXS

final class SmokeTests: XCTestCase {
    func testPlaceholder() {
        XCTAssertEqual(2 + 2, 4)
    }

    func testSigmaScheduleMatchesHandoff() {
        XCTAssertEqual(ModelConstants.sigma8Step.count, 9)
        XCTAssertEqual(ModelConstants.sigma8Step[0], 1.0)
        XCTAssertEqual(ModelConstants.sigma8Step[8], 0.0)
        XCTAssertEqual(ModelConstants.sigma8Step[1], 0.95469, accuracy: 1e-5)
        XCTAssertEqual(ModelConstants.ditBlocks, 28)
    }
}
