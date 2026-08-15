import Foundation

/// Durable "currently running diagnostic test" marker.
///
/// A native Metal/MPS assertion or memory-pressure jetsam can kill the app
/// before Swift can produce a normal `.failed` result. Before each potentially
/// dangerous diagnostic test the view/engine persists the test name and a
/// `started` state; on completion it persists `completed`; after the whole run
/// it persists `clean`. If the app is killed mid-test, the next launch reads
/// the marker and can say exactly which test the previous run died in.
struct DiagnosticRunMarker {
    private enum Key {
        static let session = "AnimaXSDiagnosticRun.sessionUUID"
        static let currentTest = "AnimaXSDiagnosticRun.currentTest"
        static let state = "AnimaXSDiagnosticRun.state"
    }

    enum RunState: String {
        case started
        case completed
        case clean
    }

    let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    /// Starts a fresh diagnostic session (new UUID, no current test yet).
    func beginSession() {
        defaults.set(UUID().uuidString, forKey: Key.session)
        defaults.set(RunState.started.rawValue, forKey: Key.state)
        defaults.removeObject(forKey: Key.currentTest)
    }

    /// Persist the test we are about to run. Called immediately before the
    /// test body executes, so a crash during the test is attributable.
    func markStarted(_ test: String) {
        defaults.set(test, forKey: Key.currentTest)
        defaults.set(RunState.started.rawValue, forKey: Key.state)
    }

    /// Record successful completion of a test. Clears the current-test slot:
    /// the next `markStarted` re-arms it for the following test.
    func markCompleted(_ test: String) {
        if defaults.string(forKey: Key.currentTest) == test {
            defaults.removeObject(forKey: Key.currentTest)
        }
        defaults.set(RunState.completed.rawValue, forKey: Key.state)
    }

    /// Marks the whole session as finished cleanly (no crash marker).
    func markSessionClean() {
        defaults.set(RunState.clean.rawValue, forKey: Key.state)
        defaults.removeObject(forKey: Key.currentTest)
    }

    /// Returns the test the previous run was executing when it died, or nil if
    /// the previous run ended cleanly (or never started a test).
    func unfinishedTest() -> String? {
        guard defaults.string(forKey: Key.state) == RunState.started.rawValue else {
            return nil
        }
        return defaults.string(forKey: Key.currentTest)
    }
}
