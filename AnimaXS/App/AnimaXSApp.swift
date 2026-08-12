import SwiftUI

@main
struct AnimaXSApp: App {
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView()
                .onChange(of: scenePhase) { _, phase in
                    switch phase {
                    case .background, .inactive:
                        // K003: safe cancellation at the next block boundary;
                        // the latest completed-step checkpoint is retained.
                        // The ContentView's coordinator handles it.
                        NotificationCenter.default.post(
                            name: .animaXSAppDidEnterBackground, object: nil)
                    case .active:
                        NotificationCenter.default.post(
                            name: .animaXSAppWillEnterForeground, object: nil)
                    @unknown default:
                        break
                    }
                }
        }
    }
}

extension Notification.Name {
    static let animaXSAppDidEnterBackground = Notification.Name("animaXSAppDidEnterBackground")
    static let animaXSAppWillEnterForeground = Notification.Name("animaXSAppWillEnterForeground")
}
