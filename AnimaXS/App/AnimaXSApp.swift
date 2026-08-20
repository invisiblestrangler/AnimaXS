import SwiftUI

@main
struct AnimaXSApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        #if !targetEnvironment(simulator)
        // Experiment branch only: Stage 2O measures how far the ~14 ms ANE
        // runtime-hot reload state survives across distinct multiprocedure model
        // identities. The key production point is reuse distance 21, matching the
        // proven 6-pinned + 2-streaming scheduler's 22 streamed blocks.
        DispatchQueue.global(qos: .userInitiated).async {
            let result = A12ANEStage2OProbe() ?? "Stage 2O hot-cache reuse-distance probe returned nil"
            print("\n========== ANIMAXS_ANE_MULTIPROC_HOT_CACHE_STAGE2O ==========\n\(result)\n================================================================\n")
        }
        #endif
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .onChange(of: scenePhase) { _, phase in
                    switch phase {
                    case .background, .inactive:
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
