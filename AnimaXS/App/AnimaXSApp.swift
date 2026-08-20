import SwiftUI

@main
struct AnimaXSApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        #if !targetEnvironment(simulator)
        // Experiment branch only: Stage 2L found a real first-compile residency
        // boundary at the 11th loaded 10-procedure block, but that admission still
        // carried the temporary ~180 MB construction/weight-map state. Stage 2M
        // first precompiles every block in isolation, then reloads descriptor-free
        // unloaded model handles to measure clean resident-program capacity and
        // warm scheduler load/unload costs. No diffusion is involved.
        DispatchQueue.global(qos: .userInitiated).async {
            let result = A12ANEStage2MProbe() ?? "Stage 2M cache-hit residency probe returned nil"
            print("\n========== ANIMAXS_ANE_MULTIPROC_RESIDENCY_STAGE2M ==========\n\(result)\n================================================================\n")
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
