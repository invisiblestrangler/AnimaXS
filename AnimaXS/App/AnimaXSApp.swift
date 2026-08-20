import SwiftUI

@main
struct AnimaXSApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        #if !targetEnvironment(simulator)
        // Experiment branch only: Stage 2K proved block-0's 10 procedures are
        // bit-exact against production. Stage 2L now measures the actual A12
        // residency ceiling of that architecture by progressively loading real
        // block containers, keeping admitted blocks resident, and stopping on
        // the first fresh pressure/pathology signal. No diffusion is involved.
        DispatchQueue.global(qos: .userInitiated).async {
            let result = A12ANEStage2LProbe() ?? "Stage 2L residency probe returned nil"
            print("\n========== ANIMAXS_ANE_MULTIPROC_RESIDENCY_STAGE2L ==========\n\(result)\n================================================================\n")
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
