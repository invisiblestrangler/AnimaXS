import SwiftUI

@main
struct AnimaXSApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        #if !targetEnvironment(simulator)
        // Experiment branch only: Stage 2N isolates the loader regression observed
        // after Stage 2K. It compares the proven Stage-K cache-hit lifecycle against
        // fresh-object, localModelPath, current Stage-2L helper, and same/new-object
        // reload paths on separate real blocks. No diffusion is involved.
        DispatchQueue.global(qos: .userInitiated).async {
            let result = A12ANEStage2NProbe() ?? "Stage 2N loader parity probe returned nil"
            print("\n========== ANIMAXS_ANE_MULTIPROC_LOADER_STAGE2N ==========\n\(result)\n============================================================\n")
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
