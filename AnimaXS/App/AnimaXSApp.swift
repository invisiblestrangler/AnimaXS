import SwiftUI

@main
struct AnimaXSApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        #if !targetEnvironment(simulator)
        // Experiment branch only: Stage 2G proved one canonical real W8
        // procedure compiles. Stage 2H tests a real canonical two-procedure
        // pair, then all eight block-0 W8 procedures in one loaded ANE model.
        // No prompt, diffusion, VAE, or image generation is involved.
        DispatchQueue.global(qos: .userInitiated).async {
            let result = A12ANEStage2HProbe() ?? "Stage 2H block0 probe returned nil"
            print("\n========== ANIMAXS_ANE_BLOCK0_MULTIPROC_STAGE2H ==========\n\(result)\n===========================================================\n")
        }
        #endif
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .onChange(of: scenePhase) { _, phase in
                    switch phase {
                    case .background, .inactive:
                        // K003: request cooperative cancellation at the next
                        // block boundary. There is no checkpoint/resume state
                        // to retain — the run simply ends cancelled and a
                        // fresh Generate is available afterward. The
                        // ContentView's coordinator handles it.
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
