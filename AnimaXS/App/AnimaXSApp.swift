import SwiftUI

@main
struct AnimaXSApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        #if !targetEnvironment(simulator)
        // Experiment branch only: Stage 2I isolated the sole illegal property
        // to self-QKV being one three-output procedure. Stage 2J splits that
        // already-lowered QKV network into q/k/v single-output procedures while
        // keeping exactly one loaded private ANE model per block. It proves the
        // split-QKV 3-procedure model first, then the full 10-procedure block.
        // No prompt, diffusion, VAE, or image generation is involved.
        DispatchQueue.global(qos: .userInitiated).async {
            let result = A12ANEStage2JProbe() ?? "Stage 2J block0 probe returned nil"
            print("\n========== ANIMAXS_ANE_BLOCK0_MULTIPROC_STAGE2J ==========\n\(result)\n===========================================================\n")
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
