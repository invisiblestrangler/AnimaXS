import SwiftUI

@main
struct AnimaXSApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        #if !targetEnvironment(simulator)
        // Experiment branch only: Stage 2H proved a canonical real W8
        // self_o+cross_q pair compiles and loads as two procedures, while the
        // all-eight container is rejected as InvalidProcedure. Stage 2I
        // isolates count, heterogeneous I/O geometry, and QKV multi-output.
        // No prompt, diffusion, VAE, or image generation is involved.
        DispatchQueue.global(qos: .userInitiated).async {
            let result = A12ANEStage2IProbe() ?? "Stage 2I block0 probe returned nil"
            print("\n========== ANIMAXS_ANE_BLOCK0_MULTIPROC_STAGE2I ==========\n\(result)\n===========================================================\n")
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
