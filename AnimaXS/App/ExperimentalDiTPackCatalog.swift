import Foundation

/// Observable catalog for the experimental W8 DiT pack, used by Diagnostics.
/// Wraps `ExperimentalDiTPackStore` and exposes import/remove/state so the
/// Diagnostics picker can gate W8 selection on a verified, ready pack.
@MainActor
final class ExperimentalDiTPackCatalog: ObservableObject {
    @Published private(set) var state: ExperimentalDiTPackStore.State = .missing
    @Published private(set) var message: String?

    private let store: ExperimentalDiTPackStore?

    init(store: ExperimentalDiTPackStore? = nil) {
        if let store {
            self.store = store
        } else {
            self.store = try? ExperimentalDiTPackStore()
        }
    }

    /// True when the experimental W8 pack is verified and available.
    var isReady: Bool {
        if case .ready = state { return true }
        return false
    }

    var readyURL: URL? {
        if case .ready(let url) = state { return url }
        return nil
    }

    /// Local-only discovery (no re-hashing of the multi-GB pack at launch).
    func refresh() async {
        guard let store else {
            state = .failed("Experimental W8 store unavailable")
            return
        }
        state = await store.discover()
    }

    /// User-triggered import from a security-scoped source URL. The caller
    /// (the Files importer) holds security-scoped access for the whole call.
    ///
    /// Publishes an in-progress state BEFORE awaiting the store so SwiftUI
    /// reflects the running multi-gigabyte import immediately (and the row
    /// hides its Import button for the duration) rather than continuing to
    /// show the previous state.
    func importPack(from source: URL) async {
        guard let store else {
            state = .failed("Experimental W8 store unavailable")
            return
        }
        state = .verifying
        message = "Importing and verifying W8 v2…"
        do {
            let url = try await store.importPack(from: source)
            state = .ready(url)
            message = "Imported experimental W8 v2 (verified size + SHA-256)."
        } catch {
            state = .failed(error.localizedDescription)
            message = error.localizedDescription
        }
    }

    /// Removes the experimental W8 pack and its receipt.
    func remove() async {
        guard let store else { return }
        await store.remove()
        state = .missing
        message = nil
    }
}
