import Foundation

/// A trusted-installation record for one model pack.
///
/// After a full size + SHA-256 verification (download/import/repair/migration),
/// the store persists a receipt so a normal cold launch can trust the file with
/// cheap stat/metadata checks only — no multi-GB re-hash on every launch.
///
/// The receipt is only valid when ALL of the following hold:
/// - the manifest entry is unchanged (same filename/size/SHA-256);
/// - the installed file still exists;
/// - the installed file size still equals the manifest size;
/// - the installed file's modification date still matches the receipt.
///
/// Any mismatch invalidates the receipt and forces one full off-main
/// verification (which then writes a fresh receipt).
struct VerificationReceipt: Codable, Equatable {
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    let component: ModelComponent
    let filename: String
    let expectedSize: UInt64
    let expectedSHA256: String
    let verifiedAt: Date
    let installedSize: UInt64
    let installedModificationDate: Date?
}

/// Small durable JSON store for the per-pack verification receipts.
/// File operations are intentionally synchronous: callers are always the
/// `ModelStore` actor (never `@MainActor`), and receipts are tiny.
struct VerificationReceiptStore {
    private let directory: URL

    init(directory: URL) {
        self.directory = directory
    }

    private var fileURL: URL {
        directory.appendingPathComponent("verification-receipts.json", isDirectory: false)
    }

    func load() -> [String: VerificationReceipt] {
        guard let data = try? Data(contentsOf: fileURL),
              let receipts = try? JSONDecoder().decode([String: VerificationReceipt].self, from: data) else {
            return [:]
        }
        return receipts
    }

    func receipt(for filename: String) -> VerificationReceipt? {
        load()[filename]
    }

    func save(_ receipts: [String: VerificationReceipt]) throws {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let data = try JSONEncoder().encode(receipts)
        try data.write(to: fileURL, options: .atomic)
    }
}
