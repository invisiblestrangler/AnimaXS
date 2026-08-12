import Foundation

/// Validated streaming groups for the fixed Wan T=1 decoder architecture.
///
/// Mirrors `DiTPreparationLocator`/`DiTFinalLayerLocator`: the pack's physical
/// blob order is alphabetical, so every logical group is selected by exact
/// tensor names and validated (shape, fp16 storage, range containment) rather
/// than by trusting blob order. Rank-5 causal weights are folded to their FINAL
/// temporal slice at execution time (D052); the two decoder `time_conv` tensors
/// are validated as present but never streamed at T=1 (no feature cache).
struct VAEDecoderLocator {
    static let expectedArchitecture: [String: [Int]] = {
        var table: [String: [Int]] = [:]
        func add(_ name: String, _ shape: [Int]) { table[name] = shape }
        func residual(_ prefix: String, _ ci: Int, _ co: Int) {
            add(prefix + "0.gamma", [ci, 1, 1, 1])
            add(prefix + "2.weight", [co, ci, 3, 3, 3])
            add(prefix + "2.bias", [co])
            add(prefix + "3.gamma", [co, 1, 1, 1])
            add(prefix + "6.weight", [co, co, 3, 3, 3])
            add(prefix + "6.bias", [co])
        }
        // post-quant + decoder input
        add("conv2.weight", [16, 16, 1, 1, 1])
        add("conv2.bias", [16])
        add("decoder.conv1.weight", [384, 16, 3, 3, 3])
        add("decoder.conv1.bias", [384])
        // middle
        residual("decoder.middle.0.residual.", 384, 384)
        add("decoder.middle.1.norm.gamma", [384, 1, 1])
        add("decoder.middle.1.to_qkv.weight", [1152, 384, 1, 1])
        add("decoder.middle.1.to_qkv.bias", [1152])
        add("decoder.middle.1.proj.weight", [384, 384, 1, 1])
        add("decoder.middle.1.proj.bias", [384])
        residual("decoder.middle.2.residual.", 384, 384)
        // upsample stage 0 (384 @ 64 -> 192 @ 128)
        for m in [0, 1, 2] { residual("decoder.upsamples.\(m).residual.", 384, 384) }
        add("decoder.upsamples.3.resample.1.weight", [192, 384, 3, 3])
        add("decoder.upsamples.3.resample.1.bias", [192])
        add("decoder.upsamples.3.time_conv.weight", [768, 384, 3, 1, 1])
        add("decoder.upsamples.3.time_conv.bias", [768])
        // stage 1 (192 -> 384 @ 128 -> 192 @ 256)
        residual("decoder.upsamples.4.residual.", 192, 384)
        add("decoder.upsamples.4.shortcut.weight", [384, 192, 1, 1, 1])
        add("decoder.upsamples.4.shortcut.bias", [384])
        for m in [5, 6] { residual("decoder.upsamples.\(m).residual.", 384, 384) }
        add("decoder.upsamples.7.resample.1.weight", [192, 384, 3, 3])
        add("decoder.upsamples.7.resample.1.bias", [192])
        add("decoder.upsamples.7.time_conv.weight", [768, 384, 3, 1, 1])
        add("decoder.upsamples.7.time_conv.bias", [768])
        // stage 2 (192 @ 256 -> 96 @ 512)
        for m in [8, 9, 10] { residual("decoder.upsamples.\(m).residual.", 192, 192) }
        add("decoder.upsamples.11.resample.1.weight", [96, 192, 3, 3])
        add("decoder.upsamples.11.resample.1.bias", [96])
        // stage 3 (96 @ 512)
        for m in [12, 13, 14] { residual("decoder.upsamples.\(m).residual.", 96, 96) }
        // head
        add("decoder.head.0.gamma", [96, 1, 1, 1])
        add("decoder.head.2.weight", [3, 96, 3, 3, 3])
        add("decoder.head.2.bias", [3])
        return table
    }()

    /// Names of decoder tensors that exist in the pack but are NOT executed at
    /// T=1 (the two upsample3d time branches have no feature cache).
    static let unexecutedAtT1: Set<String> = [
        "decoder.upsamples.3.time_conv.weight", "decoder.upsamples.3.time_conv.bias",
        "decoder.upsamples.7.time_conv.weight", "decoder.upsamples.7.time_conv.bias",
    ]

    /// Ordered execution groups. Group 0 is the smallest ring-relative span used
    /// to size the streamer; every group must fit.
    struct Group {
        let logicalIndex: Int
        let range: AnimapkExecutionRange
    }

    let groups: [Group]

    init(file: AnimapkFile, requiresCompleteArchitecture: Bool = true) throws {
        guard file.component == "vae" else {
            throw AnimapkError.validation("VAE decoder locator requires a vae component pack")
        }
        // 1) Full architecture validation: every expected tensor present, exact
        //    shape, fp16 storage, and no unexpected decoder/post-quant tensors.
        let decoderNames = Set(file.tensors.map(\.name))
        // Per-tensor validation always applies to present decoder tensors.
        for name in decoderNames {
            guard Self.expectedArchitecture[name] != nil else { continue }
            guard let tensor = file.tensor(named: name) else { continue }
            let shape = Self.expectedArchitecture[name]!
            guard tensor.shape == shape else {
                throw AnimapkError.validation(
                    "VAE decoder tensor \(name) shape \(tensor.shape) != expected \(shape)")
            }
            guard tensor.storage == .fp16 else {
                throw AnimapkError.validation("VAE decoder tensor \(name) is not fp16")
            }
        }
        // Completeness: every expected tensor must be present (production).
        if requiresCompleteArchitecture {
            for (name, shape) in Self.expectedArchitecture {
                guard let tensor = file.tensor(named: name) else {
                    throw AnimapkError.validation("VAE decoder tensor missing: \(name)")
                }
                guard tensor.shape == shape else {
                    throw AnimapkError.validation(
                        "VAE decoder tensor \(name) shape \(tensor.shape) != expected \(shape)")
                }
                guard tensor.storage == .fp16 else {
                    throw AnimapkError.validation("VAE decoder tensor \(name) is not fp16")
                }
            }
        }
        let extras = decoderNames.filter { name in
            (name.hasPrefix("decoder.") || name == "conv2.weight" || name == "conv2.bias")
                && Self.expectedArchitecture[name] == nil
        }
        guard extras.isEmpty else {
            throw AnimapkError.validation(
                "unexpected VAE decoder tensors: \(extras.sorted().joined(separator: ", "))")
        }

        // 2) Build one execution range per logical group from exact names.
        let groupNames: [[String]] = [
            ["conv2.weight", "conv2.bias"],
            ["decoder.conv1.weight", "decoder.conv1.bias"],
            ["decoder.middle.0.residual.0.gamma", "decoder.middle.0.residual.2.weight",
             "decoder.middle.0.residual.2.bias", "decoder.middle.0.residual.3.gamma",
             "decoder.middle.0.residual.6.weight", "decoder.middle.0.residual.6.bias"],
            ["decoder.middle.1.norm.gamma", "decoder.middle.1.to_qkv.weight",
             "decoder.middle.1.to_qkv.bias", "decoder.middle.1.proj.weight",
             "decoder.middle.1.proj.bias"],
            ["decoder.middle.2.residual.0.gamma", "decoder.middle.2.residual.2.weight",
             "decoder.middle.2.residual.2.bias", "decoder.middle.2.residual.3.gamma",
             "decoder.middle.2.residual.6.weight", "decoder.middle.2.residual.6.bias"],
        ]
        var ordered: [[String]] = groupNames
        // 15 upsample modules in execution order (0..14). Resample modules
        // (3, 7, 11) carry ONLY the resample 3x3; all others carry residual
        // (+ shortcut for the 192->384 channel-change module 4). Verified
        // against the real pack: modules 3/7/11 have zero residual tensors.
        for m in 0..<15 {
            var names: [String] = []
            let base = "decoder.upsamples.\(m)."
            if m == 4 {
                // 192->384 residual + shortcut (channel change).
                for suffix in ["residual.0.gamma", "residual.2.weight", "residual.2.bias",
                               "residual.3.gamma", "residual.6.weight", "residual.6.bias"] {
                    names.append(base + suffix)
                }
                names.append(contentsOf: [base + "shortcut.weight", base + "shortcut.bias"])
            } else if m == 3 || m == 7 || m == 11 {
                // Pure resample modules: nearest-exact 2x + 3x3 conv only.
                names.append(contentsOf: [base + "resample.1.weight", base + "resample.1.bias"])
            } else {
                // Residual-only module (384->384 or 192->192).
                for suffix in ["residual.0.gamma", "residual.2.weight", "residual.2.bias",
                               "residual.3.gamma", "residual.6.weight", "residual.6.bias"] {
                    names.append(base + suffix)
                }
            }
            // time_conv tensors stay OUT of the execution groups (unexecuted at T=1).
            ordered.append(names)
        }
        ordered.append([
            "decoder.head.0.gamma", "decoder.head.2.weight", "decoder.head.2.bias",
        ])

        var built: [Group] = []
        for (index, names) in ordered.enumerated() {
            let present = file.tensors.filter { names.contains($0.name) }
            if present.isEmpty {
                // In subset mode (synthetic tests) a group may be absent; the
                // full decoder relies on requiresCompleteArchitecture:true.
                if requiresCompleteArchitecture {
                    throw AnimapkError.validation(
                        "VAE decoder group \(index) has no tensors (\(names.first ?? "?"))")
                }
                continue
            }
            let range = try AnimapkRangeBuilder.executionRange(
                tensors: present, exactPrefix: commonPrefix(of: names), logicalIndex: index)
            guard Set(range.tensors.map(\.tensor.name)) == Set(names) else {
                throw AnimapkError.validation(
                    "VAE decoder group \(index) tensor set mismatch")
            }
            built.append(Group(logicalIndex: index, range: range))
        }
        // 3) Cross-group disjointness in physical space.
        let physical = built.sorted { $0.range.fileOffset < $1.range.fileOffset }
        for pair in zip(physical, physical.dropFirst()) {
            guard pair.0.range.fileRange.upperBound <= pair.1.range.fileRange.lowerBound else {
                throw AnimapkError.validation(
                    "VAE decoder groups \(pair.0.logicalIndex)/\(pair.1.logicalIndex) overlap")
            }
        }
        groups = built
    }

    func group(_ logicalIndex: Int) throws -> Group {
        guard groups.indices.contains(logicalIndex) else {
            throw AnimapkError.validation("VAE decoder group \(logicalIndex) is out of range")
        }
        return groups[logicalIndex]
    }

    /// Maximum streamed byte length across all groups (ring capacity).
    func maximumGroupLength() throws -> UInt64 {
        guard let maxLength = groups.map(\.range.length).max() else {
            throw AnimapkError.validation("VAE decoder has no groups")
        }
        return maxLength
    }
}

/// Longest common prefix of a list of names — used only as the builder's prefix
/// hint; the builder still validates exact membership via the filtered tensor set.
private func commonPrefix(of names: [String]) -> String {
    guard let first = names.first else { return "" }
    var prefix = first
    for name in names.dropFirst() {
        while !name.hasPrefix(prefix) {
            prefix = String(prefix.dropLast())
            if prefix.isEmpty { break }
        }
    }
    return prefix
}
