import SwiftUI
import UniformTypeIdentifiers
import os
#if canImport(Darwin)
import Darwin
#endif

/// Image-first generation surface. Heavy diagnostics and run metrics deliberately
/// live in Diagnostics; this screen stays focused on the image, prompt, and the
/// few controls that materially shape the next generation.
struct ContentView: View {
    @StateObject private var coordinator = GenerationCoordinator()
    @StateObject private var catalog = ModelCatalog()
    @StateObject private var loraCatalog = LoRACatalog()
    @StateObject private var optimizationSettings = InferenceOptimizationSettings()

    @AppStorage("generation.lastPrompt")
    private var prompt = "masterpiece, best quality, score_7, safe, 1girl"
    @AppStorage("generation.lastNegativePrompt")
    private var negativePrompt = ""
    @AppStorage("generation.lastSeed")
    private var seedText = "1337"
    @AppStorage("generation.loraStrength")
    private var loraStrength = 1.0

    @State private var generationStart = Date()
    @State private var elapsedText = ""
    @State private var elapsedTimer: Timer?
    @State private var showDiagnostics = false
    @State private var showingImporter = false
    @State private var importComponent: ModelComponent?
    @State private var showingLoRAImporter = false
    @State private var showModelManager = false
    @State private var expandedPanel: ComposerPanel?
    @State private var lastVisibleImage: UIImage?
    @FocusState private var focusedField: FocusField?

    private enum FocusField: Hashable {
        case prompt
        case negative
        case seed
    }

    private enum ComposerPanel: Hashable {
        case lora
        case negative
        case seed
    }

    private static let generationLog = Logger(
        subsystem: "com.invisiblestrangler.AnimaXS", category: "Generation")

    private static let background = Color(
        red: 0.047, green: 0.043, blue: 0.039)
    private static let panel = Color(
        red: 0.088, green: 0.081, blue: 0.073)
    private static let panelRaised = Color(
        red: 0.118, green: 0.108, blue: 0.096)
    private static let cream = Color(
        red: 0.91, green: 0.84, blue: 0.69)
    private static let creamMuted = Color(
        red: 0.68, green: 0.61, blue: 0.49)

    var body: some View {
        NavigationStack {
            ZStack {
                Self.background.ignoresSafeArea()

                VStack(spacing: 0) {
                    topBar
                        .padding(.horizontal, 18)
                        .padding(.top, 8)

                    resultStage
                        .padding(.horizontal, 14)
                        .padding(.top, 12)
                        .frame(maxHeight: .infinity)

                    if let reason = eligibility.blockedReason, !isGenerating {
                        Text(reason)
                            .font(.caption2)
                            .foregroundStyle(.orange.opacity(0.9))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 26)
                            .padding(.bottom, 6)
                    }

                    composer
                        .padding(.horizontal, 12)
                        .padding(.bottom, 8)
                }

                if showModelManager {
                    modelManagerOverlay
                        .zIndex(20)
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(isPresented: $showDiagnostics) {
                DiagnosticsView(
                    lastMetricsText: coordinator.lastMetricsText,
                    optimizationSettings: optimizationSettings,
                    ditNumericsPolicy: resolvedDitNumericsPolicy,
                    ditVariantID: catalog.resolved?.dit.variant.id,
                    isGenerating: coordinator.isGenerating)
            }
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { focusedField = nil }
                }
            }
            .task {
                await catalog.refresh()
                await loraCatalog.refresh()
            }
            .onReceive(
                NotificationCenter.default.publisher(for: .animaXSAppDidEnterBackground)
            ) { _ in coordinator.appDidEnterBackground() }
            .onReceive(
                NotificationCenter.default.publisher(for: .animaXSAppWillEnterForeground)
            ) { _ in coordinator.appWillEnterForeground() }
            .onReceive(
                NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)
            ) { _ in coordinator.handleMemoryWarning() }
            .onChange(of: coordinator.state) { _, newState in
                switch newState {
                case .completed:
                    if let image = coordinator.image { lastVisibleImage = image }
                    stopElapsedTimer(updateFinalElapsed: true)
                case .cancelled, .failed:
                    stopElapsedTimer(updateFinalElapsed: true)
                default:
                    break
                }
            }
            .onDisappear {
                stopElapsedTimer(updateFinalElapsed: false)
            }
            .fileImporter(
                isPresented: $showingImporter,
                allowedContentTypes: [.data],
                allowsMultipleSelection: false
            ) { result in
                let component = importComponent
                importComponent = nil
                guard let component,
                      let url = try? result.get().first else { return }
                let didStart = url.startAccessingSecurityScopedResource()
                Task {
                    defer {
                        if didStart { url.stopAccessingSecurityScopedResource() }
                    }
                    await catalog.importPack(component, from: url)
                }
            }
            .fileImporter(
                isPresented: $showingLoRAImporter,
                allowedContentTypes: [.data],
                allowsMultipleSelection: false
            ) { result in
                guard !isGenerating,
                      let url = try? result.get().first else { return }
                let didStart = url.startAccessingSecurityScopedResource()
                Task {
                    defer {
                        if didStart { url.stopAccessingSecurityScopedResource() }
                    }
                    await loraCatalog.importAdapter(from: url)
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    // MARK: - Header

    private var topBar: some View {
        HStack(spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: 11, style: .continuous)
                    .fill(Self.cream)
                    .frame(width: 36, height: 36)
                Image(systemName: "sparkles")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(Self.background)
            }

            VStack(alignment: .leading, spacing: 1) {
                Text("AnimaXS")
                    .font(.system(size: 19, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                HStack(spacing: 5) {
                    Circle()
                        .fill(catalog.resolved == nil ? Color.orange : Color.green)
                        .frame(width: 6, height: 6)
                    Text(catalog.resolved == nil ? "Models need attention" : "On-device · private")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.52))
                }
            }

            Spacer()

            topIconButton(systemName: "shippingbox", badge: missingModelCount > 0 ? "\(missingModelCount)" : nil) {
                withAnimation(.spring(response: 0.38, dampingFraction: 0.86)) {
                    focusedField = nil
                    showModelManager = true
                }
            }
            topIconButton(systemName: "waveform.path.ecg", badge: nil) {
                focusedField = nil
                showDiagnostics = true
            }
        }
        .frame(height: 46)
    }

    private func topIconButton(
        systemName: String, badge: String?, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            ZStack(alignment: .topTrailing) {
                Image(systemName: systemName)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.78))
                    .frame(width: 38, height: 38)
                    .background(Self.panelRaised, in: Circle())
                if let badge {
                    Text(badge)
                        .font(.system(size: 9, weight: .bold, design: .rounded))
                        .foregroundStyle(Self.background)
                        .frame(minWidth: 16, minHeight: 16)
                        .background(Self.cream, in: Capsule())
                        .offset(x: 3, y: -3)
                }
            }
        }
        .buttonStyle(.plain)
    }

    // MARK: - Image stage

    private var resultStage: some View {
        GeometryReader { proxy in
            let side = min(proxy.size.width, proxy.size.height)
            ZStack {
                RoundedRectangle(cornerRadius: 30, style: .continuous)
                    .fill(Self.panel)

                if let image = coordinator.image ?? lastVisibleImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: side, height: side)
                        .clipped()
                        .opacity(isGenerating ? 0.34 : 1)
                        .scaleEffect(isGenerating ? 1.015 : 1)
                        .animation(.easeInOut(duration: 0.28), value: isGenerating)
                } else {
                    ambientPlaceholder
                }

                LinearGradient(
                    colors: [.clear, .black.opacity(isGenerating ? 0.48 : 0.18)],
                    startPoint: .center,
                    endPoint: .bottom)
                    .allowsHitTesting(false)

                if isGenerating {
                    generatingOverlay
                        .transition(.opacity)
                } else if coordinator.image != nil {
                    resultActions
                        .transition(.opacity.combined(with: .scale(scale: 0.96)))
                } else if case .failed(let message) = coordinator.state {
                    errorOverlay(message)
                } else {
                    idleHint
                }
            }
            .frame(width: side, height: side)
            .clipShape(RoundedRectangle(cornerRadius: 30, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 30, style: .continuous)
                    .stroke(.white.opacity(0.07), lineWidth: 1)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            .animation(.easeInOut(duration: 0.24), value: isGenerating)
        }
        .aspectRatio(1, contentMode: .fit)
    }

    private var ambientPlaceholder: some View {
        ZStack {
            RadialGradient(
                colors: [Self.cream.opacity(0.17), Self.panel, Self.background.opacity(0.8)],
                center: .topLeading,
                startRadius: 8,
                endRadius: 360)
            Circle()
                .fill(Self.cream.opacity(0.08))
                .frame(width: 180, height: 180)
                .blur(radius: 2)
            Image(systemName: "sparkles.rectangle.stack")
                .font(.system(size: 46, weight: .ultraLight))
                .foregroundStyle(Self.cream.opacity(0.48))
        }
    }

    private var idleHint: some View {
        VStack(spacing: 8) {
            Spacer()
            Text("Your next image lives here")
                .font(.system(size: 14, weight: .medium, design: .rounded))
                .foregroundStyle(.white.opacity(0.68))
            Text("Describe it below · generated entirely on this iPhone")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.38))
                .multilineTextAlignment(.center)
                .padding(.bottom, 20)
        }
        .padding(.horizontal, 20)
    }

    private var generatingOverlay: some View {
        VStack(spacing: 11) {
            Spacer()
            HStack(spacing: 8) {
                ProgressView()
                    .tint(Self.cream)
                    .controlSize(.small)
                Text(generationStatusText)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.88))
                    .lineLimit(1)
                if !elapsedText.isEmpty {
                    Text(elapsedText)
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .foregroundStyle(Self.cream.opacity(0.78))
                }
            }
            .padding(.horizontal, 13)
            .padding(.vertical, 9)
            .background(.black.opacity(0.48), in: Capsule())

            if let progress = generationProgress {
                ProgressView(value: progress)
                    .tint(Self.cream)
                    .frame(maxWidth: 220)
            }
        }
        .padding(.bottom, 22)
    }

    private var resultActions: some View {
        VStack {
            Spacer()
            HStack {
                Spacer()
                if let image = coordinator.image,
                   let url = shareURL(for: image) {
                    ShareLink(
                        item: url,
                        preview: SharePreview("AnimaXS image", image: Image(uiImage: image))
                    ) {
                        Label("Share", systemImage: "square.and.arrow.up")
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                            .foregroundStyle(Self.background)
                            .padding(.horizontal, 15)
                            .padding(.vertical, 10)
                            .background(Self.cream, in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(16)
    }

    private func errorOverlay(_ message: String) -> some View {
        VStack(spacing: 8) {
            Spacer()
            Label("Generation failed", systemImage: "exclamationmark.triangle.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(.orange)
            Text(message)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.65))
                .multilineTextAlignment(.center)
                .lineLimit(3)
                .padding(.horizontal, 24)
                .padding(.bottom, 18)
        }
    }

    // MARK: - Composer

    private var composer: some View {
        VStack(spacing: 10) {
            if let expandedPanel {
                expandedComposerPanel(expandedPanel)
                    .transition(
                        .move(edge: .bottom)
                        .combined(with: .opacity)
                    )
            }

            HStack(alignment: .bottom, spacing: 10) {
                TextEditor(text: $prompt)
                    .focused($focusedField, equals: .prompt)
                    .font(.system(size: 15, weight: .regular, design: .rounded))
                    .foregroundStyle(.white.opacity(0.92))
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 54, maxHeight: 96)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 2)
                    .overlay(alignment: .topLeading) {
                        if prompt.isEmpty {
                            Text("Describe your image…")
                                .font(.system(size: 15, design: .rounded))
                                .foregroundStyle(.white.opacity(0.30))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 10)
                                .allowsHitTesting(false)
                        }
                    }

                generationButton
            }

            HStack(spacing: 7) {
                composerChip(
                    title: loraChipTitle,
                    systemName: "square.3.layers.3d",
                    panel: .lora,
                    active: loraCatalog.selected != nil)
                composerChip(
                    title: negativeChipTitle,
                    systemName: "minus.circle",
                    panel: .negative,
                    active: !negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                composerChip(
                    title: "Seed",
                    systemName: "dice",
                    panel: .seed,
                    active: false)
                Spacer(minLength: 0)
                Text("CFG 1 · local")
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.28))
            }
        }
        .padding(12)
        .background(Self.panel, in: RoundedRectangle(cornerRadius: 25, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 25, style: .continuous)
                .stroke(.white.opacity(0.075), lineWidth: 1)
        }
        .shadow(color: .black.opacity(0.22), radius: 18, y: 8)
        .animation(.spring(response: 0.42, dampingFraction: 0.84), value: expandedPanel)
    }

    private var generationButton: some View {
        Button {
            if isGenerating {
                stopElapsedTimer(updateFinalElapsed: true)
                coordinator.cancel()
            } else {
                startGeneration()
            }
        } label: {
            Image(systemName: isGenerating ? "xmark" : "arrow.up")
                .font(.system(size: 17, weight: .bold))
                .foregroundStyle(isGenerating ? .white : Self.background)
                .frame(width: 48, height: 48)
                .background(
                    isGenerating ? Color.white.opacity(0.12) : Self.cream,
                    in: Circle())
                .overlay {
                    if isGenerating {
                        Circle().stroke(.white.opacity(0.16), lineWidth: 1)
                    }
                }
        }
        .buttonStyle(.plain)
        .disabled(!isGenerating && !eligibility.isReady)
        .opacity(!isGenerating && !eligibility.isReady ? 0.42 : 1)
        .animation(.easeInOut(duration: 0.2), value: isGenerating)
        .accessibilityLabel(isGenerating ? "Cancel generation" : "Generate image")
    }

    private func composerChip(
        title: String,
        systemName: String,
        panel: ComposerPanel,
        active: Bool
    ) -> some View {
        let selected = expandedPanel == panel
        return Button {
            withAnimation(.spring(response: 0.38, dampingFraction: 0.84)) {
                focusedField = nil
                expandedPanel = selected ? nil : panel
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: systemName)
                    .font(.system(size: 11, weight: .semibold))
                Text(title)
                    .font(.system(size: 11, weight: .semibold, design: .rounded))
                    .lineLimit(1)
                if active {
                    Circle()
                        .fill(Self.cream)
                        .frame(width: 5, height: 5)
                }
            }
            .foregroundStyle(selected ? Self.background : .white.opacity(active ? 0.84 : 0.58))
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(
                selected ? Self.cream : Self.panelRaised,
                in: Capsule())
        }
        .buttonStyle(.plain)
        .disabled(isGenerating)
    }

    @ViewBuilder
    private func expandedComposerPanel(_ panel: ComposerPanel) -> some View {
        switch panel {
        case .negative:
            negativePanel
        case .lora:
            loraPanel
        case .seed:
            seedPanel
        }
    }

    private var negativePanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Negative · NegPiP", systemImage: "minus.circle.fill")
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(Self.cream)
                Spacer()
                Text("single-pass")
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.34))
            }
            TextEditor(text: $negativePrompt)
                .focused($focusedField, equals: .negative)
                .font(.system(size: 14, design: .rounded))
                .foregroundStyle(.white.opacity(0.86))
                .scrollContentBackground(.hidden)
                .frame(height: 68)
                .padding(6)
                .background(Self.background.opacity(0.55), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay(alignment: .topLeading) {
                    if negativePrompt.isEmpty {
                        Text("Things to avoid — watermark, blurry, bad hands…")
                            .font(.system(size: 13, design: .rounded))
                            .foregroundStyle(.white.opacity(0.25))
                            .padding(.horizontal, 13)
                            .padding(.vertical, 14)
                            .allowsHitTesting(false)
                    }
                }
            Text("Combined with the main prompt at Generate; NegPiP subtracts these tokens through cross-attention V while keeping CFG at 1.")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.34))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(11)
        .background(Self.panelRaised.opacity(0.72), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var loraPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("LoRA stack")
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(Self.cream)
                Spacer()
                if loraCatalog.isImporting {
                    ProgressView().controlSize(.small).tint(Self.cream)
                }
                Button(loraCatalog.selected == nil ? "Add LoRA" : "Import / Replace") {
                    showingLoRAImporter = true
                }
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(Self.cream)
                .buttonStyle(.plain)
                .disabled(isGenerating || loraCatalog.isImporting)
            }

            if loraCatalog.adapters.isEmpty {
                HStack(spacing: 9) {
                    Image(systemName: "square.3.layers.3d")
                        .foregroundStyle(.white.opacity(0.28))
                    Text("No adapter active")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.38))
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
                .background(Self.background.opacity(0.48), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            } else {
                ForEach(loraCatalog.adapters, id: \.url) { adapter in
                    VStack(alignment: .leading, spacing: 9) {
                        HStack(spacing: 9) {
                            ZStack {
                                RoundedRectangle(cornerRadius: 9, style: .continuous)
                                    .fill(Self.cream.opacity(0.15))
                                    .frame(width: 34, height: 34)
                                Image(systemName: "wand.and.stars")
                                    .font(.system(size: 13, weight: .semibold))
                                    .foregroundStyle(Self.cream)
                            }
                            VStack(alignment: .leading, spacing: 2) {
                                Text(adapter.displayName)
                                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                                    .foregroundStyle(.white.opacity(0.9))
                                    .lineLimit(1)
                                Text("\(adapter.moduleCount) projection\(adapter.moduleCount == 1 ? "" : "s")")
                                    .font(.caption2)
                                    .foregroundStyle(.white.opacity(0.34))
                            }
                            Spacer()
                            Button(role: .destructive) {
                                loraCatalog.remove()
                            } label: {
                                Image(systemName: "trash")
                                    .font(.system(size: 12))
                                    .foregroundStyle(.red.opacity(0.75))
                                    .frame(width: 30, height: 30)
                            }
                            .buttonStyle(.plain)
                            .disabled(isGenerating || loraCatalog.isImporting)
                        }
                        HStack(spacing: 10) {
                            Text("Strength")
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.42))
                            Slider(value: $loraStrength, in: 0...2, step: 0.05)
                                .tint(Self.cream)
                                .disabled(isGenerating || loraCatalog.isImporting)
                            Text(loraStrength, format: .number.precision(.fractionLength(2)))
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                .foregroundStyle(Self.creamMuted)
                                .frame(width: 34, alignment: .trailing)
                        }
                    }
                    .padding(10)
                    .background(Self.background.opacity(0.48), in: RoundedRectangle(cornerRadius: 15, style: .continuous))
                }
            }

            if let error = loraCatalog.errorMessage {
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(.red.opacity(0.82))
            }
        }
        .padding(11)
        .background(Self.panelRaised.opacity(0.72), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var seedPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Seed")
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(Self.cream)
            HStack(spacing: 10) {
                TextField("Seed", text: $seedText)
                    .keyboardType(.numberPad)
                    .focused($focusedField, equals: .seed)
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.86))
                    .padding(.horizontal, 12)
                    .frame(height: 38)
                    .background(Self.background.opacity(0.55), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                Button {
                    seedText = String(UInt64.random(in: 0..<UInt64.max))
                } label: {
                    Label("Random", systemImage: "dice")
                        .font(.system(size: 11, weight: .semibold, design: .rounded))
                        .foregroundStyle(Self.background)
                        .padding(.horizontal, 12)
                        .frame(height: 38)
                        .background(Self.cream, in: Capsule())
                }
                .buttonStyle(.plain)
                .disabled(isGenerating)
            }
        }
        .padding(11)
        .background(Self.panelRaised.opacity(0.72), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    // MARK: - Model manager overlay

    private var modelManagerOverlay: some View {
        ZStack(alignment: .top) {
            Color.black.opacity(0.62)
                .ignoresSafeArea()
                .onTapGesture {
                    withAnimation(.spring(response: 0.36, dampingFraction: 0.88)) {
                        showModelManager = false
                    }
                }

            VStack(spacing: 13) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Model library")
                            .font(.system(size: 18, weight: .semibold, design: .rounded))
                            .foregroundStyle(.white)
                        Text("Base packs stay out of the creative workspace")
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.35))
                    }
                    Spacer()
                    Button {
                        withAnimation(.spring(response: 0.36, dampingFraction: 0.88)) {
                            showModelManager = false
                        }
                    } label: {
                        Image(systemName: "xmark")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(.white.opacity(0.68))
                            .frame(width: 34, height: 34)
                            .background(Self.panelRaised, in: Circle())
                    }
                    .buttonStyle(.plain)
                }

                ForEach(ModelComponent.allCases, id: \.self) { component in
                    modelCard(component)
                }
            }
            .padding(16)
            .background(Self.panel, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 28, style: .continuous)
                    .stroke(.white.opacity(0.08), lineWidth: 1)
            }
            .padding(.horizontal, 12)
            .padding(.top, 62)
            .transition(.move(edge: .top).combined(with: .opacity))
        }
    }

    @ViewBuilder
    private func modelCard(_ component: ModelComponent) -> some View {
        let state = catalog.state(for: component)
        HStack(spacing: 11) {
            Circle()
                .fill(stateColor(state).opacity(0.9))
                .frame(width: 8, height: 8)
            VStack(alignment: .leading, spacing: 3) {
                Text(component.displayName)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white.opacity(0.86))
                    .lineLimit(1)
                Text(stateLabel(state))
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.38))
                if case .failed(let message) = state {
                    Text(message)
                        .font(.system(size: 9))
                        .foregroundStyle(.red.opacity(0.75))
                        .lineLimit(2)
                }
            }
            Spacer()
            modelActions(component, state: state)
        }
        .padding(12)
        .background(Self.panelRaised.opacity(0.72), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    @ViewBuilder
    private func modelActions(_ component: ModelComponent, state: ModelStore.State) -> some View {
        switch state {
        case .ready:
            smallModelButton("Repair") {
                Task { await catalog.repair(component) }
            }
        case .failed:
            HStack(spacing: 6) {
                smallModelButton("Retry") { Task { await catalog.retry(component) } }
                smallModelButton("Import") {
                    importComponent = component
                    showingImporter = true
                }
            }
        case .missing:
            HStack(spacing: 6) {
                smallModelButton("Download") { Task { await catalog.download(component) } }
                smallModelButton("Import") {
                    importComponent = component
                    showingImporter = true
                }
            }
        case .downloading, .verifying:
            ProgressView().controlSize(.small).tint(Self.cream)
        }
    }

    private func smallModelButton(_ title: String, action: @escaping () -> Void) -> some View {
        Button(title, action: action)
            .font(.system(size: 10, weight: .semibold, design: .rounded))
            .foregroundStyle(Self.cream)
            .buttonStyle(.plain)
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(Self.background.opacity(0.58), in: Capsule())
            .disabled(isGenerating)
    }

    // MARK: - Generation

    private var resolvedDitNumericsPolicy: DiTNumericsPolicy {
        if let resolved = catalog.resolved {
            return DiTNumericsPolicy.fromVariantID(resolved.dit.variant.id)
        }
        return .w4Legacy
    }

    private var eligibility: GenerationEligibility {
        let optimizationBlockingReason: String?
        if loraCatalog.isImporting {
            optimizationBlockingReason = "LoRA import is still in progress."
        } else {
            optimizationBlockingReason = InferenceOptimizationConfig.blockingReason(
                for: optimizationSettings.snapshot,
                numerics: resolvedDitNumericsPolicy,
                ditVariantID: catalog.resolved?.dit.variant.id)
        }
        return GenerationEligibility.evaluate(
            modelsResolved: catalog.resolved != nil,
            isGenerating: isGenerating,
            prompt: prompt,
            seedText: seedText,
            metalAvailable: coordinator.isMetalAvailable,
            optimizationBlockingReason: optimizationBlockingReason)
    }

    private func startGeneration() {
        guard case .ready = eligibility else {
            Self.generationLog.warning(
                "generation blocked: \(eligibility.blockedReason ?? "unknown", privacy: .public)")
            return
        }
        guard let seed = UInt64(seedText),
              let baseModels = catalog.resolved else { return }

        if let current = coordinator.image { lastVisibleImage = current }
        focusedField = nil
        withAnimation(.spring(response: 0.34, dampingFraction: 0.88)) {
            expandedPanel = nil
        }

        let adapterSnapshot: ResolvedLoRA?
        if let selected = loraCatalog.selected, loraStrength != 0 {
            adapterSnapshot = ResolvedLoRA(
                url: selected.url,
                displayName: selected.displayName,
                strength: Float(loraStrength))
        } else {
            adapterSnapshot = nil
        }
        let trimmedNegative = negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let negPiPSnapshot = trimmedNegative.isEmpty
            ? nil
            : ResolvedNegPiP(prompt: negativePrompt)
        let models = baseModels
            .withLoRA(adapterSnapshot)
            .withNegPiP(negPiPSnapshot)

        generationStart = Date()
        elapsedText = ""
        elapsedTimer?.invalidate()
        elapsedTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            elapsedText = String(format: "%.1f s", Date().timeIntervalSince(generationStart))
        }
        coordinator.generate(
            prompt: prompt, seed: seed, models: models,
            optimization: optimizationSettings.snapshot)
    }

    private func stopElapsedTimer(updateFinalElapsed: Bool = true) {
        guard elapsedTimer != nil else { return }
        elapsedTimer?.invalidate()
        elapsedTimer = nil
        if updateFinalElapsed {
            elapsedText = String(format: "%.1f s", Date().timeIntervalSince(generationStart))
        }
    }

    private var isGenerating: Bool { coordinator.isGenerating }

    private var generationProgress: Double? {
        if case .diffusing(let step, let block, let totalSteps, let totalBlocks) = coordinator.state {
            let completed = Double(max(0, step - 1)) + Double(block) / Double(totalBlocks)
            return min(1, completed / Double(totalSteps))
        }
        if case .decoding = coordinator.state { return 0.985 }
        return nil
    }

    private var generationStatusText: String {
        switch coordinator.state {
        case .idle: return "Ready"
        case .tokenizing: return "Preparing prompt"
        case .encodingPrompt: return "Reading prompt"
        case .adapting: return "Building context"
        case .diffusing(let step, let block, let totalSteps, let totalBlocks):
            return "Step \(step)/\(totalSteps) · block \(block)/\(totalBlocks)"
        case .decoding: return "Developing image"
        case .completed: return "Done"
        case .cancelled: return "Cancelled"
        case .failed: return "Failed"
        }
    }

    private var loraChipTitle: String {
        guard !loraCatalog.adapters.isEmpty else { return "LoRA" }
        return "LoRA \(loraCatalog.adapters.count)"
    }

    private var negativeChipTitle: String {
        negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "Negative" : "NegPiP"
    }

    private var missingModelCount: Int {
        ModelComponent.allCases.reduce(into: 0) { result, component in
            if case .ready = catalog.state(for: component) { return }
            result += 1
        }
    }

    private func shareURL(for image: UIImage) -> URL? {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("anima-xs-\(UUID().uuidString).png")
        guard let data = image.pngData() else { return nil }
        do {
            try data.write(to: url)
            return url
        } catch {
            return nil
        }
    }

    private func stateLabel(_ state: ModelStore.State) -> String {
        switch state {
        case .missing: return "Missing"
        case .downloading: return "Downloading…"
        case .verifying: return "Verifying…"
        case .ready: return "Ready"
        case .failed: return "Needs attention"
        }
    }

    private func stateColor(_ state: ModelStore.State) -> Color {
        switch state {
        case .ready: return .green
        case .missing: return .orange
        case .downloading, .verifying: return Self.creamMuted
        case .failed: return .red
        }
    }
}

// MARK: - Model catalog

@MainActor
final class ModelCatalog: ObservableObject {
    @Published private var states: [ModelComponent: ModelStore.State] = [:]
    @Published private(set) var resolved: ResolvedModels?

    private let store: ModelStore?

    init() {
        store = try? ModelStore()
    }

    func state(for component: ModelComponent) -> ModelStore.State {
        states[component] ?? .missing
    }

    func refresh() async {
        guard let store else {
            states = ModelComponent.allCases.reduce(into: [:]) { $0[$1] = .failed("ModelStore unavailable") }
            resolved = nil
            return
        }
        for entry in ModelManifest.entries {
            states[entry.component] = await store.discover(entry)
        }
        resolved = (try? await store.resolveInstalledModels())
    }

    func download(_ component: ModelComponent) async {
        guard let store, let entry = ModelManifest.entries.first(where: { $0.component == component }) else { return }
        do {
            states[component] = .downloading
            let url = try await store.download(entry)
            states[component] = .ready(url)
        } catch {
            states[component] = .failed(error.localizedDescription)
        }
        try? await updateResolved()
    }

    func retry(_ component: ModelComponent) async {
        guard let store, let entry = ModelManifest.entries.first(where: { $0.component == component }) else { return }
        do {
            states[component] = .verifying
            let url = try await store.verifyExisting(entry)
            states[component] = .ready(url)
        } catch {
            states[component] = .failed(error.localizedDescription)
        }
        try? await updateResolved()
    }

    func repair(_ component: ModelComponent) async {
        guard let store, let entry = ModelManifest.entries.first(where: { $0.component == component }) else { return }
        do {
            states[component] = .downloading
            let url = try await store.repair(entry)
            states[component] = .ready(url)
        } catch {
            states[component] = .failed(error.localizedDescription)
        }
        try? await updateResolved()
    }

    func importPack(_ component: ModelComponent, from source: URL) async {
        guard let store, let entry = ModelManifest.entries.first(where: { $0.component == component }) else { return }
        do {
            states[component] = .verifying
            let url = try await store.importPack(entry, from: source)
            states[component] = .ready(url)
        } catch {
            states[component] = .failed(error.localizedDescription)
        }
        try? await updateResolved()
    }

    private func updateResolved() async throws {
        guard let store else { resolved = nil; return }
        resolved = try await store.resolveInstalledModels()
    }
}

// MARK: - User LoRA catalog

struct ImportedLoRA: Equatable, Sendable {
    let url: URL
    let displayName: String
    let moduleCount: Int
}

/// Owns one v1 external DiT LoRA. `adapters` intentionally presents it as a
/// collection so the generation UI already has the right visual/data shape for
/// future simultaneous LoRAs; runtime semantics remain exactly one active LoRA.
@MainActor
final class LoRACatalog: ObservableObject {
    @Published private(set) var selected: ImportedLoRA?
    @Published private(set) var isImporting = false
    @Published private(set) var errorMessage: String?

    var adapters: [ImportedLoRA] {
        selected.map { [$0] } ?? []
    }

    private static let displayNameKey = "generation.activeLoRADisplayName"
    private let directory: URL?

    init() {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask).first
        directory = base?.appendingPathComponent("LoRA", isDirectory: true)
    }

    func refresh() async {
        guard let destination = activeURL else {
            selected = nil
            return
        }
        let storedName = UserDefaults.standard.string(forKey: Self.displayNameKey)
            ?? "Imported LoRA"
        do {
            let moduleCount = try await Task.detached(priority: .utility) {
                guard FileManager.default.fileExists(atPath: destination.path) else { return nil as Int? }
                return try DiTLoRAFile(url: destination).modules.count
            }.value
            if let moduleCount {
                selected = ImportedLoRA(
                    url: destination, displayName: storedName,
                    moduleCount: moduleCount)
                errorMessage = nil
            } else {
                selected = nil
            }
        } catch {
            selected = nil
            errorMessage = "Stored LoRA is invalid: \(error.localizedDescription)"
        }
    }

    func importAdapter(from source: URL) async {
        guard !isImporting, let destination = activeURL,
              let directory else { return }
        isImporting = true
        errorMessage = nil
        defer { isImporting = false }
        let displayName = source.deletingPathExtension().lastPathComponent
        do {
            let moduleCount = try await Task.detached(priority: .userInitiated) {
                _ = try DiTLoRAFile(url: source)
                let fm = FileManager.default
                try fm.createDirectory(
                    at: directory, withIntermediateDirectories: true)
                let temporary = directory.appendingPathComponent(
                    "import-\(UUID().uuidString).safetensors")
                defer { try? fm.removeItem(at: temporary) }
                try fm.copyItem(at: source, to: temporary)
                let copied = try DiTLoRAFile(url: temporary)
                let count = copied.modules.count
                withExtendedLifetime(copied) {}
                if fm.fileExists(atPath: destination.path) {
                    _ = try fm.replaceItemAt(destination, withItemAt: temporary)
                } else {
                    try fm.moveItem(at: temporary, to: destination)
                }
                return count
            }.value
            UserDefaults.standard.set(displayName, forKey: Self.displayNameKey)
            selected = ImportedLoRA(
                url: destination, displayName: displayName,
                moduleCount: moduleCount)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func remove() {
        guard !isImporting else { return }
        if let destination = activeURL {
            do {
                if FileManager.default.fileExists(atPath: destination.path) {
                    try FileManager.default.removeItem(at: destination)
                }
                selected = nil
                errorMessage = nil
                UserDefaults.standard.removeObject(forKey: Self.displayNameKey)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private var activeURL: URL? {
        directory?.appendingPathComponent("active.safetensors")
    }
}

extension ModelComponent {
    var displayName: String {
        switch self {
        case .dit: return "Anima Turbo DiT"
        case .textEncoder: return "Qwen3 text encoder"
        case .vae: return "Qwen-Image VAE"
        }
    }

    static var allCases: [ModelComponent] {
        [.textEncoder, .dit, .vae]
    }
}
