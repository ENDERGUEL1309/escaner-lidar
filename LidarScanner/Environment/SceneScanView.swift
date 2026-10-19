import SwiftUI
import RealityKit
import ARKit

struct SceneScanView: View {
    @State private var scanner = SceneScanner()

    var body: some View {
        ZStack {
            ARScanContainer(scanner: scanner)
                .ignoresSafeArea()

            VStack(spacing: 12) {
                statsBar
                if let hint = scanner.trackingHint {
                    Text(hint)
                        .font(.callout.weight(.semibold))
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(.yellow.opacity(0.9), in: Capsule())
                        .foregroundStyle(.black)
                }
                if scanner.isVolumeFull {
                    Text("Memoria del escaneo llena: finaliza o usa una calidad menor")
                        .font(.callout.weight(.semibold))
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(.orange.opacity(0.9), in: Capsule())
                        .foregroundStyle(.black)
                }
                Spacer()
                controls
            }
            .padding()

            if scanner.phase == .processing {
                processingOverlay
            }
        }
        .navigationTitle("Entorno")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { scanner.stop() }
        .sheet(isPresented: finishedBinding) {
            NavigationStack {
                ExportedFilesView(title: "Escaneo guardado", files: scanner.exportedFiles, objects: scanner.detectedObjects)
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Nuevo escaneo") { scanner.prepareNewScan() }
                        }
                    }
            }
        }
        .alert("Error", isPresented: failedBinding) {
            Button("Aceptar") { scanner.prepareNewScan() }
        } message: {
            if case .failed(let message) = scanner.phase { Text(message) }
        }
    }

    private var finishedBinding: Binding<Bool> {
        Binding(get: { scanner.phase == .finished },
                set: { if !$0 { scanner.prepareNewScan() } })
    }

    private var failedBinding: Binding<Bool> {
        Binding(get: { if case .failed = scanner.phase { return true } else { return false } },
                set: { _ in })
    }

    private var statsBar: some View {
        HStack(spacing: 16) {
            stat("Vóxeles", scanner.voxelCount.formatted(.number.notation(.compactName)), "cube.fill")
            stat("Malla", "\(scanner.meshAnchorCount)", "square.grid.3x3.square")
            stat("Vistas", "\(scanner.keyframeCount)", "camera.fill")
        }
        .padding(.horizontal, 16).padding(.vertical, 10)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private func stat(_ label: String, _ value: String, _ icon: String) -> some View {
        VStack(spacing: 2) {
            Label(value, systemImage: icon).font(.subheadline.monospacedDigit().bold())
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var controls: some View {
        VStack(spacing: 12) {
            switch scanner.phase {
            case .ready:
                Picker("Calidad", selection: $scanner.settings.quality) {
                    ForEach(ScanSettings.Quality.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                VStack(spacing: 4) {
                    Toggle("Rellenar huecos", isOn: $scanner.settings.fillHoles)
                    Toggle("Detectar objetos", isOn: $scanner.settings.detectObjects)
                }
                .font(.subheadline)
                .tint(.cyan)
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                Text("Recorre la zona despacio, a 0,5–3 m, apuntando a todas las superficies.")
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .padding(8)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
                PrimaryButton("Iniciar escaneo", systemImage: "record.circle") { scanner.start() }

            case .scanning:
                HStack {
                    SecondaryButton("Pausar", systemImage: "pause.fill") { scanner.pause() }
                    PrimaryButton("Finalizar", systemImage: "checkmark") { scanner.finish() }
                }

            case .paused:
                HStack {
                    SecondaryButton("Reanudar", systemImage: "play.fill") { scanner.resume() }
                    PrimaryButton("Finalizar", systemImage: "checkmark") { scanner.finish() }
                }

            default:
                EmptyView()
            }
        }
    }

    private var processingOverlay: some View {
        ZStack {
            Color.black.opacity(0.7).ignoresSafeArea()
            VStack(spacing: 14) {
                ProgressView(value: scanner.progress)
                    .frame(maxWidth: 260)
                Text(scanner.processingStep).font(.headline)
                Text("\(Int(scanner.progress * 100)) %").font(.subheadline.monospacedDigit()).foregroundStyle(.secondary)
            }
            .padding(24)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
        }
    }
}

/// `ARView` de RealityKit mostrando la cámara con la malla LiDAR superpuesta mientras se construye.
private struct ARScanContainer: UIViewRepresentable {
    let scanner: SceneScanner

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        view.renderOptions.insert(.disableMotionBlur)
        view.debugOptions.insert(.showSceneUnderstanding)
        scanner.attach(view.session)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    static func dismantleUIView(_ uiView: ARView, coordinator: ()) {
        uiView.session.pause()
    }
}
