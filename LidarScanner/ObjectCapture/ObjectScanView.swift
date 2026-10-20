import SwiftUI
import RealityKit

struct ObjectScanView: View {
    @State private var model = ObjectScanModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch model.phase {
            case .idle:
                ProgressView("Preparando cámara…")
                    .task { model.start() }

            case .capturing:
                if let session = model.session {
                    ObjectCaptureView(session: session)
                        .ignoresSafeArea()
                    CaptureOverlay(session: session, onAreaCapture: model.startAreaCapture)
                }

            case .reconstructing(let fraction):
                VStack(spacing: 16) {
                    ProgressView(value: fraction)
                        .progressViewStyle(.linear)
                        .frame(maxWidth: 260)
                    Text("Reconstruyendo modelo 3D… \(Int(fraction * 100)) %")
                        .font(.headline)
                    Text("Mantén la app abierta. Puede tardar varios minutos.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .padding()

            case .done(let url):
                ExportedFilesView(title: "Modelo listo", files: [url])

            case .failed(let message):
                ContentUnavailableView("No se pudo completar",
                                       systemImage: "exclamationmark.triangle",
                                       description: Text(message))
            }
        }
        .navigationTitle("Objeto")
        .navigationBarTitleDisplayMode(.inline)
        .onDisappear { model.cancel() }
    }
}

/// Controles sobre `ObjectCaptureView` según el estado de la sesión.
private struct CaptureOverlay: View {
    let session: ObjectCaptureSession
    let onAreaCapture: () -> Void

    var body: some View {
        VStack {
            if let hint = session.feedback.compactMap({ ObjectScanModel.message(for: $0) }).first {
                Text(hint)
                    .font(.callout.weight(.semibold))
                    .padding(.horizontal, 14).padding(.vertical, 8)
                    .background(.yellow.opacity(0.9), in: Capsule())
                    .foregroundStyle(.black)
                    .padding(.top, 8)
            }

            Spacer()

            if session.userCompletedScanPass {
                scanPassPanel
            } else {
                controls
            }
        }
        .padding()
    }

    @ViewBuilder
    private var controls: some View {
        switch session.state {
        case .ready:
            instruction("Objeto: apúntalo sobre una superficie despejada · Área: un rincón o zona completa")
            PrimaryButton("Detectar objeto", systemImage: "viewfinder") { _ = session.startDetecting() }
            SecondaryButton("Capturar área (sin caja)", systemImage: "square.dashed") { onAreaCapture() }

        case .detecting:
            instruction("Ajusta la caja para que envuelva el objeto")
            HStack {
                SecondaryButton("Reiniciar", systemImage: "arrow.counterclockwise") { session.resetDetection() }
                PrimaryButton("Iniciar captura", systemImage: "camera.fill") { session.startCapturing() }
            }

        case .capturing:
            instruction("Rodea el objeto despacio · \(session.numberOfShotsTaken) fotos")
            SecondaryButton("Terminar ahora", systemImage: "checkmark") { session.finish() }

        case .finishing:
            ProgressView("Guardando fotos…")
                .padding()
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))

        default:
            EmptyView()
        }
    }

    private var scanPassPanel: some View {
        VStack(spacing: 12) {
            Text("Vuelta completada")
                .font(.headline)
            Text("Para un modelo más completo haz otra vuelta desde otra altura, o voltea el objeto para capturar la base.")
                .font(.footnote)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            HStack {
                SecondaryButton("Otra altura", systemImage: "arrow.up.and.down") { session.beginNewScanPass() }
                SecondaryButton("Voltear", systemImage: "arrow.triangle.2.circlepath") { session.beginNewScanPassAfterFlip() }
            }
            PrimaryButton("Terminar y crear modelo", systemImage: "cube.fill") { session.finish() }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
    }

    private func instruction(_ text: String) -> some View {
        Text(text)
            .font(.subheadline)
            .padding(.horizontal, 14).padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
    }
}
