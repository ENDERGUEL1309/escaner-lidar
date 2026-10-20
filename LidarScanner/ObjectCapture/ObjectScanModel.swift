import Foundation
import Observation
import RealityKit
import SwiftUI   // ObjectCaptureSession está en la superposición RealityKit + SwiftUI

/// Flujo de escaneo de objetos con Object Capture (RealityKit):
/// 1. `ObjectCaptureSession` guía la captura (usa LiDAR para delimitar el objeto y elegir las fotos).
/// 2. `PhotogrammetrySession` reconstruye en el propio iPhone un modelo USDZ con textura PBR.
@MainActor
@Observable
final class ObjectScanModel {
    enum Phase {
        case idle
        case capturing
        case reconstructing(Double)
        case done(URL)
        case failed(String)
    }

    var phase: Phase = .idle
    var session: ObjectCaptureSession?
    /// Modo área: captura una zona (rincón, mesa con objetos, fachada…) sin caja delimitadora.
    var isAreaMode = false

    @ObservationIgnored private var folder: URL?
    @ObservationIgnored private var stateTask: Task<Void, Never>?
    @ObservationIgnored private var reconstructionTask: Task<Void, Never>?

    static var isSupported: Bool { ObjectCaptureSession.isSupported }

    private var imagesFolder: URL? { folder?.appendingPathComponent("Fotos", isDirectory: true) }
    private var checkpointFolder: URL? { folder?.appendingPathComponent("Checkpoints", isDirectory: true) }

    func start() {
        do {
            let folder = try ScanStorage.newScanFolder(prefix: "Objeto")
            self.folder = folder
            let images = folder.appendingPathComponent("Fotos", isDirectory: true)
            let checkpoints = folder.appendingPathComponent("Checkpoints", isDirectory: true)
            try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: checkpoints, withIntermediateDirectories: true)

            var config = ObjectCaptureSession.Configuration()
            config.checkpointDirectory = checkpoints

            let session = ObjectCaptureSession()
            session.start(imagesDirectory: images, configuration: config)
            self.session = session
            phase = .capturing

            stateTask = Task { [weak self] in
                for await state in session.stateUpdates {
                    guard let self else { return }
                    switch state {
                    case .completed:
                        self.startReconstruction()
                        return
                    case .failed(let error):
                        self.session = nil
                        self.phase = .failed(error.localizedDescription)
                        return
                    default:
                        break
                    }
                }
            }
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func startAreaCapture() {
        isAreaMode = true
        session?.startCapturing()
    }

    func cancel() {
        stateTask?.cancel()
        reconstructionTask?.cancel()
        session?.cancel()
        session = nil
    }

    private func startReconstruction() {
        // Liberar la sesión de captura antes de reconstruir: ambas consumen mucha memoria.
        session = nil
        guard let folder, let images = imagesFolder, let checkpoints = checkpointFolder else { return }
        phase = .reconstructing(0)

        let maskObject = !isAreaMode
        reconstructionTask = Task { [weak self] in
            let output = folder.appendingPathComponent("modelo.usdz")
            do {
                guard PhotogrammetrySession.isSupported else {
                    self?.phase = .failed("Este dispositivo no admite la reconstrucción en el propio iPhone.")
                    return
                }
                var config = PhotogrammetrySession.Configuration()
                config.checkpointDirectory = checkpoints
                // En modo área no hay un objeto que recortar del fondo: se reconstruye todo lo fotografiado.
                config.isObjectMaskingEnabled = maskObject
                let photogrammetry = try PhotogrammetrySession(input: images, configuration: config)
                try photogrammetry.process(requests: [.modelFile(url: output)])

                for try await message in photogrammetry.outputs {
                    guard let self else { return }
                    switch message {
                    case .requestProgress(_, let fraction):
                        self.phase = .reconstructing(fraction)
                    case .requestComplete(_, let result):
                        if case .modelFile(let url) = result { self.phase = .done(url) }
                    case .requestError(_, let error):
                        self.phase = .failed(error.localizedDescription)
                    case .processingCancelled:
                        self.phase = .failed("Reconstrucción cancelada.")
                    default:
                        break
                    }
                }
            } catch {
                self?.phase = .failed(error.localizedDescription)
            }
        }
    }

    static func message(for feedback: ObjectCaptureSession.Feedback) -> String? {
        switch feedback {
        case .objectTooClose: return "Aléjate un poco del objeto"
        case .objectTooFar: return "Acércate al objeto"
        case .movingTooFast: return "Muévete más despacio"
        case .environmentLowLight: return "Poca luz: mejora la iluminación"
        case .environmentTooDark: return "Demasiado oscuro para escanear"
        case .outOfFieldOfView: return "Mantén el objeto dentro del encuadre"
        case .objectNotFlippable: return "No conviene voltear este objeto"
        case .overCapturing: return "Ya hay suficientes fotos de esta zona"
        case .objectNotDetected: return "No se detecta el objeto"
        default: return nil
        }
    }
}
