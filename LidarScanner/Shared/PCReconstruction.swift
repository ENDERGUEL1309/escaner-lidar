import Foundation
import Observation
import SceneKit
import simd
import SwiftUI
import UIKit

/// "Malla de alta calidad en mi PC": sube el escaneo al servidor marcado como `malla_pc`; el trabajador del PC
/// (trabajador-gpu/trabajador_pc.py) lo toma, lo reconstruye con fotogrametría + Gaussian Splatting 2D guiado por
/// el LiDAR, y deja la malla en el servidor. Aquí se descarga a `<carpeta>/pc/malla_alta.ply` y se borra del servidor.
/// El estado se guarda en `<carpeta>/pc.json` para poder salir de la pantalla y volver.
@MainActor
@Observable
final class PCReconstruction {
    enum Phase: Equatable {
        case idle
        case uploading(Double)
        case processing(Double, String)
        case downloading
        case ready
        case failed(String)
    }

    private struct Job: Codable {
        var id: String
        var descargado = false
    }

    let folder: URL
    var phase: Phase = .idle

    @ObservationIgnored private var job: Job?
    @ObservationIgnored private var work: Task<Void, Never>?

    private var jobURL: URL { folder.appendingPathComponent("pc.json") }
    private var resultsFolder: URL { folder.appendingPathComponent("pc", isDirectory: true) }
    var meshFile: URL { resultsFolder.appendingPathComponent("malla_alta.ply") }

    /// Hacen falta las fotos con su pose (fotos clave y fotogramas) además de la malla LiDAR.
    static func isAvailable(for folder: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: folder.appendingPathComponent("malla_color.ply").path)
            && fm.fileExists(atPath: folder.appendingPathComponent("fotos_clave/camaras.json").path)
    }

    init(folder: URL) {
        self.folder = folder
        if let data = try? Data(contentsOf: jobURL), let saved = try? JSONDecoder().decode(Job.self, from: data) {
            job = saved
            phase = saved.descargado ? .ready : .processing(0, "Consultando…")
        }
    }

    func start() {
        guard work == nil else { return }
        work = Task { await run(upload: true) }
    }

    func resume() {
        guard work == nil, let job, !job.descargado else { return }
        work = Task { await run(upload: false) }
    }

    func cancelMonitoring() {
        if case .uploading = phase { return }
        work?.cancel()
        work = nil
    }

    /// Borra la malla descargada para poder pedir otra.
    func discard() {
        try? FileManager.default.removeItem(at: resultsFolder)
        try? FileManager.default.removeItem(at: jobURL)
        job = nil
        phase = .idle
    }

    private func run(upload: Bool) async {
        let background = UIApplication.shared.beginBackgroundTask(withName: "Subida para el PC")
        defer {
            UIApplication.shared.endBackgroundTask(background)
            work = nil
        }
        do {
            let client = try CloudClient()
            if upload {
                phase = .uploading(0)
                let id = try await client.upload(folder: folder, splats: false, tipo: "malla_pc") { [weak self] fraction in
                    Task { @MainActor in self?.phase = .uploading(fraction) }
                }
                job = Job(id: id)
                save()
            }
            guard let id = job?.id else { phase = .idle; return }

            // El PC tarda de 30 a 60 minutos: se consulta cada 10 s mientras la pantalla esté abierta.
            var scan = try await client.status(id)
            while scan.isRunning {
                phase = .processing(scan.progreso ?? 0, scan.etapa ?? "Esperando a tu PC")
                try await Task.sleep(for: .seconds(10))
                scan = try await client.status(id)
            }
            guard scan.estado == "listo" else {
                try? await client.delete(id)
                throw CloudClient.CloudError.server(scan.error ?? "El procesado terminó en «\(scan.estado)»")
            }
            phase = .downloading
            try await client.download(scan, to: resultsFolder)
            job?.descargado = true
            save()
            try? await client.delete(id)
            phase = .ready
        } catch is CancellationError {
            // Se salió de la pantalla: el trabajo sigue en el servidor y en el PC.
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func save() {
        guard let job, let data = try? JSONEncoder().encode(job) else { return }
        try? data.write(to: jobURL, options: .atomic)
    }
}

/// Sección "Malla de alta calidad (tu PC)" de la pantalla de un escaneo.
struct PCReconstructionSection: View {
    @State private var model: PCReconstruction

    init(folder: URL) {
        _model = State(initialValue: PCReconstruction(folder: folder))
    }

    var body: some View {
        Section {
            switch model.phase {
            case .idle:
                Button { model.start() } label: {
                    Label("Reconstruir en alta calidad en mi PC", systemImage: "desktopcomputer")
                        .font(.headline)
                        .foregroundStyle(.indigo)
                }
            case .uploading(let fraction):
                row("Subiendo escaneo… \(Int(fraction * 100)) %", value: fraction)
            case .processing(let fraction, let stage):
                row("\(stage) · \(Int(fraction * 100)) %", value: fraction)
                Text("Tu PC tarda de 30 a 60 minutos. Puedes salir y volver: el trabajo sigue en el PC.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            case .downloading:
                row("Descargando la malla…", value: nil)
            case .ready:
                NavigationLink {
                    MeshPreviewView(url: model.meshFile, title: "Alta calidad")
                } label: {
                    Label("Ver la malla de alta calidad", systemImage: "cube.fill")
                        .font(.headline)
                        .foregroundStyle(.indigo)
                }
                ShareLink(item: model.meshFile) {
                    Label("Compartir malla_alta.ply", systemImage: "square.and.arrow.up")
                }
                Button(role: .destructive) { model.discard() } label: {
                    Label("Descartar y volver a pedirla", systemImage: "trash")
                }
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .font(.callout)
                Button("Reintentar") { model.start() }
            }
        } header: {
            Text("Malla de alta calidad (tu PC)")
        } footer: {
            Text("Tu PC combina las fotos y el LiDAR (fotogrametría y Gaussian Splatting 2D) en una sola malla limpia, "
                 + "con paredes planas y cada mueble en su sitio. El PC tiene que estar encendido con el trabajador en marcha.")
        }
        .onAppear { model.resume() }
        .onDisappear { model.cancelMonitoring() }
    }

    private func row(_ title: String, value: Double?) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline)
            if let value { ProgressView(value: value) } else { ProgressView() }
        }
        .padding(.vertical, 4)
    }
}

/// Visor sencillo de una malla PLY con color por vértice (el color ya trae la luz real, así que no se sombrea).
struct MeshPreviewView: View {
    let url: URL
    let title: String
    @State private var scene: SCNScene?
    @State private var failed = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let scene {
                SceneView(scene: scene, options: [.allowsCameraControl, .autoenablesDefaultLighting])
                    .ignoresSafeArea()
            } else if failed {
                ContentUnavailableView("No se pudo abrir la malla", systemImage: "exclamationmark.triangle")
            } else {
                ProgressView("Cargando malla…").tint(.white).foregroundStyle(.white)
            }
        }
        .navigationTitle(title)
        .navigationBarTitleDisplayMode(.inline)
        .task { await load() }
    }

    private func load() async {
        guard scene == nil else { return }
        let url = url
        let mesh = await Task.detached(priority: .userInitiated) { try? PLYReader.readMesh(from: url) }.value
        guard let mesh, !mesh.isEmpty else { failed = true; return }

        let geometry = ModelExporter.makeGeometry(mesh)
        geometry.firstMaterial?.lightingModel = .constant
        let node = SCNNode(geometry: geometry)
        let scene = SCNScene()
        scene.rootNode.addChildNode(node)

        // Cámara por encima, en diagonal, mirando al centro.
        let (minP, maxP) = node.boundingBox
        let center = SCNVector3((minP.x + maxP.x) / 2, (minP.y + maxP.y) / 2, (minP.z + maxP.z) / 2)
        let radius = max(simd_length(SIMD3(maxP.x - minP.x, maxP.y - minP.y, maxP.z - minP.z)) / 2, 0.5)
        let camera = SCNNode()
        camera.camera = SCNCamera()
        camera.camera?.zNear = 0.01
        camera.camera?.zFar = Double(radius * 20)
        camera.position = SCNVector3(center.x + radius * 0.9, center.y + radius * 1.3, center.z + radius * 0.9)
        camera.look(at: center)
        scene.rootNode.addChildNode(camera)
        self.scene = scene
    }
}
