import Foundation
import Observation
import SceneKit
import simd
import SwiftUI
import UIKit
import UserNotifications

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
    /// `true`: directo a pc-windows-server por la Wi-Fi. `false`: a través del servidor de la nube.
    let local: Bool
    /// Añade al refinado la fotogrametría completa de trabajador-gpu (30–60 min).
    var fullReconstruction = false
    var phase: Phase = .idle

    @ObservationIgnored private var job: Job?
    @ObservationIgnored private var work: Task<Void, Never>?

    private var jobURL: URL { folder.appendingPathComponent(local ? "pc_local.json" : "pc.json") }
    private var resultsFolder: URL { folder.appendingPathComponent(local ? "pc_local" : "pc", isDirectory: true) }
    var meshFile: URL { resultsFolder.appendingPathComponent("malla_alta.ply") }
    /// La misma habitación con los muebles reconocidos sustituidos por modelos limpios (si el PC pudo modelarlos).
    var modeledFile: URL? {
        let url = resultsFolder.appendingPathComponent("escena_modelada.ply")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Hacen falta las fotos con su pose (fotos clave y fotogramas) además de la malla LiDAR.
    static func isAvailable(for folder: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: folder.appendingPathComponent("malla_color.ply").path)
            && fm.fileExists(atPath: folder.appendingPathComponent("fotos_clave/camaras.json").path)
    }

    /// Modelos y plano devueltos por el refinado local, para compartirlos.
    var resultFiles: [URL] {
        ["modelo.obj", "modelo.glb", "modelo.usdz", "plano.dxf", "informe.json"]
            .map { resultsFolder.appendingPathComponent($0) }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }

    init(folder: URL, local: Bool = false) {
        self.folder = folder
        self.local = local
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
            let client = try local ? CloudClient.localPC() : CloudClient()
            if upload {
                phase = .uploading(0)
                if local {
                    _ = try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound])
                }
                let tipo = !local ? "malla_pc" : fullReconstruction ? "refinado_completo" : "refinado"
                let id = try await client.upload(folder: folder, splats: false, tipo: tipo) { [weak self] fraction in
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
            if local { notifyFinished() }
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

    private func notifyFinished() {
        let content = UNMutableNotificationContent()
        content.title = "Modelo mejorado listo"
        content.body = "Tu PC terminó de refinar \(folder.lastPathComponent)."
        content.sound = .default
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil))
    }
}

/// Sección "Malla de alta calidad (tu PC)" de la pantalla de un escaneo.
struct PCReconstructionSection: View {
    @State private var model: PCReconstruction
    @State private var showSettings = false

    init(folder: URL, local: Bool = false) {
        _model = State(initialValue: PCReconstruction(folder: folder, local: local))
    }

    var body: some View {
        Section {
            switch model.phase {
            case .idle:
                Button { model.start() } label: {
                    Label(model.local ? "Mejorar en PC (Wi-Fi)" : "Reconstruir en alta calidad en mi PC", systemImage: "desktopcomputer")
                        .font(.headline)
                        .foregroundStyle(.indigo)
                }
                if model.local {
                    Toggle("Fotogrametría completa (30–60 min)", isOn: $model.fullReconstruction)
                    Button("Dirección y token del PC") { showSettings = true }
                }
            case .uploading(let fraction):
                row("Subiendo escaneo… \(Int(fraction * 100)) %", value: fraction)
            case .processing(let fraction, let stage):
                row("\(stage) · \(Int(fraction * 100)) %", value: fraction)
                Text(model.local && !model.fullReconstruction
                     ? "Puedes salir y volver: el trabajo sigue en el PC."
                     : "Tu PC tarda de 30 a 60 minutos. Puedes salir y volver: el trabajo sigue en el PC.")
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
                if SplatSceneModel.file(in: model.folder) != nil {
                    NavigationLink {
                        SplatViewerView(folder: model.folder)
                    } label: {
                        Label("Ver el render fotorrealista", systemImage: "camera.aperture")
                            .font(.headline)
                            .foregroundStyle(.indigo)
                    }
                }
                if let modeled = model.modeledFile {
                    NavigationLink {
                        MeshPreviewView(url: modeled, title: "Habitación modelada")
                    } label: {
                        Label("Ver la habitación modelada", systemImage: "bed.double.fill")
                            .foregroundStyle(.indigo)
                    }
                }
                ShareLink(item: model.meshFile) {
                    Label("Compartir malla_alta.ply", systemImage: "square.and.arrow.up")
                }
                ForEach(model.resultFiles, id: \.self) { file in
                    ShareLink(item: file) {
                        Label("Compartir \(file.lastPathComponent)", systemImage: "square.and.arrow.up")
                    }
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
            Text(model.local ? "Mejorar en PC" : "Malla de alta calidad (tu PC)")
        } footer: {
            Text(model.local
                 ? "Envía el escaneo por la Wi-Fi a tu PC (pc-windows-server) y recibe OBJ, GLB, USDZ y plano DXF. El iPhone y el PC deben estar en la misma red."
                 : "Tu PC combina las fotos y el LiDAR (fotogrametría y Gaussian Splatting 2D) en una sola malla limpia, "
                    + "con paredes planas y cada mueble en su sitio. El PC tiene que estar encendido con el trabajador en marcha.")
        }
        .sheet(isPresented: $showSettings) { LocalPCSettingsView() }
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

/// Dirección y token del servidor del PC: los imprime pc-windows-server al arrancar.
struct LocalPCSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var server = LocalPCConfig.server
    @State private var token = LocalPCConfig.token ?? ""
    @State private var testResult: String?
    @State private var testing = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Dirección") {
                    TextField("http://192.168.1.20:8787", text: $server)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Section {
                    SecureField("Token", text: $token)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Token")
                } footer: {
                    Text("El PC los muestra al arrancar el servidor. El token se guarda en el llavero del iPhone.")
                }
                Section {
                    Button(testing ? "Probando…" : "Probar conexión") { Task { await test() } }
                        .disabled(testing || token.isEmpty || server.isEmpty)
                    if let testResult { Text(testResult).font(.callout) }
                }
            }
            .navigationTitle("Servidor del PC")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancelar") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Guardar") {
                        saveFields()
                        dismiss()
                    }
                    .disabled(token.isEmpty || server.isEmpty)
                }
            }
        }
    }

    private func saveFields() {
        LocalPCConfig.server = server
        LocalPCConfig.token = token
    }

    private func test() async {
        testing = true
        defer { testing = false }
        saveFields()
        do {
            try await CloudClient.localPC().checkConnection()
            testResult = "✅ Conectado a tu PC"
        } catch {
            testResult = "❌ \(error.localizedDescription)"
        }
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
        // Las caras miran hacia dentro de la habitación: desde fuera, las paredes cercanas no tapan (casa de muñecas).
        geometry.firstMaterial?.isDoubleSided = false
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
