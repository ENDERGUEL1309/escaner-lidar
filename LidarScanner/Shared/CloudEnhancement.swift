import Foundation
import Observation
import SwiftUI
import UIKit

/// "Mejorar con IA en la nube" para un escaneo: sube la carpeta al servidor, espera a la GPU, descarga el
/// resultado en `<carpeta>/ia/` y, si el usuario lo acepta, sustituye los objetos del iPhone por los de la nube
/// (guardando los originales en `<carpeta>/original/` para poder volver).
/// El estado se guarda en `<carpeta>/nube.json`, así se puede salir de la pantalla y volver más tarde.
@MainActor
@Observable
final class CloudEnhancement {
    enum Phase: Equatable {
        case idle
        case uploading(Double)
        case processing(Double, String)
        case downloading
        case ready
        case applied
        case failed(String)
    }

    private struct Job: Codable {
        var id: String
        var servidor: String
        var descargado = false
        var aplicado = false
    }

    let folder: URL
    var phase: Phase = .idle

    @ObservationIgnored private var job: Job?
    @ObservationIgnored private var work: Task<Void, Never>?

    private var jobURL: URL { folder.appendingPathComponent("nube.json") }
    private var resultsFolder: URL { folder.appendingPathComponent("ia", isDirectory: true) }
    private var backupFolder: URL { folder.appendingPathComponent("original", isDirectory: true) }

    /// Solo los escaneos de entorno con malla y objetos se pueden mejorar.
    static func isAvailable(for folder: URL) -> Bool {
        FileManager.default.fileExists(atPath: folder.appendingPathComponent("malla_color.ply").path)
            && FileManager.default.fileExists(atPath: folder.appendingPathComponent("escena.json").path)
    }

    var hasPhotos: Bool {
        FileManager.default.fileExists(atPath: folder.appendingPathComponent("fotos_clave/camaras.json").path)
    }

    /// Hay fotogramas para la vista fotorrealista (Gaussian Splatting).
    var hasSplatFrames: Bool {
        FileManager.default.fileExists(atPath: folder.appendingPathComponent("fotogramas/camaras.json").path)
    }

    /// Pedir también la vista fotorrealista (≈10–15 min más de GPU).
    var includeSplats = true

    /// La vista fotorrealista ya descargada, si la hay.
    var splatFile: URL? { SplatViewerView.file(in: folder) }

    var uploadSizeText: String {
        let bytes = CloudClient.filesToUpload(in: folder).reduce(Int64(0)) { total, path in
            total + Int64((try? folder.appendingPathComponent(path).resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    init(folder: URL) {
        self.folder = folder
        if let data = try? Data(contentsOf: jobURL), let saved = try? JSONDecoder().decode(Job.self, from: data) {
            job = saved
            phase = saved.aplicado ? .applied : (saved.descargado ? .ready : .processing(0, "Consultando…"))
        }
    }

    // MARK: - Acciones

    func start() {
        guard work == nil else { return }
        work = Task { await run(upload: true) }
    }

    /// Reanuda el seguimiento de un trabajo en marcha (al volver a la pantalla).
    func resume() {
        guard work == nil, let job, !job.descargado else { return }
        work = Task { await run(upload: false) }
    }

    /// Deja de consultar el estado al salir de la pantalla. Una subida en curso no se corta.
    func cancelMonitoring() {
        if case .uploading = phase { return }
        work?.cancel()
        work = nil
    }

    /// Sustituye escena.json y objetos/ por los de la nube; los del iPhone van a original/.
    func apply() {
        let fm = FileManager.default
        do {
            if !fm.fileExists(atPath: backupFolder.path) {
                try fm.createDirectory(at: backupFolder, withIntermediateDirectories: true)
                for name in ["escena.json", "objetos", "edicion.json"] {
                    let source = folder.appendingPathComponent(name)
                    if fm.fileExists(atPath: source.path) { try fm.moveItem(at: source, to: backupFolder.appendingPathComponent(name)) }
                }
            } else {
                // Ya hay copia de los originales: solo se quitan los actuales.
                for name in ["escena.json", "objetos", "edicion.json"] {
                    try? fm.removeItem(at: folder.appendingPathComponent(name))
                }
            }
            for name in ["escena.json", "objetos"] {
                let source = resultsFolder.appendingPathComponent(name)
                if fm.fileExists(atPath: source.path) { try fm.copyItem(at: source, to: folder.appendingPathComponent(name)) }
            }
            job?.aplicado = true
            save()
            phase = .applied
        } catch {
            phase = .failed("No se pudieron aplicar los resultados: \(error.localizedDescription)")
        }
    }

    /// Vuelve a los objetos detectados por el iPhone.
    func restoreOriginal() {
        let fm = FileManager.default
        guard fm.fileExists(atPath: backupFolder.path) else { return }
        for name in ["escena.json", "objetos", "edicion.json"] {
            try? fm.removeItem(at: folder.appendingPathComponent(name))
            let saved = backupFolder.appendingPathComponent(name)
            if fm.fileExists(atPath: saved.path) { try? fm.copyItem(at: saved, to: folder.appendingPathComponent(name)) }
        }
        job?.aplicado = false
        save()
        phase = .ready
    }

    // MARK: - Proceso

    private func run(upload: Bool) async {
        // Si la app pasa a segundo plano, iOS le da unos minutos más para terminar la subida.
        let background = UIApplication.shared.beginBackgroundTask(withName: "Subida a la nube")
        defer {
            UIApplication.shared.endBackgroundTask(background)
            work = nil
        }

        do {
            let client = try CloudClient()
            if upload {
                phase = .uploading(0)
                let id = try await client.upload(folder: folder, splats: includeSplats && hasSplatFrames) { [weak self] fraction in
                    Task { @MainActor in self?.phase = .uploading(fraction) }
                }
                job = Job(id: id, servidor: CloudConfig.server)
                save()
            }
            guard let id = job?.id else { phase = .idle; return }

            // La GPU tarda unos minutos: se consulta cada 5 s mientras la pantalla esté abierta.
            var scan = try await client.status(id)
            while scan.isRunning {
                phase = .processing(scan.progreso ?? 0, scan.etapa ?? "En cola")
                try await Task.sleep(for: .seconds(5))
                scan = try await client.status(id)
            }
            guard scan.estado == "listo" else {
                try? await client.delete(id)   // un trabajo fallido tampoco debe ocupar el disco del servidor
                throw CloudClient.CloudError.server(scan.error ?? "El procesado terminó en «\(scan.estado)»")
            }

            phase = .downloading
            try await client.download(scan, to: resultsFolder)
            job?.descargado = true
            save()
            // Ya está todo en el iPhone: se borra del servidor para dejar su disco libre para el siguiente escaneo.
            try? await client.delete(id)
            phase = .ready
        } catch is CancellationError {
            // Se salió de la pantalla: el trabajo sigue en el servidor y se retoma al volver.
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func save() {
        guard let job, let data = try? JSONEncoder().encode(job) else { return }
        try? data.write(to: jobURL, options: .atomic)
    }
}

/// Sección "IA en la nube" de la pantalla de un escaneo.
struct CloudEnhancementSection: View {
    @State private var model: CloudEnhancement
    @State private var showSettings = false
    @State private var configured = CloudConfig.isConfigured

    init(folder: URL) {
        _model = State(initialValue: CloudEnhancement(folder: folder))
    }

    var body: some View {
        Section {
            content
        } header: {
            Text("IA en la nube")
        } footer: {
            Text(footer)
        }
        .sheet(isPresented: $showSettings, onDismiss: { configured = CloudConfig.isConfigured }) {
            CloudSettingsView()
        }
        .onAppear { model.resume() }
        .onDisappear { model.cancelMonitoring() }
    }

    @ViewBuilder
    private var content: some View {
        if !configured {
            Button { showSettings = true } label: {
                Label("Configurar servidor de IA", systemImage: "gearshape")
            }
        } else {
            switch model.phase {
            case .idle:
                Button { model.start() } label: {
                    Label("Mejorar con IA en la nube", systemImage: "sparkles")
                        .font(.headline)
                        .foregroundStyle(.purple)
                }
                if model.hasSplatFrames {
                    Toggle("Incluir vista fotorrealista (+10–15 min)", isOn: $model.includeSplats)
                        .font(.subheadline)
                        .tint(.purple)
                }
                Text("Se subirán \(model.uploadSizeText)\(model.hasPhotos ? " (con fotos de 12 MP)" : " (sin fotos: detección más limitada)").")
                    .font(.caption)
                    .foregroundStyle(.secondary)

            case .uploading(let fraction):
                progressRow("Subiendo escaneo… \(Int(fraction * 100)) %", value: fraction)

            case .processing(let fraction, let stage):
                progressRow("\(stage) · \(Int(fraction * 100)) %", value: fraction)
                Text("La GPU tarda unos minutos. Puedes salir y volver: el trabajo sigue en el servidor.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

            case .downloading:
                progressRow("Descargando resultados…", value: nil)

            case .ready:
                Label("Resultados listos", systemImage: "checkmark.seal.fill")
                    .foregroundStyle(.green)
                Button { model.apply() } label: {
                    Label("Usar los objetos de la IA en la nube", systemImage: "arrow.triangle.2.circlepath")
                }
                splatLink

            case .applied:
                Label("Usando los objetos de la IA en la nube", systemImage: "sparkles")
                    .foregroundStyle(.purple)
                splatLink
                Button(role: .destructive) { model.restoreOriginal() } label: {
                    Label("Volver a los objetos del iPhone", systemImage: "arrow.uturn.backward")
                }

            case .failed(let message):
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .font(.callout)
                Button("Reintentar") { model.start() }
            }
        }
    }

    @ViewBuilder
    private var splatLink: some View {
        if let file = model.splatFile {
            NavigationLink {
                SplatViewerView(file: file)
            } label: {
                Label("Vista fotorrealista", systemImage: "camera.filters")
                    .foregroundStyle(.purple)
            }
        }
    }

    private var footer: String {
        switch model.phase {
        case .ready, .applied:
            "Abre «Ver en 3D a color» para ver la escena con los objetos elegidos. Siempre puedes volver a los originales."
        default:
            "Una GPU en la nube reconoce y separa los objetos con modelos de IA más grandes que los del iPhone."
        }
    }

    private func progressRow(_ title: String, value: Double?) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline)
            if let value { ProgressView(value: value) } else { ProgressView() }
        }
        .padding(.vertical, 4)
    }
}

/// Dirección del servidor y token de acceso (el token está en servidor/.api-token del PC).
struct CloudSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var server = CloudConfig.server
    @State private var token = CloudConfig.token ?? ""
    @State private var testResult: String?
    @State private var testing = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("https://…", text: $server)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Servidor")
                }
                Section {
                    SecureField("Token de acceso", text: $token)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                } header: {
                    Text("Token")
                } footer: {
                    Text("Se guarda en el llavero del iPhone. Lo encontrarás en el PC, en servidor/.api-token.")
                }
                Section {
                    Button(testing ? "Probando…" : "Probar conexión") { Task { await test() } }
                        .disabled(testing || token.isEmpty)
                    if let testResult {
                        Text(testResult).font(.callout)
                    }
                }
            }
            .navigationTitle("Ajustes de la nube")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancelar") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Guardar") {
                        saveFields()
                        dismiss()
                    }
                    .disabled(token.isEmpty)
                }
            }
        }
    }

    private func saveFields() {
        CloudConfig.server = server
        CloudConfig.token = token
    }

    private func test() async {
        testing = true
        defer { testing = false }
        saveFields()
        do {
            try await CloudClient().checkConnection()
            testResult = "✅ Conectado al servidor"
        } catch {
            testResult = "❌ \(error.localizedDescription)"
        }
    }
}
