import CryptoKit
import Foundation
import Security

/// Configuración del servidor de IA en la nube (ver servidor/README.md y docs/ia-nube.md).
/// La dirección se guarda en `UserDefaults`; el token de acceso, en el llavero del iPhone.
enum CloudConfig {
    static let defaultServer = "https://servidor-production-dbef.up.railway.app"
    private static let serverKey = "nube.servidor"
    private static let tokenAccount = "nube.token"

    static var server: String {
        get { UserDefaults.standard.string(forKey: serverKey) ?? defaultServer }
        set { UserDefaults.standard.set(newValue.trimmingCharacters(in: .whitespacesAndNewlines), forKey: serverKey) }
    }

    static var token: String? {
        get { Keychain.read(tokenAccount) }
        set {
            if let value = newValue?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty {
                Keychain.write(value, account: tokenAccount)
            } else {
                Keychain.delete(tokenAccount)
            }
        }
    }

    static var isConfigured: Bool { URL(string: server) != nil && token != nil }
}

private enum Keychain {
    private static let service = "EscanerLidar"

    static func read(_ account: String) -> String? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service,
                                    kSecAttrAccount as String: account,
                                    kSecReturnData as String: true]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func write(_ value: String, account: String) {
        delete(account)
        let item: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                   kSecAttrService as String: service,
                                   kSecAttrAccount as String: account,
                                   kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock,
                                   kSecValueData as String: Data(value.utf8)]
        SecItemAdd(item as CFDictionary, nil)
    }

    static func delete(_ account: String) {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                    kSecAttrService as String: service,
                                    kSecAttrAccount as String: account]
        SecItemDelete(query as CFDictionary)
    }
}

/// Cliente de la API del servidor: crear escaneo → subir archivos → procesar → estado → descargar resultados.
struct CloudClient {
    struct RemoteScan: Decodable {
        struct ResultFile: Decodable {
            let ruta: String
            let url: String
        }
        let id: String
        let estado: String
        let progreso: Double?
        let etapa: String?
        let error: String?
        let resultados: [ResultFile]

        var isRunning: Bool { estado == "en_cola" || estado == "procesando" }
    }

    private struct Created: Decodable {
        struct Upload: Decodable {
            let ruta: String
            let metodo: String
            let url: String
        }
        let id: String
        let subidas: [Upload]
    }

    private struct ServerError: Decodable { let error: String }

    enum CloudError: LocalizedError {
        case notConfigured
        case server(String)
        case nothingToUpload

        var errorDescription: String? {
            switch self {
            case .notConfigured: "Configura la dirección del servidor y el token en Ajustes de la nube."
            case .server(let message): message
            case .nothingToUpload: "Este escaneo no tiene malla_color.ply ni escena.json: hazlo con «Detectar objetos» activado."
            }
        }
    }

    let base: URL
    let token: String

    init() throws {
        guard let base = URL(string: CloudConfig.server), let token = CloudConfig.token else { throw CloudError.notConfigured }
        self.base = base
        self.token = token
    }

    // MARK: - API

    /// Comprueba que el servidor responde y que el token es válido.
    func checkConnection() async throws {
        _ = try await request("GET", "/api/escaneos")
    }

    /// Sube la carpeta del escaneo y lanza el procesado. Devuelve el identificador del trabajo.
    func upload(folder: URL, splats: Bool, progress: @escaping (Double) -> Void) async throws -> String {
        let files = Self.filesToUpload(in: folder)
        guard files.contains(where: { $0 == "malla_color.ply" || $0 == "escena.json" }) else { throw CloudError.nothingToUpload }

        var entries: [[String: Any]] = []
        var total: Int64 = 0
        for path in files {
            let url = folder.appendingPathComponent(path)
            let size = Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            entries.append(["ruta": path, "bytes": size, "sha256": try Self.sha256(of: url)])
            total += size
        }
        let manifest: [String: Any] = [
            "version": 1,
            "nombre": folder.lastPathComponent,
            "opciones": ["segmentacion": true, "splats": splats, "vistas": "auto"],
            "archivos": entries,
        ]
        let created: Created = try decode(try await request("POST", "/api/escaneos",
                                                            json: try JSONSerialization.data(withJSONObject: manifest)))

        var sent: Int64 = 0
        for upload in created.subidas {
            let file = folder.appendingPathComponent(upload.ruta)
            var request = URLRequest(url: try absolute(upload.url))
            request.httpMethod = upload.metodo
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = 600
            let (data, response) = try await URLSession.shared.upload(for: request, fromFile: file)
            try check(data, response)
            sent += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            progress(total > 0 ? Double(sent) / Double(total) : 1)
        }

        _ = try await request("POST", "/api/escaneos/\(created.id)/procesar")
        return created.id
    }

    func status(_ id: String) async throws -> RemoteScan {
        try decode(try await request("GET", "/api/escaneos/\(id)"))
    }

    /// Borra el escaneo y todos sus archivos del servidor (libera su disco).
    func delete(_ id: String) async throws {
        _ = try await request("DELETE", "/api/escaneos/\(id)")
    }

    /// Descarga todos los resultados en `destination` (se vacía antes).
    func download(_ scan: RemoteScan, to destination: URL) async throws {
        try? FileManager.default.removeItem(at: destination)
        for result in scan.resultados {
            var request = URLRequest(url: try absolute(result.url))
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            request.timeoutInterval = 600
            let (temporary, response) = try await URLSession.shared.download(for: request)
            try check(Data(), response)
            let target = destination.appendingPathComponent(result.ruta)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.moveItem(at: temporary, to: target)
        }
    }

    // MARK: - Utilidades

    /// Lo que necesita la IA de la carpeta del escaneo (igual que servidor/scripts/cliente.js).
    static func filesToUpload(in folder: URL) -> [String] {
        let fm = FileManager.default
        var files = ["malla_color.ply", "escena.json", "textura.jpg", "edicion.json"]
            .filter { fm.fileExists(atPath: folder.appendingPathComponent($0).path) }
        for (sub, extensions) in [("objetos", ["ply"]), ("fotos_clave", ["jpg", "jpeg", "prof", "json"]),
                                  ("fotogramas", ["jpg", "json"])] {
            let dir = folder.appendingPathComponent(sub)
            let names = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
            files += names.filter { extensions.contains(($0 as NSString).pathExtension.lowercased()) }
                .sorted()
                .map { "\(sub)/\($0)" }
        }
        return files
    }

    private static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func absolute(_ path: String) throws -> URL {
        guard let url = URL(string: path, relativeTo: base)?.absoluteURL else { throw CloudError.server("URL no válida: \(path)") }
        return url
    }

    private func request(_ method: String, _ path: String, json: Data? = nil) async throws -> Data {
        var request = URLRequest(url: try absolute(path))
        request.httpMethod = method
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 60
        if let json {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = json
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        try check(data, response)
        return data
    }

    private func check(_ data: Data, _ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { throw CloudError.server("Respuesta no válida del servidor") }
        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 { throw CloudError.server("Token no válido: revísalo en Ajustes de la nube.") }
            let message = (try? JSONDecoder().decode(ServerError.self, from: data))?.error
            throw CloudError.server(message ?? "Error \(http.statusCode) del servidor")
        }
    }

    private func decode<T: Decodable>(_ data: Data) throws -> T {
        try JSONDecoder().decode(T.self, from: data)
    }
}
