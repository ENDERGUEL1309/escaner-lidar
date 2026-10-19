import Foundation

/// Organiza los escaneos en Documentos/Escaneos (visible desde la app Archivos).
enum ScanStorage {
    static var root: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("Escaneos", isDirectory: true)
    }

    static func newScanFolder(prefix: String) throws -> URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let folder = root.appendingPathComponent("\(prefix)_\(formatter.string(from: Date()))", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    /// Carpetas de escaneo, más recientes primero.
    static func allScanFolders() -> [URL] {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(at: root,
                                                      includingPropertiesForKeys: [.creationDateKey, .isDirectoryKey],
                                                      options: [.skipsHiddenFiles]) else { return [] }
        return items
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { creationDate($0) > creationDate($1) }
    }

    /// Archivos exportados (no subcarpetas como las fotos de Object Capture).
    static func files(in folder: URL) -> [URL] {
        let items = (try? FileManager.default.contentsOfDirectory(at: folder,
                                                                  includingPropertiesForKeys: [.isDirectoryKey],
                                                                  options: [.skipsHiddenFiles])) ?? []
        return items
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == false }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Objetos de `escena.json`, si el escaneo se procesó con detección de objetos.
    static func sceneObjects(in folder: URL) -> [SceneObjectInfo] {
        let url = folder.appendingPathComponent("escena.json")
        guard let data = try? Data(contentsOf: url),
              let scene = try? JSONDecoder().decode(SceneDescription.self, from: data) else { return [] }
        return scene.objetos
    }

    static func fileSizeText(_ url: URL) -> String {
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)
    }

    private static func creationDate(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
    }
}
