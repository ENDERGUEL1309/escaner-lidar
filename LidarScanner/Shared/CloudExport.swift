import Foundation
import simd

/// Guarda en la carpeta del escaneo lo que necesita la IA en la nube (ver docs/ia-nube.md §3):
///
///     fotos_clave/
///       camaras.json   intrínsecos y pose de ARKit de cada foto (matriz por columnas, como simd_float4x4)
///       000.jpg …      fotos a resolución completa (hasta 12 MP)
///       000.prof …     profundidad LiDAR: ancho y alto (uint32) + metros en Float16, 0 = sin dato
///
/// Las fotos se mueven (no se copian) desde la carpeta temporal: se borrarían al terminar el escaneo.
enum CloudExport {
    struct DepthInfo: Codable {
        let archivo: String
        let ancho: Int
        let alto: Int
        let fx: Float
        let fy: Float
        let cx: Float
        let cy: Float
    }

    struct Photo: Codable {
        let archivo: String
        let ancho: Int
        let alto: Int
        let fx: Float
        let fy: Float
        let cx: Float
        let cy: Float
        let cameraToWorld: [Float]
        let marcaTiempo: Double
        let profundidad: DepthInfo?
    }

    struct Cameras: Codable {
        var version = 1
        var convencion = "arkit"
        let fotos: [Photo]
    }

    /// Devuelve cuántas fotos se guardaron.
    @discardableResult
    static func saveKeyframes(_ frames: [Keyframe], to folder: URL) throws -> Int {
        let directory = folder.appendingPathComponent("fotos_clave", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        var photos: [Photo] = []
        for frame in frames.sorted(by: { $0.timestamp < $1.timestamp }) {
            guard let source = frame.fullImageURL, FileManager.default.fileExists(atPath: source.path) else { continue }
            let name = String(format: "%03d", photos.count)
            try FileManager.default.moveItem(at: source, to: directory.appendingPathComponent("\(name).jpg"))

            let depthName = "\(name).prof"
            try depthData(frame).write(to: directory.appendingPathComponent(depthName), options: .atomic)

            let camera = frame.fullCamera
            let depthCamera = frame.depthCamera
            photos.append(Photo(archivo: "\(name).jpg",
                                ancho: frame.fullWidth, alto: frame.fullHeight,
                                fx: camera.fx, fy: camera.fy, cx: camera.cx, cy: camera.cy,
                                cameraToWorld: columns(of: camera.cameraToWorld),
                                marcaTiempo: frame.timestamp,
                                profundidad: DepthInfo(archivo: depthName,
                                                       ancho: frame.depthWidth, alto: frame.depthHeight,
                                                       fx: depthCamera.fx, fy: depthCamera.fy,
                                                       cx: depthCamera.cx, cy: depthCamera.cy)))
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(Cameras(fotos: photos)).write(to: directory.appendingPathComponent("camaras.json"), options: .atomic)
        return photos.count
    }

    /// Mueve los fotogramas para Gaussian Splatting a `fotogramas/` y escribe su `camaras.json`
    /// (mismo formato que `fotos_clave/`, sin profundidad).
    static func saveFrames(_ records: [Photo], from temporary: URL, to folder: URL) throws {
        guard !records.isEmpty else { return }
        let directory = folder.appendingPathComponent("fotogramas", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var saved: [Photo] = []
        for record in records {
            let source = temporary.appendingPathComponent(record.archivo)
            guard (try? FileManager.default.moveItem(at: source, to: directory.appendingPathComponent(record.archivo))) != nil
            else { continue }
            saved.append(record)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(Cameras(fotos: saved)).write(to: directory.appendingPathComponent("camaras.json"), options: .atomic)
    }

    /// 16 valores por columnas, igual que `simd_float4x4`.
    private static func columns(of m: simd_float4x4) -> [Float] {
        [m.columns.0, m.columns.1, m.columns.2, m.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }
    }

    private static func depthData(_ frame: Keyframe) -> Data {
        var data = Data(capacity: 8 + frame.depth.count * 2)
        withUnsafeBytes(of: UInt32(frame.depthWidth).littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: UInt32(frame.depthHeight).littleEndian) { data.append(contentsOf: $0) }
        for d in frame.depth {
            withUnsafeBytes(of: Float16(d).bitPattern.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }
}
