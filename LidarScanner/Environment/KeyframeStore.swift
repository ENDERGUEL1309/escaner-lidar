import Foundation
import simd

/// Fotograma clave: foto a color + profundidad + pose.
///  • `image` (~1000 px) se usa para el color por vértice y para la IA de identificación.
///  • `fullImageURL` es la foto a resolución completa (hasta 12 MP) en disco, para la textura.
struct Keyframe {
    let image: RGBImage
    let imageCamera: CameraModel
    let depth: [Float]
    let depthWidth: Int
    let depthHeight: Int
    let depthCamera: CameraModel
    let fullImageURL: URL?
    let fullCamera: CameraModel
    let fullWidth: Int
    let fullHeight: Int
    /// Momento de la captura (segundos de ARKit), para ordenar las fotos.
    let timestamp: TimeInterval

    var position: SIMD3<Float> { imageCamera.position }

    @inline(__always)
    func depth(atU u: Float, v: Float) -> Float? {
        let x = Int(u), y = Int(v)
        guard u >= 0, v >= 0, x < depthWidth, y < depthHeight else { return nil }
        let d = depth[y * depthWidth + x]
        return d > 0 ? d : nil
    }
}

/// Guarda las vistas del escaneo. Si se llena, descarta una de cada dos (y sus fotos en disco);
/// `decimations` indica cuántas veces ha pasado, para espaciar más las siguientes capturas.
/// No es thread-safe: úsalo siempre desde la misma cola serie.
final class KeyframeStore {
    private(set) var frames: [Keyframe] = []
    private(set) var decimations = 0
    private let maxCount: Int

    init(maxCount: Int = 60) {
        self.maxCount = maxCount
    }

    var count: Int { frames.count }

    func reset() {
        for frame in frames { deleteFile(of: frame) }
        frames.removeAll()
        decimations = 0
    }

    func add(_ keyframe: Keyframe) {
        frames.append(keyframe)
        guard frames.count > maxCount else { return }
        var kept: [Keyframe] = []
        for (i, frame) in frames.enumerated() {
            if i % 2 == 0 { kept.append(frame) } else { deleteFile(of: frame) }
        }
        frames = kept
        decimations += 1
    }

    private func deleteFile(of frame: Keyframe) {
        if let url = frame.fullImageURL { try? FileManager.default.removeItem(at: url) }
    }
}
