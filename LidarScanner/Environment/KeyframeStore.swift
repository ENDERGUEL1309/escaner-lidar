import simd

/// Fotograma clave: foto a color + profundidad + pose. Se usan al final para colorear la malla.
struct Keyframe {
    let image: RGBImage
    let imageCamera: CameraModel
    let depth: [Float]
    let depthWidth: Int
    let depthHeight: Int
    let depthCamera: CameraModel

    var position: SIMD3<Float> { imageCamera.position }

    @inline(__always)
    func depth(atU u: Float, v: Float) -> Float? {
        let x = Int(u), y = Int(v)
        guard u >= 0, v >= 0, x < depthWidth, y < depthHeight else { return nil }
        let d = depth[y * depthWidth + x]
        return d > 0 ? d : nil
    }
}

/// Guarda vistas bien repartidas: solo añade una nueva cuando la cámara se ha movido o girado lo suficiente.
/// Si se llena, descarta una de cada dos y vuelve más exigente el criterio, para cubrir toda la escena.
final class KeyframeStore {
    private(set) var frames: [Keyframe] = []
    private let maxCount: Int
    private var minDistance: Float = 0.15            // metros
    private var minAngleCos: Float = cos(.pi / 12)   // 15°

    init(maxCount: Int = 90) {
        self.maxCount = maxCount
    }

    var count: Int { frames.count }

    func reset() {
        frames.removeAll()
        minDistance = 0.15
        minAngleCos = cos(.pi / 12)
    }

    func shouldAdd(camera: CameraModel) -> Bool {
        guard let last = frames.last else { return true }
        let moved = simd_distance(camera.position, last.position) > minDistance
        let turned = simd_dot(camera.forward, last.imageCamera.forward) < minAngleCos
        return moved || turned
    }

    func add(_ keyframe: Keyframe) {
        frames.append(keyframe)
        if frames.count > maxCount {
            frames = frames.enumerated().filter { $0.offset % 2 == 0 }.map(\.element)
            minDistance *= 1.5
            minAngleCos = cos(min(acos(minAngleCos) * 1.5, .pi / 3))
        }
    }
}
