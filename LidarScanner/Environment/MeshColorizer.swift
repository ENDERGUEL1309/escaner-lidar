import Foundation
import simd

/// Asigna a cada vértice de la malla el color visto por las cámaras de los fotogramas clave.
///
/// Para cada vértice y cada vista:
///  1. Se descarta si la cara mira hacia el otro lado (normal vs. dirección a la cámara).
///  2. Se proyecta en la foto; si cae fuera, se descarta.
///  3. Prueba de oclusión con el mapa de profundidad LiDAR de esa vista: si hay algo delante, se descarta.
///  4. Se pondera favoreciendo vistas frontales y cercanas (más resolución, menos distorsión).
enum MeshColorizer {
    static func colorize(_ mesh: inout ColoredMesh,
                         keyframes: [Keyframe],
                         progress: @escaping (Double) -> Void) {
        let count = mesh.positions.count
        guard count > 0, !keyframes.isEmpty else { return }

        let positions = mesh.positions
        let normals = mesh.normals
        var colors = mesh.colors
        let chunkSize = 8192
        let chunks = (count + chunkSize - 1) / chunkSize
        let lock = NSLock()
        var completed = 0

        colors.withUnsafeMutableBufferPointer { out in
            DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                let end = min(count, (chunk + 1) * chunkSize)
                for i in (chunk * chunkSize)..<end {
                    if let c = color(at: positions[i], normal: normals[i], keyframes: keyframes) {
                        out[i] = c
                    }
                }
                lock.lock()
                completed += 1
                let done = completed
                lock.unlock()
                if done % 4 == 0 || done == chunks { progress(Double(done) / Double(chunks)) }
            }
        }
        mesh.colors = colors
    }

    @inline(__always)
    private static func color(at p: SIMD3<Float>, normal n: SIMD3<Float>, keyframes: [Keyframe]) -> SIMD3<UInt8>? {
        var accumulated = SIMD3<Float>(repeating: 0)
        var totalWeight: Float = 0

        for kf in keyframes {
            let toCamera = kf.position - p
            let distance = simd_length(toCamera)
            guard distance > 0.1 else { continue }
            let cosAngle = simd_dot(n, toCamera / distance)
            guard cosAngle > 0.25 else { continue }

            guard let proj = kf.imageCamera.project(p),
                  let c = kf.image.color(atX: proj.u, y: proj.v) else { continue }

            if let dp = kf.depthCamera.project(p), let sensed = kf.depth(atU: dp.u, v: dp.v) {
                let tolerance: Float = 0.03 + 0.02 * sensed
                if dp.z > sensed + tolerance { continue }   // hay algo delante: ocluido
            }

            // Bordes de la foto: más distorsión y viñeteo → menos peso.
            let ex = min(proj.u, Float(kf.image.width) - proj.u) / Float(kf.image.width)
            let ey = min(proj.v, Float(kf.image.height) - proj.v) / Float(kf.image.height)
            let edge = min(1, min(ex, ey) * 10)

            let c2 = cosAngle * cosAngle
            let weight = c2 * c2 * c2 * edge / (distance * distance)
            accumulated += c * weight
            totalWeight += weight
        }

        guard totalWeight > 0 else { return nil }
        return rgb8(accumulated / totalWeight)
    }
}
