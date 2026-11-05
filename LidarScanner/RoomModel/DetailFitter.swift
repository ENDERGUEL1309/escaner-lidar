import Foundation
import SceneKit
import simd

/// Coloca el modelo de Object Capture de un mueble dentro de su caja de RoomPlan.
///
/// RoomPlan sabe dónde está el mueble en la sala; Object Capture sabe cómo es, a escala real pero con un giro
/// cualquiera. Aquí se busca el giro que deja la planta del modelo en el rectángulo más pequeño, se elige cuál de
/// sus lados es el ancho comparando con la caja, y se apoya en el suelo, centrado en la planta.
enum DetailFitter {
    enum Failure: LocalizedError {
        case unreadable, empty

        var errorDescription: String? {
            switch self {
            case .unreadable: "No se pudo abrir el modelo escaneado."
            case .empty: "El modelo escaneado no tiene geometría."
            }
        }
    }

    /// Todo el contenido del USDZ bajo un nodo nuevo, listo para clonar.
    static func loadNode(_ url: URL) -> SCNNode? {
        guard let scene = try? SCNScene(url: url, options: nil) else { return nil }
        let node = SCNNode()
        for child in scene.rootNode.childNodes { node.addChildNode(child) }
        return node
    }

    /// Vértices del modelo en las coordenadas de `root` (como mucho `limit`, repartidos por todas las mallas).
    static func points(in root: SCNNode, limit: Int = 60_000) -> [SIMD3<Float>] {
        var sources: [(SCNGeometrySource, simd_float4x4)] = []
        root.enumerateHierarchy { node, _ in
            guard let geometry = node.geometry else { return }
            let m = node.simdConvertTransform(matrix_identity_float4x4, to: root)
            for s in geometry.sources(for: .vertex)
            where s.usesFloatComponents && s.bytesPerComponent == 4 && s.componentsPerVector >= 3 {
                sources.append((s, m))
            }
        }
        let total = sources.reduce(0) { $0 + $1.0.vectorCount }
        let step = max(1, total / limit)
        var result: [SIMD3<Float>] = []
        result.reserveCapacity(min(total, limit + sources.count))
        for (s, m) in sources {
            s.data.withUnsafeBytes { buffer in
                for i in stride(from: 0, to: s.vectorCount, by: step) {
                    let o = s.dataOffset + i * s.dataStride
                    guard o + 12 <= buffer.count else { break }
                    let p = SIMD4<Float>(buffer.loadUnaligned(fromByteOffset: o, as: Float.self),
                                         buffer.loadUnaligned(fromByteOffset: o + 4, as: Float.self),
                                         buffer.loadUnaligned(fromByteOffset: o + 8, as: Float.self), 1)
                    let w = m * p
                    result.append(SIMD3(w.x, w.y, w.z))
                }
            }
        }
        return result
    }

    private static func rotationY(_ angle: Float) -> simd_float4x4 {
        simd_float4x4(simd_quatf(angle: angle, axis: SIMD3(0, 1, 0)))
    }

    private static func bounds(_ points: [SIMD3<Float>], _ m: simd_float4x4) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for p in points {
            let q = m * SIMD4(p, 1)
            let v = SIMD3(q.x, q.y, q.z)
            lo = simd_min(lo, v)
            hi = simd_max(hi, v)
        }
        return (lo, hi)
    }

    /// Altura del 90 % más alto de los puntos en una mitad (fondo o frente) del mueble.
    private static func tallHeight(_ ys: [Float]) -> Float {
        guard !ys.isEmpty else { return 0 }
        let sorted = ys.sorted()
        return sorted[min(sorted.count - 1, Int(Float(sorted.count) * 0.9))]
    }

    static func fit(modelURL: URL, file: String, furniture f: ParametricFurniture) throws -> DetailModel {
        guard let node = loadNode(modelURL) else { throw Failure.unreadable }
        let points = points(in: node)
        guard points.count >= 50 else { throw Failure.empty }

        // Unidades: Object Capture escribe metros, pero si algún visor lo lee en centímetros salta a la vista.
        var scale: Float = 1
        let raw = bounds(points, matrix_identity_float4x4)
        let height = raw.max.y - raw.min.y
        if f.size.y > 0.05 {
            let ratio = height / f.size.y
            if ratio > 20 { scale = 0.01 } else if ratio < 0.05 { scale = 100 }
        }
        let scaling = simd_float4x4(diagonal: SIMD4(scale, scale, scale, 1))

        // Giro de la planta con el rectángulo envolvente más pequeño (cada grado de 0 a 89).
        let sample = stride(from: 0, to: points.count, by: max(1, points.count / 20_000)).map { points[$0] * scale }
        var bestAngle: Float = 0, bestArea = Float.infinity
        for degree in 0..<90 {
            let a = Float(degree) * .pi / 180, c = cos(a), s = sin(a)
            var minX = Float.infinity, maxX = -Float.infinity, minZ = Float.infinity, maxZ = -Float.infinity
            for p in sample {
                let x = c * p.x + s * p.z, z = -s * p.x + c * p.z
                minX = min(minX, x); maxX = max(maxX, x)
                minZ = min(minZ, z); maxZ = max(maxZ, z)
            }
            let area = (maxX - minX) * (maxZ - minZ)
            if area < bestArea { bestArea = area; bestAngle = a }
        }

        // De los dos lados, el que se parece al ancho de la caja de RoomPlan va en x.
        func extents(_ angle: Float) -> SIMD3<Float> {
            let b = bounds(sample, rotationY(angle))
            return b.max - b.min
        }
        let e0 = extents(bestAngle), e1 = extents(bestAngle + .pi / 2)
        func error(_ e: SIMD3<Float>) -> Float { abs(e.x - f.size.x) + abs(e.z - f.size.z) }
        var angle = error(e0) <= error(e1) ? bestAngle : bestAngle + .pi / 2

        // Sofás, camas y sillas tienen el respaldo o cabecero detrás (−z): si lo alto queda delante, media vuelta.
        if f.kind == .sofa || f.kind == .bed || f.kind == .chair {
            let r = rotationY(angle)
            let b = bounds(sample, r)
            let midZ = (b.min.z + b.max.z) / 2
            var back: [Float] = [], front: [Float] = []
            for p in sample {
                let q = r * SIMD4(p, 1)
                if q.z < midZ { back.append(q.y) } else { front.append(q.y) }
            }
            if tallHeight(front) > tallHeight(back) + 0.05 { angle += .pi }
        }

        let rotated = rotationY(angle) * scaling
        let b = bounds(points, rotated)
        let pivot = SIMD3((b.min.x + b.max.x) / 2, b.min.y, (b.min.z + b.max.z) / 2)
        var translation = matrix_identity_float4x4
        translation.columns.3 = SIMD4(-pivot, 1)

        var detail = DetailModel(file: file, matrix: [], measured: b.max - b.min, roomPlanSize: f.size)
        detail.transform = translation * rotated
        return detail
    }
}
