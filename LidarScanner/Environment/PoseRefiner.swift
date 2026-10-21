import simd

/// Alinea un fotograma con lo ya escaneado (seguimiento "fotograma contra modelo", como KinectFusion).
///
/// ARKit da una pose muy buena, pero al caminar acumula unos centímetros de deriva: el fotograma siguiente
/// cae ligeramente desplazado respecto a lo que ya está fusionado y aparecen paredes dobles y texturas
/// movidas. Aquí se busca la pequeña corrección rígida (giro + traslación) que hace que los puntos LiDAR del
/// fotograma queden sobre la superficie del volumen TSDF (distancia con signo = 0).
///
/// Método: Gauss-Newton con pesos de Huber sobre la distancia con signo y su gradiente, 6 grados de libertad.
enum PoseRefiner {
    struct Result {
        let correction: simd_float4x4   // se multiplica por la izquierda a la pose de ARKit
        let inliers: Int
        let rms: Float
    }

    /// - Parameters:
    ///   - points: puntos del fotograma en coordenadas de la cámara de ARKit (x derecha, y arriba, z hacia atrás).
    ///   - pose: pose inicial cámara→mundo (la de ARKit ya con la corrección acumulada).
    ///   - maxTranslation / maxRotation: tope del ajuste respecto a `pose` (metros / radianes).
    static func refine(points: [SIMD3<Float>], pose: simd_float4x4, volume: TSDFVolume,
                       maxTranslation: Float, maxRotation: Float, iterations: Int = 6) -> Result? {
        guard points.count >= 300 else { return nil }
        var delta = matrix_identity_float4x4
        let huber: Float = 0.015
        let band = volume.truncation * 0.8
        var lastInliers = 0
        var lastRMS: Float = 0

        for _ in 0..<iterations {
            let transform = delta * pose
            var h = [Float](repeating: 0, count: 36)
            var b = [Float](repeating: 0, count: 6)
            var inliers = 0
            var squared: Float = 0

            for local in points {
                let w4 = transform * SIMD4<Float>(local, 1)
                let q = SIMD3(w4.x, w4.y, w4.z)
                guard let sample = volume.sample(q), abs(sample.distance) < band else { continue }
                let s = sample.distance, g = sample.gradient
                let gl = simd_length(g)
                guard gl > 0.3 else { continue }   // gradiente fiable (cerca de una superficie)
                let weight: Float = abs(s) < huber ? 1 : huber / abs(s)
                let rot = simd_cross(q, g)
                let j: [Float] = [rot.x, rot.y, rot.z, g.x, g.y, g.z]
                for r in 0..<6 {
                    b[r] += weight * j[r] * s
                    for c in r..<6 { h[r * 6 + c] += weight * j[r] * j[c] }
                }
                inliers += 1
                squared += s * s
            }
            guard inliers >= 300 else { return nil }
            for r in 0..<6 { for c in 0..<r { h[r * 6 + c] = h[c * 6 + r] } }
            for d in 0..<6 { h[d * 6 + d] += 1e-6 * Float(inliers) }   // amortiguación

            guard let x = solve6(h, b.map { -$0 }) else { return nil }
            let omega = SIMD3(x[0], x[1], x[2]), t = SIMD3(x[3], x[4], x[5])
            delta = rigid(rotation: omega, translation: t) * delta
            lastInliers = inliers
            lastRMS = (squared / Float(inliers)).squareRoot()
            if simd_length(omega) < 1e-4 && simd_length(t) < 1e-4 { break }
        }

        // Rechaza saltos implausibles: mejor no corregir que corregir mal.
        let translation = simd_length(SIMD3(delta.columns.3.x, delta.columns.3.y, delta.columns.3.z))
        let trace = delta.columns.0.x + delta.columns.1.y + delta.columns.2.z
        let angle = acos(max(-1, min(1, (trace - 1) / 2)))
        guard translation <= maxTranslation, angle <= maxRotation else { return nil }
        return Result(correction: delta, inliers: lastInliers, rms: lastRMS)
    }

    /// Transformación rígida de un giro pequeño (vector de rotación) y una traslación (fórmula de Rodrigues).
    static func rigid(rotation omega: SIMD3<Float>, translation t: SIMD3<Float>) -> simd_float4x4 {
        let angle = simd_length(omega)
        var r = matrix_identity_float3x3
        if angle > 1e-8 {
            r = simd_float3x3(simd_quatf(angle: angle, axis: omega / angle))
        }
        return simd_float4x4(SIMD4(r.columns.0, 0), SIMD4(r.columns.1, 0), SIMD4(r.columns.2, 0), SIMD4(t, 1))
    }

    /// Resuelve el sistema 6×6 H·x = b por eliminación de Gauss con pivote parcial.
    private static func solve6(_ hIn: [Float], _ bIn: [Float]) -> [Float]? {
        var a = hIn, b = bIn
        for col in 0..<6 {
            var pivot = col
            for r in (col + 1)..<6 where abs(a[r * 6 + col]) > abs(a[pivot * 6 + col]) { pivot = r }
            guard abs(a[pivot * 6 + col]) > 1e-12 else { return nil }
            if pivot != col {
                for c in 0..<6 { a.swapAt(col * 6 + c, pivot * 6 + c) }
                b.swapAt(col, pivot)
            }
            for r in (col + 1)..<6 {
                let factor = a[r * 6 + col] / a[col * 6 + col]
                guard factor != 0 else { continue }
                for c in col..<6 { a[r * 6 + c] -= factor * a[col * 6 + c] }
                b[r] -= factor * b[col]
            }
        }
        var x = [Float](repeating: 0, count: 6)
        for r in stride(from: 5, through: 0, by: -1) {
            var sum = b[r]
            for c in (r + 1)..<6 { sum -= a[r * 6 + c] * x[c] }
            x[r] = sum / a[r * 6 + r]
        }
        return x
    }
}
