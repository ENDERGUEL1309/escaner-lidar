import Foundation
import simd

/// Limpieza de la malla fusionada: quita restos flotantes y suaviza el ruido residual del LiDAR.
enum MeshCleaner {
    /// Elimina las piezas sueltas con menos de `minTriangles` triángulos (lecturas falsas, reflejos, polvo).
    static func removeSmallComponents(_ mesh: inout ColoredMesh, minTriangles: Int) {
        let n = mesh.positions.count
        let triCount = mesh.triangleCount
        guard n > 0, triCount > 0 else { return }

        var parent = (0..<n).map { Int32($0) }
        func find(_ x: Int32) -> Int32 {
            var x = x
            while parent[Int(x)] != x {
                parent[Int(x)] = parent[Int(parent[Int(x)])]
                x = parent[Int(x)]
            }
            return x
        }
        for t in 0..<triCount {
            let a = find(Int32(mesh.indices[t * 3]))
            let b = find(Int32(mesh.indices[t * 3 + 1]))
            let c = find(Int32(mesh.indices[t * 3 + 2]))
            if a != b { parent[Int(a)] = b }
            let rb = find(b)
            if rb != c { parent[Int(rb)] = c }
        }

        var size: [Int32: Int] = [:]
        for t in 0..<triCount { size[find(Int32(mesh.indices[t * 3])), default: 0] += 1 }
        let keep = (0..<triCount).filter { size[find(Int32(mesh.indices[$0 * 3])), default: 0] >= minTriangles }
        guard keep.count < triCount, !keep.isEmpty else { return }
        mesh = mesh.submesh(triangles: keep)
    }

    /// Suavizado de Taubin (λ/μ): quita el "granulado" sin encoger la malla como haría un suavizado simple.
    static func smooth(_ mesh: inout ColoredMesh, iterations: Int, lambda: Float = 0.5, mu: Float = -0.53) {
        let n = mesh.positions.count
        guard n > 0, iterations > 0 else { return }

        // Vecinos de cada vértice (CSR) a partir de las aristas de los triángulos.
        var counts = [Int](repeating: 0, count: n + 1)
        for idx in mesh.indices { counts[Int(idx) + 1] += 2 }
        for i in 0..<n { counts[i + 1] += counts[i] }
        var fill = counts
        var neighbors = [Int32](repeating: 0, count: counts[n])
        for t in 0..<mesh.triangleCount {
            let v = [Int(mesh.indices[t * 3]), Int(mesh.indices[t * 3 + 1]), Int(mesh.indices[t * 3 + 2])]
            for k in 0..<3 {
                let a = v[k]
                neighbors[fill[a]] = Int32(v[(k + 1) % 3]); fill[a] += 1
                neighbors[fill[a]] = Int32(v[(k + 2) % 3]); fill[a] += 1
            }
        }

        var current = mesh.positions
        for iteration in 0..<(iterations * 2) {
            let factor = iteration % 2 == 0 ? lambda : mu
            let source = current
            current.withUnsafeMutableBufferPointer { out in
                let chunk = 8192
                DispatchQueue.concurrentPerform(iterations: (n + chunk - 1) / chunk) { c in
                    for i in (c * chunk)..<min(n, (c + 1) * chunk) {
                        let start = counts[i], end = counts[i + 1]
                        guard end > start else { continue }
                        var sum = SIMD3<Float>(repeating: 0)
                        for j in start..<end { sum += source[Int(neighbors[j])] }
                        let average = sum / Float(end - start)
                        out[i] = source[i] + (average - source[i]) * factor
                    }
                }
            }
        }
        mesh.positions = current
    }

    /// Normales por vértice a partir de las caras (ponderadas por área), orientadas como las originales.
    static func recomputeNormals(_ mesh: inout ColoredMesh) {
        var accumulated = [SIMD3<Float>](repeating: .zero, count: mesh.positions.count)
        for t in 0..<mesh.triangleCount {
            let a = Int(mesh.indices[t * 3]), b = Int(mesh.indices[t * 3 + 1]), c = Int(mesh.indices[t * 3 + 2])
            let n = simd_cross(mesh.positions[b] - mesh.positions[a], mesh.positions[c] - mesh.positions[a])
            accumulated[a] += n; accumulated[b] += n; accumulated[c] += n
        }
        for i in accumulated.indices {
            let len = simd_length(accumulated[i])
            guard len > 0 else { continue }
            var n = accumulated[i] / len
            if simd_dot(n, mesh.normals[i]) < 0 { n = -n }
            mesh.normals[i] = n
        }
    }
}
