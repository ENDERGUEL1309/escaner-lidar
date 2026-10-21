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

    /// Quita los triángulos con algún lado más largo que `maxEdge`. En los bordes de lo observado la fusión deja
    /// triángulos largos y finos (y la textura se estira sobre ellos): son los "pinchos" de los bordes.
    static func removeStretchedTriangles(_ mesh: inout ColoredMesh, maxEdge: Float) {
        let limit = maxEdge * maxEdge
        let keep = (0..<mesh.triangleCount).filter { t in
            let a = mesh.positions[Int(mesh.indices[t * 3])]
            let b = mesh.positions[Int(mesh.indices[t * 3 + 1])]
            let c = mesh.positions[Int(mesh.indices[t * 3 + 2])]
            return simd_distance_squared(a, b) <= limit && simd_distance_squared(b, c) <= limit
                && simd_distance_squared(a, c) <= limit
        }
        guard keep.count < mesh.triangleCount, !keep.isEmpty else { return }
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

    /// Pulido de la estructura: los vértices que la red de ARKit clasificó como suelo, techo o pared se
    /// ajustan al plano dominante más cercano (si están a menos de `tolerance`). Elimina las ondulaciones
    /// del LiDAR en superficies grandes sin tocar muebles ni detalles de más de unos centímetros.
    static func flattenStructures(_ mesh: inout ColoredMesh, labels: [SurfaceLabel], tolerance: Float = 0.03) {
        guard labels.count == mesh.positions.count else { return }

        // Suelo y techo: alturas dominantes (puede haber varias: escalones, falsos techos).
        for label in [SurfaceLabel.floor, .ceiling] {
            let members = mesh.positions.indices.filter { labels[$0] == label && abs(mesh.normals[$0].y) > 0.8 }
            guard members.count > 500 else { continue }
            var histogram: [Int: Int] = [:]
            for i in members { histogram[Int((mesh.positions[i].y / 0.01).rounded()), default: 0] += 1 }
            let minVotes = max(300, members.count / 20)
            let peaks = histogram.filter { bin, votes in
                votes >= minVotes && (-3...3).allSatisfy { histogram[bin + $0, default: 0] <= votes }
            }.keys
            var heights: [Float] = []
            for bin in peaks {
                let center = Float(bin) * 0.01
                let near = members.filter { abs(mesh.positions[$0].y - center) < tolerance }
                guard !near.isEmpty else { continue }
                heights.append(near.reduce(Float(0)) { $0 + mesh.positions[$1].y } / Float(near.count))
            }
            guard !heights.isEmpty else { continue }
            for i in mesh.positions.indices where labels[i] == label {
                let y = mesh.positions[i].y
                if let h = heights.min(by: { abs($0 - y) < abs($1 - y) }), abs(h - y) < tolerance {
                    mesh.positions[i].y = h
                }
            }
        }

        // Paredes: planos verticales, votados por orientación (3°) y distancia al origen (2 cm).
        let walls = mesh.positions.indices.filter { labels[$0] == .wall && abs(mesh.normals[$0].y) < 0.3 }
        guard walls.count > 500 else { return }
        func horizontal(_ n: SIMD3<Float>) -> SIMD3<Float> {
            let h = SIMD3<Float>(n.x, 0, n.z)
            let len = simd_length(h)
            return len > 0 ? h / len : SIMD3(1, 0, 0)
        }
        struct Bin: Hashable { let angle: Int; let distance: Int }
        var votes: [Bin: Int] = [:]
        for i in walls {
            let n = horizontal(mesh.normals[i])
            let angle = Int((atan2(n.z, n.x) * 180 / .pi / 3).rounded())
            let distance = Int((simd_dot(n, mesh.positions[i]) / 0.02).rounded())
            votes[Bin(angle: angle, distance: distance), default: 0] += 1
        }
        var planes: [(normal: SIMD3<Float>, offset: Float)] = []
        for (bin, count) in votes.sorted(by: { $0.value > $1.value }) where count >= 400 {
            let angle = Float(bin.angle) * 3 * .pi / 180
            let normal = SIMD3<Float>(cos(angle), 0, sin(angle))
            let offset = Float(bin.distance) * 0.02
            let duplicate = planes.contains { simd_dot($0.normal, normal) > cos(6 * Float.pi / 180) && abs($0.offset - offset) < 0.06 }
            guard !duplicate else { continue }
            // Ajuste fino: media de los vértices que caen cerca del plano votado.
            var normalSum = SIMD3<Float>(repeating: 0), offsetSum: Float = 0, count: Float = 0
            for i in walls {
                let n = horizontal(mesh.normals[i])
                let d = simd_dot(normal, mesh.positions[i])
                if simd_dot(n, normal) > cos(10 * Float.pi / 180), abs(d - offset) < tolerance {
                    normalSum += n; offsetSum += d; count += 1
                }
            }
            guard count > 0 else { continue }
            let refined = horizontal(normalSum)
            planes.append((refined, offsetSum / count))
            if planes.count >= 24 { break }
        }
        for i in walls {
            let n = horizontal(mesh.normals[i])
            let p = mesh.positions[i]
            var best: (normal: SIMD3<Float>, distance: Float)?
            for plane in planes where simd_dot(n, plane.normal) > cos(15 * Float.pi / 180) {
                let distance = simd_dot(plane.normal, p) - plane.offset
                if abs(distance) < tolerance, abs(distance) < abs(best?.distance ?? .greatestFiniteMagnitude) {
                    best = (plane.normal, distance)
                }
            }
            if let best { mesh.positions[i] = p - best.normal * best.distance }
        }
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
