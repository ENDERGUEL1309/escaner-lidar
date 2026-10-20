import ARKit

/// Malla triangular en coordenadas del mundo (metros, Y hacia arriba) con color por vértice.
struct ColoredMesh {
    var positions: [SIMD3<Float>] = []
    var normals: [SIMD3<Float>] = []
    var colors: [SIMD3<UInt8>] = []
    var indices: [UInt32] = []
    /// Coordenadas de textura (0–1, origen arriba a la izquierda) si la malla está texturizada.
    var uvs: [SIMD2<Float>] = []

    var isEmpty: Bool { positions.isEmpty || indices.isEmpty }
    var triangleCount: Int { indices.count / 3 }
    var hasTexture: Bool { !uvs.isEmpty }

    /// Submalla con los triángulos indicados (índice de triángulo, no de vértice), con vértices reindexados.
    func submesh(triangles: [Int]) -> ColoredMesh {
        var sub = ColoredMesh()
        var remap: [UInt32: UInt32] = [:]
        remap.reserveCapacity(triangles.count * 2)
        for t in triangles {
            for k in 0..<3 {
                let v = indices[t * 3 + k]
                if let r = remap[v] {
                    sub.indices.append(r)
                    continue
                }
                let r = UInt32(sub.positions.count)
                remap[v] = r
                let i = Int(v)
                sub.positions.append(positions[i])
                sub.normals.append(normals[i])
                sub.colors.append(colors[i])
                if hasTexture { sub.uvs.append(uvs[i]) }
                sub.indices.append(r)
            }
        }
        return sub
    }
}

enum MeshBuilder {
    /// Une todos los fragmentos (`ARMeshAnchor`) de la reconstrucción de escena en una sola malla.
    static func build(from anchors: [ARMeshAnchor]) -> ColoredMesh {
        var mesh = ColoredMesh()
        let totalVertices = anchors.reduce(0) { $0 + $1.geometry.vertices.count }
        mesh.positions.reserveCapacity(totalVertices)
        mesh.normals.reserveCapacity(totalVertices)

        for anchor in anchors {
            let geometry = anchor.geometry
            let transform = anchor.transform
            let rotation = simd_float3x3(SIMD3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
                                         SIMD3(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
                                         SIMD3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z))
            let base = UInt32(mesh.positions.count)

            let vertices = geometry.vertices
            let normals = geometry.normals
            for i in 0..<vertices.count {
                let local = vertices.float3(at: i)
                let world = transform * SIMD4<Float>(local, 1)
                mesh.positions.append(SIMD3(world.x, world.y, world.z))
                let n = rotation * normals.float3(at: i)
                let len = simd_length(n)
                mesh.normals.append(len > 0 ? n / len : SIMD3(0, 1, 0))
            }

            let faces = geometry.faces
            let indexCount = faces.count * faces.indexCountPerPrimitive
            let raw = faces.buffer.contents()
            if faces.bytesPerIndex == 4 {
                let ptr = raw.assumingMemoryBound(to: UInt32.self)
                for k in 0..<indexCount { mesh.indices.append(base + ptr[k]) }
            } else {
                let ptr = raw.assumingMemoryBound(to: UInt16.self)
                for k in 0..<indexCount { mesh.indices.append(base + UInt32(ptr[k])) }
            }
        }

        mesh.colors = Array(repeating: SIMD3(160, 160, 160), count: mesh.positions.count)
        return mesh
    }
}

extension ARGeometrySource {
    @inline(__always)
    func float3(at index: Int) -> SIMD3<Float> {
        let p = buffer.contents()
            .advanced(by: offset + stride * index)
            .assumingMemoryBound(to: Float.self)
        return SIMD3(p[0], p[1], p[2])
    }
}
