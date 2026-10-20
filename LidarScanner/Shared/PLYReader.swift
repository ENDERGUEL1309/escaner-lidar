import Foundation

/// Lee los PLY binarios que escribe `ModelExporter` (posición, normal, color y caras triangulares).
enum PLYReader {
    enum ReadError: LocalizedError {
        case invalidFile(String)
        var errorDescription: String? {
            switch self { case .invalidFile(let name): "No se pudo leer \(name)." }
        }
    }

    static func readMesh(from url: URL) throws -> ColoredMesh {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard let headerEnd = data.range(of: Data("end_header\n".utf8)) else {
            throw ReadError.invalidFile(url.lastPathComponent)
        }
        let header = String(decoding: data[data.startIndex..<headerEnd.lowerBound], as: UTF8.self)
        var vertexCount = 0, faceCount = 0, hasNormals = false
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ")
            if parts.count == 3, parts[0] == "element" {
                if parts[1] == "vertex" { vertexCount = Int(parts[2]) ?? 0 }
                if parts[1] == "face" { faceCount = Int(parts[2]) ?? 0 }
            }
            if line.hasPrefix("property float nx") { hasNormals = true }
        }

        let vertexSize = hasNormals ? 27 : 15
        let start = headerEnd.upperBound - data.startIndex
        guard data.count >= start + vertexCount * vertexSize + faceCount * 13 else {
            throw ReadError.invalidFile(url.lastPathComponent)
        }

        var mesh = ColoredMesh()
        mesh.positions.reserveCapacity(vertexCount)
        mesh.normals.reserveCapacity(vertexCount)
        mesh.colors.reserveCapacity(vertexCount)
        mesh.indices.reserveCapacity(faceCount * 3)

        data.withUnsafeBytes { raw in
            var o = start
            for _ in 0..<vertexCount {
                let p = SIMD3<Float>(raw.loadUnaligned(fromByteOffset: o, as: Float.self),
                                     raw.loadUnaligned(fromByteOffset: o + 4, as: Float.self),
                                     raw.loadUnaligned(fromByteOffset: o + 8, as: Float.self))
                o += 12
                var n = SIMD3<Float>(0, 1, 0)
                if hasNormals {
                    n = SIMD3(raw.loadUnaligned(fromByteOffset: o, as: Float.self),
                              raw.loadUnaligned(fromByteOffset: o + 4, as: Float.self),
                              raw.loadUnaligned(fromByteOffset: o + 8, as: Float.self))
                    o += 12
                }
                let c = SIMD3<UInt8>(raw[o], raw[o + 1], raw[o + 2])
                o += 3
                mesh.positions.append(p)
                mesh.normals.append(n)
                mesh.colors.append(c)
            }
            for _ in 0..<faceCount {
                guard raw[o] == 3 else { break }   // solo triángulos
                mesh.indices.append(raw.loadUnaligned(fromByteOffset: o + 1, as: UInt32.self))
                mesh.indices.append(raw.loadUnaligned(fromByteOffset: o + 5, as: UInt32.self))
                mesh.indices.append(raw.loadUnaligned(fromByteOffset: o + 9, as: UInt32.self))
                o += 13
            }
        }
        return mesh
    }
}
