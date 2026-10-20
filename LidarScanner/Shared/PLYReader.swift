import Foundation

/// Lee los PLY binarios que escribe `ModelExporter`: posición, normal, coordenadas de textura (opcionales),
/// color y caras triangulares. Las propiedades se localizan por nombre, en el orden del encabezado.
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

        var vertexCount = 0, faceCount = 0
        var offsets: [String: Int] = [:]   // propiedad del vértice → desplazamiento en bytes
        var vertexSize = 0
        var inVertex = false
        for line in header.split(separator: "\n") {
            let parts = line.split(separator: " ").map(String.init)
            if parts.count == 3, parts[0] == "element" {
                inVertex = parts[1] == "vertex"
                if parts[1] == "vertex" { vertexCount = Int(parts[2]) ?? 0 }
                if parts[1] == "face" { faceCount = Int(parts[2]) ?? 0 }
            } else if inVertex, parts.count == 3, parts[0] == "property" {
                offsets[parts[2]] = vertexSize
                vertexSize += (parts[1] == "float" ? 4 : 1)
            }
        }
        guard let ox = offsets["x"], let oy = offsets["y"], let oz = offsets["z"], vertexSize > 0 else {
            throw ReadError.invalidFile(url.lastPathComponent)
        }
        let start = headerEnd.upperBound - data.startIndex
        guard data.count >= start + vertexCount * vertexSize + faceCount * 13 else {
            throw ReadError.invalidFile(url.lastPathComponent)
        }
        let normalOffsets = offsets["nx"].flatMap { nx in offsets["ny"].flatMap { ny in offsets["nz"].map { (nx, ny, $0) } } }
        let uvOffsets = offsets["s"].flatMap { s in offsets["t"].map { (s, $0) } }
        let colorOffsets = offsets["red"].flatMap { r in offsets["green"].flatMap { g in offsets["blue"].map { (r, g, $0) } } }

        var mesh = ColoredMesh()
        mesh.positions.reserveCapacity(vertexCount)
        mesh.normals.reserveCapacity(vertexCount)
        mesh.colors.reserveCapacity(vertexCount)
        if uvOffsets != nil { mesh.uvs.reserveCapacity(vertexCount) }
        mesh.indices.reserveCapacity(faceCount * 3)

        data.withUnsafeBytes { raw in
            func float(_ o: Int) -> Float { raw.loadUnaligned(fromByteOffset: o, as: Float.self) }
            for i in 0..<vertexCount {
                let base = start + i * vertexSize
                mesh.positions.append(SIMD3(float(base + ox), float(base + oy), float(base + oz)))
                if let (nx, ny, nz) = normalOffsets {
                    mesh.normals.append(SIMD3(float(base + nx), float(base + ny), float(base + nz)))
                } else {
                    mesh.normals.append(SIMD3(0, 1, 0))
                }
                if let (s, t) = uvOffsets {
                    // Origen arriba a la izquierda, como espera SceneKit.
                    mesh.uvs.append(SIMD2(float(base + s), 1 - float(base + t)))
                }
                if let (r, g, b) = colorOffsets {
                    mesh.colors.append(SIMD3(raw[base + r], raw[base + g], raw[base + b]))
                } else {
                    mesh.colors.append(SIMD3(200, 200, 200))
                }
            }
            var o = start + vertexCount * vertexSize
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
