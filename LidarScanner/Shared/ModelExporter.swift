import Foundation
import SceneKit
import UIKit

/// Exportadores de malla/nube de puntos.
///  • PLY binario: el más fiel (Blender, MeshLab, CloudCompare). Normales, color por vértice y, si la hay,
///    coordenadas de textura (s, t) con `comment TextureFile`.
///  • OBJ (+ MTL + JPG): cada parte es un objeto `o` independiente (Blender lo importa como objetos
///    separados que se pueden mover, borrar o editar). Con textura si la malla la tiene.
///  • USDZ: vista 3D / AR en el iPhone; cada parte es un nodo con nombre; incluye la textura.
///  • JSON: inventario de la escena (nombre, categoría, centro, medidas) para procesarla con otros programas.
///
/// Ojo: `SIMD3<Float>` ocupa 16 bytes en memoria, así que los valores se escriben componente a componente.
enum ModelExporter {
    struct Part {
        let name: String
        let mesh: ColoredMesh
    }

    // MARK: - PLY

    /// `textureFile`: imagen de textura (ruta relativa al PLY) si la malla tiene coordenadas de textura.
    static func writePLY(mesh: ColoredMesh, to url: URL, textureFile: String? = nil) throws {
        let header = plyHeader(vertexCount: mesh.positions.count, faceCount: mesh.triangleCount,
                               textured: mesh.hasTexture, textureFile: textureFile)
        var bytes = [UInt8](header.utf8)
        bytes.reserveCapacity(bytes.count + mesh.positions.count * 35 + mesh.triangleCount * 13)
        appendVertices(of: mesh, to: &bytes)
        for t in 0..<mesh.triangleCount {
            bytes.append(3)
            bytes.append(uint32: mesh.indices[t * 3])
            bytes.append(uint32: mesh.indices[t * 3 + 1])
            bytes.append(uint32: mesh.indices[t * 3 + 2])
        }
        try Data(bytes).write(to: url, options: .atomic)
    }

    /// Nube de puntos densa: los vértices de la superficie fusionada (sin ruido y con huecos cerrados), con normales.
    static func writePointCloudPLY(mesh: ColoredMesh, to url: URL) throws {
        var points = mesh
        points.uvs = []
        var bytes = [UInt8](plyHeader(vertexCount: points.positions.count, faceCount: nil,
                                      textured: false, textureFile: nil).utf8)
        bytes.reserveCapacity(bytes.count + points.positions.count * 27)
        appendVertices(of: points, to: &bytes)
        try Data(bytes).write(to: url, options: .atomic)
    }

    private static func plyHeader(vertexCount: Int, faceCount: Int?, textured: Bool, textureFile: String?) -> String {
        var header = "ply\nformat binary_little_endian 1.0\ncomment Escaner LiDAR - iPhone\n"
        if let textureFile { header += "comment TextureFile \(textureFile)\n" }
        header += "element vertex \(vertexCount)\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property float nx\nproperty float ny\nproperty float nz\n"
        if textured { header += "property float s\nproperty float t\n" }
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        if let faceCount {
            header += "element face \(faceCount)\nproperty list uchar uint vertex_indices\n"
        }
        return header + "end_header\n"
    }

    private static func appendVertices(of mesh: ColoredMesh, to bytes: inout [UInt8]) {
        let textured = mesh.hasTexture
        for i in mesh.positions.indices {
            let p = mesh.positions[i], n = mesh.normals[i], c = mesh.colors[i]
            bytes.append(float: p.x); bytes.append(float: p.y); bytes.append(float: p.z)
            bytes.append(float: n.x); bytes.append(float: n.y); bytes.append(float: n.z)
            if textured {
                // PLY y OBJ usan el origen de la textura abajo a la izquierda.
                bytes.append(float: mesh.uvs[i].x); bytes.append(float: 1 - mesh.uvs[i].y)
            }
            bytes.append(c.x); bytes.append(c.y); bytes.append(c.z)
        }
    }

    // MARK: - OBJ

    /// `materialFile`: archivo .mtl con la textura (solo se usa si las partes tienen coordenadas de textura).
    static func writeOBJ(parts: [Part], to url: URL, materialFile: String? = nil) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw ScanError.exportFailed("OBJ")
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }

        var text = "# Escaner LiDAR - iPhone\n# Unidades: metros, eje Y hacia arriba\n"
        let useMaterial = materialFile != nil && parts.contains { $0.mesh.hasTexture }
        if useMaterial, let materialFile { text += "mtllib \(materialFile)\n" }
        text.reserveCapacity(1 << 21)
        func flushIfNeeded(force: Bool = false) throws {
            if force || text.utf8.count > (1 << 20) {
                try handle.write(contentsOf: Data(text.utf8))
                text.removeAll(keepingCapacity: true)
            }
        }

        var offset: UInt32 = 1   // los índices de OBJ empiezan en 1 y son globales al archivo
        for part in parts {
            let mesh = part.mesh
            let textured = useMaterial && mesh.hasTexture
            let name = objectName(part.name)
            text += "o \(name)\ng \(name)\n"
            if textured { text += "usemtl escaneo\n" }
            for i in mesh.positions.indices {
                let p = mesh.positions[i], c = mesh.colors[i]
                text += String(format: "v %.5f %.5f %.5f %.4f %.4f %.4f\n",
                               p.x, p.y, p.z, Float(c.x) / 255, Float(c.y) / 255, Float(c.z) / 255)
                try flushIfNeeded()
            }
            if textured {
                for uv in mesh.uvs {
                    text += String(format: "vt %.6f %.6f\n", uv.x, 1 - uv.y)
                    try flushIfNeeded()
                }
            }
            for n in mesh.normals {
                text += String(format: "vn %.4f %.4f %.4f\n", n.x, n.y, n.z)
                try flushIfNeeded()
            }
            for t in 0..<mesh.triangleCount {
                let a = mesh.indices[t * 3] + offset, b = mesh.indices[t * 3 + 1] + offset, c = mesh.indices[t * 3 + 2] + offset
                text += textured
                    ? "f \(a)/\(a)/\(a) \(b)/\(b)/\(b) \(c)/\(c)/\(c)\n"
                    : "f \(a)//\(a) \(b)//\(b) \(c)//\(c)\n"
                try flushIfNeeded()
            }
            offset += UInt32(mesh.positions.count)
        }
        try flushIfNeeded(force: true)
    }

    static func writeMTL(to url: URL, textureFile: String) throws {
        let text = "# Escaner LiDAR - iPhone\nnewmtl escaneo\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\n"
            + "Ks 0.000 0.000 0.000\nillum 1\nmap_Kd \(textureFile)\n"
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    // MARK: - USDZ

    static func writeUSDZ(parts: [Part], texture: UIImage? = nil, to url: URL) throws {
        let scene = SCNScene()
        for part in parts where !part.mesh.isEmpty {
            let node = SCNNode(geometry: makeGeometry(part.mesh, texture: texture))
            node.name = objectName(part.name)
            scene.rootNode.addChildNode(node)
        }
        guard scene.write(to: url, options: nil, delegate: nil, progressHandler: nil) else {
            throw ScanError.exportFailed("USDZ")
        }
    }

    /// Geometría SceneKit: con textura si la malla tiene coordenadas de textura y se pasa la imagen;
    /// si no, con el color por vértice.
    static func makeGeometry(_ mesh: ColoredMesh, texture: UIImage? = nil) -> SCNGeometry {
        let vertexSource = SCNGeometrySource(vertices: mesh.positions.map { SCNVector3($0.x, $0.y, $0.z) })
        let normalSource = SCNGeometrySource(normals: mesh.normals.map { SCNVector3($0.x, $0.y, $0.z) })
        var sources = [vertexSource, normalSource]

        let material = SCNMaterial()
        material.lightingModel = .physicallyBased
        material.roughness.contents = 0.9
        material.isDoubleSided = true

        if let texture, mesh.hasTexture {
            sources.append(SCNGeometrySource(textureCoordinates: mesh.uvs.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }))
            material.diffuse.contents = texture
            material.diffuse.mipFilter = .linear
        } else {
            var packedColors = [Float]()
            packedColors.reserveCapacity(mesh.colors.count * 3)
            for c in mesh.colors {
                packedColors.append(Float(c.x) / 255)
                packedColors.append(Float(c.y) / 255)
                packedColors.append(Float(c.z) / 255)
            }
            let colorData = packedColors.withUnsafeBufferPointer { Data(buffer: $0) }
            sources.append(SCNGeometrySource(data: colorData,
                                             semantic: .color,
                                             vectorCount: mesh.colors.count,
                                             usesFloatComponents: true,
                                             componentsPerVector: 3,
                                             bytesPerComponent: MemoryLayout<Float>.size,
                                             dataOffset: 0,
                                             dataStride: MemoryLayout<Float>.size * 3))
            material.diffuse.contents = UIColor.white
        }

        let element = SCNGeometryElement(indices: mesh.indices, primitiveType: .triangles)
        let geometry = SCNGeometry(sources: sources, elements: [element])
        geometry.materials = [material]
        return geometry
    }

    // MARK: - JSON

    static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: url, options: .atomic)
    }

    /// Nombre válido para OBJ/USD: sin espacios ni acentos.
    static func objectName(_ name: String) -> String {
        let plain = name.folding(options: .diacriticInsensitive, locale: nil)
        let allowed = plain.map { $0.isASCII && ($0.isLetter || $0.isNumber) ? $0 : "_" }
        let result = String(allowed)
        return result.first?.isNumber == true ? "_" + result : result
    }
}

private extension Array where Element == UInt8 {
    @inline(__always)
    mutating func append(float value: Float) {
        Swift.withUnsafeBytes(of: value.bitPattern.littleEndian) { append(contentsOf: $0) }
    }

    @inline(__always)
    mutating func append(uint32 value: UInt32) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}
