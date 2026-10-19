import Foundation
import SceneKit
import UIKit

/// Exportadores de malla/nube de puntos a color.
///  • PLY binario: el más fiel (Blender, MeshLab, CloudCompare). Incluye normales y color por vértice.
///  • OBJ con color por vértice (`v x y z r g b`): cada parte es un objeto `o` independiente (Blender lo
///    importa como objetos separados que se pueden mover, borrar o editar).
///  • USDZ: vista previa en el iPhone (Vista rápida / AR); cada parte es un nodo con nombre.
///  • JSON: inventario de la escena (nombre, categoría, centro, medidas) para procesarla con otros programas.
///
/// Ojo: `SIMD3<Float>` ocupa 16 bytes en memoria, así que los valores se escriben componente a componente.
enum ModelExporter {
    struct Part {
        let name: String
        let mesh: ColoredMesh
    }

    // MARK: - PLY

    static func writePLY(mesh: ColoredMesh, to url: URL) throws {
        var bytes = [UInt8](plyHeader(vertexCount: mesh.positions.count, faceCount: mesh.triangleCount).utf8)
        bytes.reserveCapacity(bytes.count + mesh.positions.count * 27 + mesh.triangleCount * 13)
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
        var bytes = [UInt8](plyHeader(vertexCount: mesh.positions.count, faceCount: nil).utf8)
        bytes.reserveCapacity(bytes.count + mesh.positions.count * 27)
        appendVertices(of: mesh, to: &bytes)
        try Data(bytes).write(to: url, options: .atomic)
    }

    private static func plyHeader(vertexCount: Int, faceCount: Int?) -> String {
        var header = """
        ply
        format binary_little_endian 1.0
        comment Escaner LiDAR - iPhone
        element vertex \(vertexCount)
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property uchar red
        property uchar green
        property uchar blue

        """
        if let faceCount {
            header += "element face \(faceCount)\nproperty list uchar uint vertex_indices\n"
        }
        return header + "end_header\n"
    }

    private static func appendVertices(of mesh: ColoredMesh, to bytes: inout [UInt8]) {
        for i in mesh.positions.indices {
            let p = mesh.positions[i], n = mesh.normals[i], c = mesh.colors[i]
            bytes.append(float: p.x); bytes.append(float: p.y); bytes.append(float: p.z)
            bytes.append(float: n.x); bytes.append(float: n.y); bytes.append(float: n.z)
            bytes.append(c.x); bytes.append(c.y); bytes.append(c.z)
        }
    }

    // MARK: - OBJ

    static func writeOBJ(mesh: ColoredMesh, to url: URL) throws {
        try writeOBJ(parts: [Part(name: "Escaneo", mesh: mesh)], to: url)
    }

    static func writeOBJ(parts: [Part], to url: URL) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw ScanError.exportFailed("OBJ")
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }

        var text = "# Escaner LiDAR - iPhone\n# Unidades: metros, eje Y hacia arriba\n"
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
            let name = objectName(part.name)
            text += "o \(name)\ng \(name)\n"
            for i in mesh.positions.indices {
                let p = mesh.positions[i], c = mesh.colors[i]
                text += String(format: "v %.5f %.5f %.5f %.4f %.4f %.4f\n",
                               p.x, p.y, p.z, Float(c.x) / 255, Float(c.y) / 255, Float(c.z) / 255)
                try flushIfNeeded()
            }
            for n in mesh.normals {
                text += String(format: "vn %.4f %.4f %.4f\n", n.x, n.y, n.z)
                try flushIfNeeded()
            }
            for t in 0..<mesh.triangleCount {
                let a = mesh.indices[t * 3] + offset, b = mesh.indices[t * 3 + 1] + offset, c = mesh.indices[t * 3 + 2] + offset
                text += "f \(a)//\(a) \(b)//\(b) \(c)//\(c)\n"
                try flushIfNeeded()
            }
            offset += UInt32(mesh.positions.count)
        }
        try flushIfNeeded(force: true)
    }

    // MARK: - USDZ

    static func writeUSDZ(mesh: ColoredMesh, to url: URL) throws {
        try writeUSDZ(parts: [Part(name: "Escaneo", mesh: mesh)], to: url)
    }

    static func writeUSDZ(parts: [Part], to url: URL) throws {
        let scene = SCNScene()
        for part in parts where !part.mesh.isEmpty {
            let node = SCNNode(geometry: makeGeometry(part.mesh))
            node.name = objectName(part.name)
            scene.rootNode.addChildNode(node)
        }
        guard scene.write(to: url, options: nil, delegate: nil, progressHandler: nil) else {
            throw ScanError.exportFailed("USDZ")
        }
    }

    private static func makeGeometry(_ mesh: ColoredMesh) -> SCNGeometry {
        let vertexSource = SCNGeometrySource(vertices: mesh.positions.map { SCNVector3($0.x, $0.y, $0.z) })
        let normalSource = SCNGeometrySource(normals: mesh.normals.map { SCNVector3($0.x, $0.y, $0.z) })

        var packedColors = [Float]()
        packedColors.reserveCapacity(mesh.colors.count * 3)
        for c in mesh.colors {
            packedColors.append(Float(c.x) / 255)
            packedColors.append(Float(c.y) / 255)
            packedColors.append(Float(c.z) / 255)
        }
        let colorData = packedColors.withUnsafeBufferPointer { Data(buffer: $0) }
        let colorSource = SCNGeometrySource(data: colorData,
                                            semantic: .color,
                                            vectorCount: mesh.colors.count,
                                            usesFloatComponents: true,
                                            componentsPerVector: 3,
                                            bytesPerComponent: MemoryLayout<Float>.size,
                                            dataOffset: 0,
                                            dataStride: MemoryLayout<Float>.size * 3)

        let element = SCNGeometryElement(indices: mesh.indices, primitiveType: .triangles)
        let geometry = SCNGeometry(sources: [vertexSource, normalSource, colorSource], elements: [element])
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.white
        material.lightingModel = .physicallyBased
        material.roughness.contents = 0.9
        material.isDoubleSided = true
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
