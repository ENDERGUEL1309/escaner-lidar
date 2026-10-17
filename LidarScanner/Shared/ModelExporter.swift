import Foundation
import SceneKit
import UIKit

/// Exportadores de malla/nube de puntos a color.
///  • PLY binario: el más fiel (Blender, MeshLab, CloudCompare). Incluye normales y color por vértice.
///  • OBJ con color por vértice (`v x y z r g b`): extensión muy extendida (Blender, MeshLab, ZBrush).
///  • USDZ: vista previa en el iPhone (Vista rápida / AR) y compatible con el ecosistema Apple.
///
/// Ojo: `SIMD3<Float>` ocupa 16 bytes en memoria, así que los valores se escriben componente a componente.
enum ModelExporter {
    static func writePLY(mesh: ColoredMesh, to url: URL) throws {
        let header = """
        ply
        format binary_little_endian 1.0
        comment Escaner LiDAR - iPhone
        element vertex \(mesh.positions.count)
        property float x
        property float y
        property float z
        property float nx
        property float ny
        property float nz
        property uchar red
        property uchar green
        property uchar blue
        element face \(mesh.triangleCount)
        property list uchar uint vertex_indices
        end_header

        """
        var bytes = [UInt8](header.utf8)
        bytes.reserveCapacity(bytes.count + mesh.positions.count * 27 + mesh.triangleCount * 13)
        for i in mesh.positions.indices {
            let p = mesh.positions[i], n = mesh.normals[i], c = mesh.colors[i]
            bytes.append(float: p.x); bytes.append(float: p.y); bytes.append(float: p.z)
            bytes.append(float: n.x); bytes.append(float: n.y); bytes.append(float: n.z)
            bytes.append(c.x); bytes.append(c.y); bytes.append(c.z)
        }
        for t in 0..<mesh.triangleCount {
            bytes.append(3)
            bytes.append(uint32: mesh.indices[t * 3])
            bytes.append(uint32: mesh.indices[t * 3 + 1])
            bytes.append(uint32: mesh.indices[t * 3 + 2])
        }
        try Data(bytes).write(to: url, options: .atomic)
    }

    static func writePLY(points: [ColoredPoint], to url: URL) throws {
        let header = """
        ply
        format binary_little_endian 1.0
        comment Escaner LiDAR - iPhone
        element vertex \(points.count)
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        var bytes = [UInt8](header.utf8)
        bytes.reserveCapacity(bytes.count + points.count * 15)
        for pt in points {
            bytes.append(float: pt.position.x); bytes.append(float: pt.position.y); bytes.append(float: pt.position.z)
            bytes.append(pt.color.x); bytes.append(pt.color.y); bytes.append(pt.color.z)
        }
        try Data(bytes).write(to: url, options: .atomic)
    }

    static func writeOBJ(mesh: ColoredMesh, to url: URL) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw ScanError.exportFailed("OBJ")
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }

        var text = "# Escaner LiDAR - iPhone\n# \(mesh.positions.count) vertices, \(mesh.triangleCount) triangulos\n"
        text.reserveCapacity(1 << 21)
        func flushIfNeeded(force: Bool = false) throws {
            if force || text.utf8.count > (1 << 20) {
                try handle.write(contentsOf: Data(text.utf8))
                text.removeAll(keepingCapacity: true)
            }
        }

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
            let a = mesh.indices[t * 3] + 1, b = mesh.indices[t * 3 + 1] + 1, c = mesh.indices[t * 3 + 2] + 1
            text += "f \(a)//\(a) \(b)//\(b) \(c)//\(c)\n"
            try flushIfNeeded()
        }
        try flushIfNeeded(force: true)
    }

    static func writeUSDZ(mesh: ColoredMesh, to url: URL) throws {
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

        let scene = SCNScene()
        scene.rootNode.addChildNode(SCNNode(geometry: geometry))
        guard scene.write(to: url, options: nil, delegate: nil, progressHandler: nil) else {
            throw ScanError.exportFailed("USDZ")
        }
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
