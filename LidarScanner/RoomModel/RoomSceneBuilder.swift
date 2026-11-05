import SceneKit
import UIKit
import simd

/// Nodos SceneKit de la habitación: paredes con sus medidas, puertas y ventanas, y cada mueble como su modelo
/// escaneado en detalle o, si aún no lo tiene, como sus piezas. Lo usan el editor y la exportación a USDZ.
@MainActor
final class RoomSceneBuilder {
    let folder: URL
    private var models: [String: SCNNode] = [:]

    init(folder: URL) {
        self.folder = folder
    }

    /// Modelo de Object Capture ya cargado; se clona para no leer el USDZ en cada redibujado.
    func detailNode(_ detail: DetailModel) -> SCNNode? {
        if models[detail.file] == nil {
            models[detail.file] = DetailFitter.loadNode(folder.appendingPathComponent(detail.file))
        }
        guard let node = models[detail.file]?.clone() else { return nil }
        node.simdTransform = detail.transform
        return node
    }

    func forget(_ file: String) {
        models[file] = nil
    }

    static func label(_ text: String, at position: SIMD3<Float>, size: Float = 0.07) -> SCNNode {
        let geometry = SCNText(string: text, extrusionDepth: 0)
        geometry.font = .systemFont(ofSize: 1, weight: .semibold)
        geometry.flatness = 0.05
        geometry.firstMaterial?.diffuse.contents = UIColor.white
        geometry.firstMaterial?.lightingModel = .constant
        geometry.firstMaterial?.readsFromDepthBuffer = false
        let text = SCNNode(geometry: geometry)
        let (lo, hi) = text.boundingBox
        text.pivot = SCNMatrix4MakeTranslation((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, 0)
        text.scale = SCNVector3(size, size, size)
        let node = SCNNode()
        node.addChildNode(text)
        node.simdPosition = position
        node.constraints = [SCNBillboardConstraint()]
        node.renderingOrder = 10
        return node
    }

    static func meters(_ value: Float) -> String {
        value.formatted(.number.precision(.fractionLength(2))) + " m"
    }

    /// Cono naranja en el punto desde el que conviene volver a grabar, apuntando a lo que falla.
    static func viewpoint(eye: SIMD3<Float>, target: SIMD3<Float>, text: String) -> [SCNNode] {
        let cone = SCNCone(topRadius: 0, bottomRadius: 0.09, height: 0.22)
        cone.firstMaterial?.diffuse.contents = UIColor.systemOrange
        cone.firstMaterial?.lightingModel = .constant
        let tip = SCNNode(geometry: cone)
        tip.eulerAngles.x = -.pi / 2                          // la punta del cono hacia −z, como mira una cámara
        let node = SCNNode()
        node.addChildNode(tip)
        node.simdPosition = eye
        node.simdLook(at: target)
        let line = SCNGeometry(sources: [SCNGeometrySource(vertices: [SCNVector3(eye), SCNVector3(target)])],
                               elements: [SCNGeometryElement(indices: [UInt32(0), 1], primitiveType: .line)])
        line.firstMaterial?.diffuse.contents = UIColor.systemOrange
        line.firstMaterial?.lightingModel = .constant
        return [node, SCNNode(geometry: line), label(text, at: eye + SIMD3(0, 0.2, 0), size: 0.05)]
    }

    func nodes(for room: RoomModel, selectedID: UUID?, labels: Bool) -> [SCNNode] {
        var result: [SCNNode] = []
        if labels {
            // Solo lo que falla de verdad (rojo), o lo seleccionado, para no llenar la vista de conos.
            for (i, wall) in room.walls.enumerated() {
                if let v = wall.validation, v.state == .red, let eye = v.suggestedEye, let target = v.suggestedTarget {
                    result += Self.viewpoint(eye: eye, target: target, text: "Graba la pared \(i + 1) desde aquí")
                }
            }
            for f in room.furniture where f.detail == nil {
                if let v = f.validation, v.state == .red || f.id == selectedID,
                   let eye = v.suggestedEye, let target = v.suggestedTarget {
                    result += Self.viewpoint(eye: eye, target: target, text: "\(f.name): grábalo desde aquí")
                }
            }
        }

        for (i, wall) in room.walls.enumerated() {
            let box = SCNBox(width: CGFloat(wall.width), height: CGFloat(wall.height), length: 0.04, chamferRadius: 0)
            box.firstMaterial?.diffuse.contents = labels && wall.validation != nil
                ? Self.color(for: wall.validation!.state).withAlphaComponent(0.3)
                : UIColor(white: 0.85, alpha: labels ? 0.35 : 1)
            box.firstMaterial?.isDoubleSided = true
            let node = SCNNode(geometry: box)
            node.name = "Pared_\(i + 1)"
            node.simdPosition = wall.center - wall.normal * 0.02
            node.simdEulerAngles = SIMD3(0, atan2(wall.normal.x, wall.normal.z), 0)
            result.append(node)
            if labels {
                let top = SIMD3(wall.center.x, room.floorY + wall.height + 0.08, wall.center.z)
                let checks = wall.validation.map { " · \($0.passed)/\($0.checks.count) ✓" } ?? ""
                result.append(Self.label("\(Self.meters(wall.width)) × \(Self.meters(wall.height))\(checks)", at: top))
            }
        }

        for (i, opening) in room.openings.enumerated() {
            let box = SCNBox(width: CGFloat(opening.width), height: CGFloat(opening.height), length: 0.05, chamferRadius: 0)
            let color: UIColor = switch opening.kind {
            case .door: UIColor(red: 0.55, green: 0.38, blue: 0.24, alpha: 1)
            case .window: UIColor(red: 0.45, green: 0.7, blue: 0.95, alpha: 0.6)
            case .opening: UIColor(white: 0.2, alpha: 0.8)
            }
            box.firstMaterial?.diffuse.contents = color
            let node = SCNNode(geometry: box)
            node.name = "\(opening.kind.displayName)_\(i + 1)"
            node.simdPosition = opening.center + opening.normal * 0.005
            node.simdEulerAngles = SIMD3(0, atan2(opening.normal.x, opening.normal.z), 0)
            result.append(node)
            if labels {
                result.append(Self.label("\(opening.kind.displayName) \(Self.meters(opening.width)) × \(Self.meters(opening.height))",
                                         at: opening.center + opening.normal * 0.06, size: 0.05))
            }
        }

        for furniture in room.furniture {
            let group = SCNNode()
            group.name = furniture.id.uuidString
            group.simdTransform = furniture.localToWorld
            let selected = furniture.id == selectedID
            if let detail = furniture.detail, let model = detailNode(detail) {
                group.addChildNode(model)
                if selected {
                    // Caja de selección alrededor del objeto real.
                    let s = furniture.size
                    let box = SCNBox(width: CGFloat(s.x), height: CGFloat(s.y), length: CGFloat(s.z), chamferRadius: 0)
                    box.firstMaterial?.diffuse.contents = UIColor.cyan.withAlphaComponent(0.18)
                    box.firstMaterial?.lightingModel = .constant
                    box.firstMaterial?.writesToDepthBuffer = false
                    let node = SCNNode(geometry: box)
                    node.simdPosition = SIMD3(0, s.y / 2, 0)
                    group.addChildNode(node)
                }
            } else {
                let color = Self.color(for: furniture.state)
                for part in furniture.parts {
                    let box = SCNBox(width: CGFloat(part.size.x), height: CGFloat(part.size.y), length: CGFloat(part.size.z),
                                     chamferRadius: 0.003)
                    box.firstMaterial?.diffuse.contents = color
                    box.firstMaterial?.emission.contents = selected ? UIColor(white: 0.25, alpha: 1) : UIColor.black
                    let node = SCNNode(geometry: box)
                    node.name = ModelExporter.objectName(part.name)
                    node.simdPosition = part.center
                    group.addChildNode(node)
                }
            }
            result.append(group)
            if labels && selected {
                let s = furniture.size
                let text = "\(Int((s.x * 100).rounded())) × \(Int((s.y * 100).rounded())) × \(Int((s.z * 100).rounded())) cm"
                result.append(Self.label(text, at: furniture.position + SIMD3(0, s.y + 0.12, 0), size: 0.06))
            }
        }
        return result
    }

    static func color(for state: VerificationState) -> UIColor {
        switch state {
        case .red: UIColor.systemRed
        case .yellow: UIColor.systemYellow
        case .green: UIColor.systemGreen
        }
    }

    /// Toda la habitación en un USDZ: paredes, huecos y muebles (con su malla real si se escanearon en detalle).
    func writeUSDZ(_ room: RoomModel, to url: URL) throws {
        let scene = SCNScene()
        for node in nodes(for: room, selectedID: nil, labels: false) {
            if let name = node.name, UUID(uuidString: name) != nil,
               let f = room.furniture.first(where: { $0.id.uuidString == name }) {
                node.name = ModelExporter.objectName(f.name)
            }
            scene.rootNode.addChildNode(node)
        }
        guard scene.write(to: url, options: nil, delegate: nil, progressHandler: nil) else {
            throw ScanError.exportFailed("USDZ")
        }
    }
}
