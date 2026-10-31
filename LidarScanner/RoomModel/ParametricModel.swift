import Foundation
import simd

/// Rojo: faltan vistas. Amarillo: modelado, sin superar las 4 pruebas. Verde: las superó todas.
enum VerificationState: String, Codable {
    case red, yellow, green

    var title: String {
        switch self {
        case .red: "Faltan vistas"
        case .yellow: "Pendiente de verificar"
        case .green: "Verificado"
        }
    }
}

/// Resultado de las 4 pruebas. `nil` = no medido: la regla de neutralidad impide contarlo a favor o en contra.
struct VerificationReport: Codable {
    var views = 0
    var silhouetteIoU: Float?
    var geometryRMSE: Float?
    var freeSpaceViolations: Int?
    /// `nil` también cuando el mueble no tiene variantes que distinguir.
    var aspectConsistent: Bool?
    var aspectApplies = false
}

enum FurnitureKind: String, Codable {
    case table, desk, bed, sofa, chair, storage, appliance

    var displayName: String {
        switch self {
        case .table: "Mesa"
        case .desk: "Escritorio"
        case .bed: "Cama"
        case .sofa: "Sofá"
        case .chair: "Silla"
        case .storage: "Armario"
        case .appliance: "Electrodoméstico"
        }
    }

    /// Muebles con siluetas irregulares: la meta de IoU es 0,75 en lugar de 0,90.
    var isComplex: Bool { self == .bed || self == .sofa || self == .chair }
    var snapsToWall: Bool { self == .desk || self == .bed || self == .storage || self == .sofa }
}

/// Lo que distingue a dos muebles de silueta idéntica; lo decide la prueba de aspecto.
enum FurnitureVariant: String, Codable {
    case standard
    case deskWithDrawers, deskOpen
    case storageDoors, storageOpen
}

/// Caja de una pieza, en coordenadas del mueble: origen en el suelo, en el centro de la planta; +z es el frente.
struct FurniturePart: Codable, Identifiable {
    var id = UUID()
    var name: String
    var center: SIMD3<Float>
    var size: SIMD3<Float>
}

struct ParametricFurniture: Codable, Identifiable {
    var id = UUID()
    var name: String
    var kind: FurnitureKind
    var variant: FurnitureVariant
    /// Centro de la planta, a la altura del suelo.
    var position: SIMD3<Float>
    var yaw: Float
    /// Ancho (x), alto (y) y fondo (z).
    var size: SIMD3<Float>
    var parts: [FurniturePart] = []
    var state: VerificationState = .red
    var report = VerificationReport()

    init(name: String, kind: FurnitureKind, position: SIMD3<Float>, yaw: Float, size: SIMD3<Float>) {
        self.name = name
        self.kind = kind
        self.variant = kind == .desk ? .deskWithDrawers : kind == .storage ? .storageDoors : .standard
        self.position = position
        self.yaw = yaw
        self.size = size
        rebuildParts()
    }

    var localToWorld: simd_float4x4 {
        var m = simd_float4x4(simd_quatf(angle: yaw, axis: SIMD3(0, 1, 0)))
        m.columns.3 = SIMD4(position, 1)
        return m
    }

    var worldToLocal: simd_float4x4 { localToWorld.inverse }

    mutating func rebuildParts() {
        parts = FurnitureBuilder.parts(kind: kind, variant: variant, size: size)
    }
}

enum FurnitureBuilder {
    private static func part(_ name: String, _ min: SIMD3<Float>, _ max: SIMD3<Float>) -> FurniturePart {
        FurniturePart(name: name, center: (min + max) / 2, size: max - min)
    }

    static func parts(kind: FurnitureKind, variant: FurnitureVariant, size: SIMD3<Float>) -> [FurniturePart] {
        let w = size.x / 2, h = size.y, d = size.z / 2
        switch kind {
        case .table:
            let t = min(0.04, h * 0.1), leg = min(0.07, size.x * 0.1)
            return [part("Tablero", SIMD3(-w, h - t, -d), SIMD3(w, h, d))] + legs(w: w, d: d, height: h - t, side: leg)
        case .desk:
            let t = min(0.04, h * 0.1), panel: Float = 0.03
            let top = part("Tablero", SIMD3(-w, h - t, -d), SIMD3(w, h, d))
            let left = part("Lateral", SIMD3(-w, 0, -d), SIMD3(-w + panel, h - t, d))
            if variant == .deskOpen {
                return [top, left, part("Lateral derecho", SIMD3(w - panel, 0, -d), SIMD3(w, h - t, d))]
            }
            let block = Swift.min(0.45, size.x * 0.4)
            return [top, left, part("Cajonera", SIMD3(w - block, 0, -d), SIMD3(w, h - t, d))]
        case .bed:
            let board = min(0.08, size.z * 0.05), mattress = min(0.25, h * 0.45)
            return [part("Cabecero", SIMD3(-w, 0, -d), SIMD3(w, h, -d + board)),
                    part("Canapé", SIMD3(-w, 0, -d + board), SIMD3(w, h * 0.5 - mattress / 2, d)),
                    part("Colchón", SIMD3(-w, h * 0.5 - mattress / 2, -d + board), SIMD3(w, h * 0.5 + mattress / 2, d))]
        case .sofa:
            let seat = h * 0.5, back = min(0.22, size.z * 0.28), arm = min(0.2, size.x * 0.12)
            return [part("Asiento", SIMD3(-w + arm, 0, -d + back), SIMD3(w - arm, seat, d)),
                    part("Respaldo", SIMD3(-w, 0, -d), SIMD3(w, h, -d + back)),
                    part("Brazo izquierdo", SIMD3(-w, 0, -d + back), SIMD3(-w + arm, seat * 1.3, d)),
                    part("Brazo derecho", SIMD3(w - arm, 0, -d + back), SIMD3(w, seat * 1.3, d))]
        case .chair:
            let seat = h * 0.52, t: Float = 0.05, leg = min(0.05, size.x * 0.1), back: Float = 0.05
            return [part("Asiento", SIMD3(-w, seat - t, -d), SIMD3(w, seat, d)),
                    part("Respaldo", SIMD3(-w, seat, -d), SIMD3(w, h, -d + back))]
                + legs(w: w, d: d, height: seat - t, side: leg)
        case .storage:
            if variant == .storageOpen {
                let t: Float = 0.02, shelves = max(2, Int(h / 0.35))
                var result = [part("Lateral izquierdo", SIMD3(-w, 0, -d), SIMD3(-w + t, h, d)),
                              part("Lateral derecho", SIMD3(w - t, 0, -d), SIMD3(w, h, d)),
                              part("Trasera", SIMD3(-w, 0, -d), SIMD3(w, h, -d + t))]
                for i in 0...shelves {
                    let y = (h - t) * Float(i) / Float(shelves)
                    result.append(part("Balda \(i + 1)", SIMD3(-w, y, -d), SIMD3(w, y + t, d)))
                }
                return result
            }
            return [part("Cuerpo", SIMD3(-w, 0, -d), SIMD3(w, h, d - 0.02)),
                    part("Puertas", SIMD3(-w, 0, d - 0.02), SIMD3(w, h, d))]
        case .appliance:
            return [part("Cuerpo", SIMD3(-w, 0, -d), SIMD3(w, h, d))]
        }
    }

    private static func legs(w: Float, d: Float, height: Float, side: Float) -> [FurniturePart] {
        let signs: [(Float, Float)] = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
        return signs.enumerated().map { i, sign -> FurniturePart in
            let c = SIMD3<Float>(sign.0 * (w - side / 2), height / 2, sign.1 * (d - side / 2))
            return FurniturePart(name: "Pata \(i + 1)", center: c, size: SIMD3<Float>(side, height, side))
        }
    }
}

struct RoomWall: Codable, Identifiable {
    var id = UUID()
    var center: SIMD3<Float>
    var width: Float
    var height: Float
    /// Normal hacia el interior de la habitación.
    var normal: SIMD3<Float>

    /// Dirección a lo largo de la pared.
    var tangent: SIMD3<Float> { simd_normalize(simd_cross(SIMD3(0, 1, 0), normal)) }
}

struct RoomModel: Codable {
    var version = 1
    var unidades = "metros"
    var floorY: Float = 0
    var walls: [RoomWall] = []
    var furniture: [ParametricFurniture] = []

    private static func footprint(_ f: ParametricFurniture) -> [SIMD3<Float>] {
        let m = f.localToWorld
        let signs: [(Float, Float)] = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        return signs.map { sign -> SIMD3<Float> in
            let p = m * SIMD4<Float>(sign.0 * f.size.x / 2, 0, sign.1 * f.size.z / 2, 1)
            return SIMD3(p.x, p.y, p.z)
        }
    }

    /// Restricción física: los muebles no atraviesan paredes. Empuja el mueble hacia dentro lo que haya penetrado.
    func keepInside(_ furniture: ParametricFurniture) -> ParametricFurniture {
        var f = furniture
        for wall in walls {
            let half = wall.width / 2 + 0.05
            var deepest: Float = 0
            for corner in Self.footprint(f) {
                let offset = corner - wall.center
                guard abs(simd_dot(offset, wall.tangent)) < half else { continue }
                let distance = simd_dot(offset, wall.normal)
                if distance < 0, distance > -0.6 { deepest = min(deepest, distance) }
            }
            f.position -= wall.normal * deepest
        }
        return f
    }

    /// Escritorios, armarios, sofás y cabeceros se alinean con la normal de la pared más próxima y quedan pegados.
    func snappedToWall(_ furniture: ParametricFurniture, maxGap: Float = 0.25, maxAngle: Float = .pi / 12) -> ParametricFurniture {
        guard furniture.kind.snapsToWall else { return keepInside(furniture) }
        var best: (wall: RoomWall, gap: Float, yaw: Float, swapped: Bool)?
        for wall in walls {
            let wallYaw = atan2(wall.normal.x, wall.normal.z)
            // Eje del mueble que mira a esta pared: su giro relativo redondeado a 90° dice cuál es.
            var delta = (furniture.yaw - wallYaw).truncatingRemainder(dividingBy: .pi * 2)
            if delta < 0 { delta += .pi * 2 }
            let quarter = Int((delta / (.pi / 2)).rounded()) % 4
            let residual = abs(delta - Float(quarter) * .pi / 2)
            guard min(residual, .pi * 2 - residual) <= maxAngle else { continue }
            let swapped = quarter % 2 == 1
            let halfDepth = (swapped ? furniture.size.x : furniture.size.z) / 2
            let offset = furniture.position - wall.center
            guard abs(simd_dot(offset, wall.tangent)) < wall.width / 2 + 0.1 else { continue }
            let gap = simd_dot(offset, wall.normal) - halfDepth
            guard gap > -maxGap, gap < maxGap, best == nil || abs(gap) < abs(best!.gap) else { continue }
            best = (wall, gap, wallYaw, swapped)
        }
        guard let best else { return keepInside(furniture) }
        var f = furniture
        if best.swapped {
            f.size = SIMD3(f.size.z, f.size.y, f.size.x)
        }
        f.yaw = best.yaw
        f.position -= best.wall.normal * best.gap
        f.rebuildParts()
        return keepInside(f)
    }
}
