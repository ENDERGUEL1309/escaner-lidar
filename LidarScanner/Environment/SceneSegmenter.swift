import ARKit
import simd

/// Clases que la red neuronal de ARKit asigna a cada cara de la malla LiDAR (mismo `rawValue` que `ARMeshClassification`).
enum SurfaceLabel: UInt8 {
    case none = 0, wall, floor, ceiling, table, seat, window, door

    var isStructure: Bool {
        switch self {
        case .wall, .floor, .ceiling, .window, .door: true
        default: false
        }
    }

    /// Nombre del grupo (estructura) o prefijo del objeto.
    var displayName: String {
        switch self {
        case .wall: "Paredes"
        case .floor: "Suelo"
        case .ceiling: "Techo"
        case .window: "Ventanas"
        case .door: "Puertas"
        case .table: "Mesa"
        case .seat: "Asiento"
        case .none: "Objeto"
        }
    }

    var category: String {
        switch self {
        case .wall: "pared"
        case .floor: "suelo"
        case .ceiling: "techo"
        case .window: "ventana"
        case .door: "puerta"
        case .table: "mesa"
        case .seat: "asiento"
        case .none: "objeto"
        }
    }
}

/// Busca la clase de ARKit más cercana a un punto, a partir de los centros de las caras clasificadas.
struct SurfaceClassifier {
    private let cellSize: Float = 0.1
    private var grid: [Int64: [(SIMD3<Float>, UInt8)]] = [:]

    init(anchors: [ARMeshAnchor]) {
        for anchor in anchors {
            let geometry = anchor.geometry
            guard let classification = geometry.classification else { continue }
            let transform = anchor.transform
            let faces = geometry.faces
            let perFace = faces.indexCountPerPrimitive
            let raw = faces.buffer.contents()
            let labels = classification.buffer.contents()

            for f in 0..<faces.count {
                var centroid = SIMD3<Float>(repeating: 0)
                for k in 0..<perFace {
                    let i = f * perFace + k
                    let vertex = faces.bytesPerIndex == 4
                        ? Int(raw.load(fromByteOffset: i * 4, as: UInt32.self))
                        : Int(raw.load(fromByteOffset: i * 2, as: UInt16.self))
                    centroid += geometry.vertices.float3(at: vertex)
                }
                centroid /= Float(perFace)
                let world = transform * SIMD4<Float>(centroid, 1)
                let p = SIMD3(world.x, world.y, world.z)
                let label = labels.load(fromByteOffset: classification.offset + classification.stride * f, as: UInt8.self)
                grid[key(p), default: []].append((p, label))
            }
        }
    }

    var isEmpty: Bool { grid.isEmpty }

    func label(at p: SIMD3<Float>) -> SurfaceLabel {
        let c = cell(p)
        var best: Float = cellSize * cellSize
        var result: UInt8 = 0
        for dz in -1...1 {
            for dy in -1...1 {
                for dx in -1...1 {
                    guard let bucket = grid[pack(c &+ SIMD3(Int32(dx), Int32(dy), Int32(dz)))] else { continue }
                    for (q, label) in bucket {
                        let d = simd_distance_squared(p, q)
                        if d < best { best = d; result = label }
                    }
                }
            }
        }
        return SurfaceLabel(rawValue: result) ?? .none
    }

    private func cell(_ p: SIMD3<Float>) -> SIMD3<Int32> {
        let q = (p / cellSize).rounded(.down)
        return SIMD3(Int32(q.x), Int32(q.y), Int32(q.z))
    }

    private func key(_ p: SIMD3<Float>) -> Int64 { pack(cell(p)) }

    private func pack(_ c: SIMD3<Int32>) -> Int64 {
        let mask: Int64 = 0x1FFFFF
        return (Int64(c.x) & mask) | ((Int64(c.y) & mask) << 21) | ((Int64(c.z) & mask) << 42)
    }
}

struct SceneSegment {
    enum Kind { case structure, object, fragments }
    var name: String
    let label: SurfaceLabel
    let kind: Kind
    let mesh: ColoredMesh
    /// Índices de los triángulos de la malla original que forman este segmento.
    let triangles: [Int]
}

/// Datos de cada objeto para el archivo `escena.json` y la interfaz.
struct SceneObjectInfo: Codable, Hashable, Identifiable {
    var id: String { nombre }
    let nombre: String
    let categoria: String
    let estructura: Bool
    /// Centro de la caja orientada, en metros (Y hacia arriba).
    let centro: [Float]
    /// Ancho, alto y fondo de la caja orientada, en metros.
    let dimensiones: [Float]
    /// Giro de la caja alrededor del eje vertical, en grados.
    let rotacionY: Float
    let vertices: Int
    let triangulos: Int
    /// Ruta relativa a la carpeta del escaneo del PLY de este objeto (p. ej. "objetos/Silla_1.ply").
    let archivo: String
    /// Lo que la IA de visión reconoció en las fotos (nil si no estaba segura).
    var etiquetaIA: String? = nil
    var confianzaIA: Float? = nil

    var dimensionsText: String {
        dimensiones.map { $0.formatted(.number.precision(.fractionLength(2))) }.joined(separator: " × ") + " m"
    }

    var detailText: String {
        guard let confianzaIA else { return dimensionsText }
        return "\(dimensionsText) · IA \(Int(confianzaIA * 100)) %"
    }
}

struct SceneDescription: Codable {
    var version = 1
    var unidades = "metros"
    var ejeVertical = "Y"
    var objetos: [SceneObjectInfo]
}

/// Divide la malla de la escena en partes editables:
///  1. Cada vértice recibe la clase de ARKit más cercana (pared, suelo, mesa, asiento…).
///  2. Paredes, suelo, techo, puertas y ventanas forman un grupo cada uno.
///  3. El resto se separa en objetos por conectividad: al quitar suelo y paredes, cada mueble queda aislado.
///  4. Cada objeto se nombra por la clase mayoritaria de sus vértices.
enum SceneSegmenter {
    /// Clase de ARKit de cada vértice de la malla.
    static func labels(for mesh: ColoredMesh, classifier: SurfaceClassifier) -> [SurfaceLabel] {
        let n = mesh.positions.count
        var labels = [SurfaceLabel](repeating: .none, count: n)
        let positions = mesh.positions
        let chunkSize = 4096
        labels.withUnsafeMutableBufferPointer { out in
            DispatchQueue.concurrentPerform(iterations: (n + chunkSize - 1) / chunkSize) { chunk in
                for i in (chunk * chunkSize)..<min(n, (chunk + 1) * chunkSize) {
                    out[i] = classifier.label(at: positions[i])
                }
            }
        }
        return labels
    }

    /// `signatures`: separación de instancias de la IA de visión; si se pasa, los vértices que la IA ve
    /// como objetos distintos no se unen aunque estén pegados.
    static func segment(_ mesh: ColoredMesh, labels: [SurfaceLabel], signatures: InstanceSignatures? = nil,
                        minObjectVertices: Int = 300) -> [SceneSegment] {
        let n = mesh.positions.count
        guard n > 0, labels.count == n else { return [] }

        // Componentes conexas (union-find) de los vértices que no son estructura.
        var parent = (0..<n).map { Int32($0) }
        func find(_ x: Int32) -> Int32 {
            var x = x
            while parent[Int(x)] != x {
                parent[Int(x)] = parent[Int(parent[Int(x)])]
                x = parent[Int(x)]
            }
            return x
        }
        func union(_ a: UInt32, _ b: UInt32) {
            guard !labels[Int(a)].isStructure, !labels[Int(b)].isStructure else { return }
            if let signatures, signatures.separates(Int(a), Int(b)) { return }
            let ra = find(Int32(a)), rb = find(Int32(b))
            if ra != rb { parent[Int(ra)] = rb }
        }
        let tri = mesh.indices
        for t in stride(from: 0, to: tri.count, by: 3) {
            union(tri[t], tri[t + 1])
            union(tri[t + 1], tri[t + 2])
            union(tri[t], tri[t + 2])
        }

        // Tamaño y votos de clase (mesa, asiento, otro) de cada componente.
        var componentSize: [Int32: Int] = [:]
        var votes: [Int32: SIMD3<Int32>] = [:]
        for i in 0..<n where !labels[i].isStructure {
            let root = find(Int32(i))
            componentSize[root, default: 0] += 1
            var v = votes[root] ?? .zero
            switch labels[i] {
            case .table: v.x += 1
            case .seat: v.y += 1
            default: v.z += 1
            }
            votes[root] = v
        }

        // Lista de segmentos: estructura primero, luego objetos de mayor a menor.
        var segments: [(name: String, label: SurfaceLabel, kind: SceneSegment.Kind)] = []
        var structureIndex: [SurfaceLabel: Int] = [:]
        let present = Set(labels)
        for label in [SurfaceLabel.floor, .wall, .ceiling, .door, .window] where present.contains(label) {
            structureIndex[label] = segments.count
            segments.append((label.displayName, label, .structure))
        }
        var componentIndex: [Int32: Int] = [:]
        var counters: [SurfaceLabel: Int] = [:]
        for (root, size) in componentSize.sorted(by: { $0.value > $1.value }) where size >= minObjectVertices {
            let v = votes[root] ?? .zero
            let total = Float(v.x + v.y + v.z)
            let label: SurfaceLabel
            if Float(v.x) >= 0.3 * total && v.x >= v.y { label = .table }
            else if Float(v.y) >= 0.3 * total { label = .seat }
            else { label = .none }
            counters[label, default: 0] += 1
            componentIndex[root] = segments.count
            segments.append(("\(label.displayName) \(counters[label]!)", label, .object))
        }

        var segmentOf = [Int32](repeating: 0, count: n)
        var fragmentsIndex: Int?
        for i in 0..<n {
            if labels[i].isStructure, let s = structureIndex[labels[i]] {
                segmentOf[i] = Int32(s)
            } else if let s = componentIndex[find(Int32(i))] {
                segmentOf[i] = Int32(s)
            } else {
                if fragmentsIndex == nil {
                    fragmentsIndex = segments.count
                    segments.append(("Fragmentos", .none, .fragments))
                }
                segmentOf[i] = Int32(fragmentsIndex!)
            }
        }

        // Cada triángulo va al segmento de la mayoría de sus vértices.
        var triangles = [[Int]](repeating: [], count: segments.count)
        for t in stride(from: 0, to: tri.count, by: 3) {
            let s0 = segmentOf[Int(tri[t])], s1 = segmentOf[Int(tri[t + 1])], s2 = segmentOf[Int(tri[t + 2])]
            triangles[Int(s1 == s2 ? s1 : s0)].append(t / 3)
        }

        // Submalla independiente por segmento (se guardan también los triángulos originales,
        // para recortar después la misma parte de la malla texturizada).
        var result: [SceneSegment] = []
        for (s, segment) in segments.enumerated() where !triangles[s].isEmpty {
            result.append(SceneSegment(name: segment.name, label: segment.label, kind: segment.kind,
                                       mesh: mesh.submesh(triangles: triangles[s]), triangles: triangles[s]))
        }
        return result
    }

    /// Caja orientada (giro solo en el eje vertical, como los muebles) para medir cada objeto.
    static func info(for segment: SceneSegment, file: String, recognition: ObjectRecognizer.Result? = nil) -> SceneObjectInfo {
        let points = segment.mesh.positions
        var mean = SIMD3<Float>(repeating: 0)
        for p in points { mean += p }
        mean /= Float(max(points.count, 1))

        var cxx: Float = 0, czz: Float = 0, cxz: Float = 0
        for p in points {
            let dx = p.x - mean.x, dz = p.z - mean.z
            cxx += dx * dx; czz += dz * dz; cxz += dx * dz
        }
        let angle = 0.5 * atan2(2 * cxz, cxx - czz)
        let cosA = cos(angle), sinA = sin(angle)

        var minP = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxP = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in points {
            let dx = p.x - mean.x, dz = p.z - mean.z
            let local = SIMD3<Float>(dx * cosA + dz * sinA, p.y, -dx * sinA + dz * cosA)
            minP = simd_min(minP, local)
            maxP = simd_max(maxP, local)
        }
        let size = maxP - minP
        let mid = (minP + maxP) / 2
        let center = SIMD3<Float>(mean.x + mid.x * cosA - mid.z * sinA,
                                  mid.y,
                                  mean.z + mid.x * sinA + mid.z * cosA)

        return SceneObjectInfo(nombre: segment.name,
                               categoria: segment.kind == .fragments ? "fragmentos" : segment.label.category,
                               estructura: segment.kind == .structure,
                               centro: [center.x, center.y, center.z],
                               dimensiones: [size.x, size.y, size.z],
                               rotacionY: angle * 180 / .pi,
                               vertices: segment.mesh.positions.count,
                               triangulos: segment.mesh.triangleCount,
                               archivo: file,
                               etiquetaIA: recognition?.label,
                               confianzaIA: recognition?.confidence)
    }
}
