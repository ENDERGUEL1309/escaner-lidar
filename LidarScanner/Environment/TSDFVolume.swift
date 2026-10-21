import Foundation
import simd

/// Fusión volumétrica TSDF (Truncated Signed Distance Function) con bloques dispersos.
///
/// En lugar de guardar puntos sueltos, el espacio se divide en vóxeles que almacenan la distancia
/// con signo a la superficie más cercana (+ delante, − detrás), promediada sobre todas las mediciones
/// LiDAR que los ven. Así:
///  • el ruido del sensor se cancela al promediar decenas de lecturas por vóxel,
///  • los objetos que se movieron se "borran" (las lecturas de espacio libre los sobrescriben),
///  • la superficie final es continua y cerrada, sin capas duplicadas.
/// Solo se reservan bloques de 8×8×8 vóxeles cerca de superficies observadas (hash espacial).
/// No es thread-safe: llámalo siempre desde la misma cola serie.
final class TSDFVolume {
    struct Voxel {
        var sdf: Float16 = 1
        var weight: Float16 = 0
        var r: UInt8 = 0
        var g: UInt8 = 0
        var b: UInt8 = 0
        var colorSamples: UInt8 = 0
    }

    final class Block {
        let origin: SIMD3<Int32>          // coordenadas del bloque (en unidades de bloque)
        var voxels = [Voxel](repeating: Voxel(), count: 512)
        var vertexIndex: [Int32] = []     // solo durante la extracción de la malla
        init(origin: SIMD3<Int32>) { self.origin = origin }
    }

    let voxelSize: Float
    let truncation: Float
    let maxBlocks: Int
    private let maxWeight: Float = 64
    private var blocks: [Int64: Block] = [:]

    init(voxelSize: Float, maxBlocks: Int) {
        self.voxelSize = voxelSize
        self.truncation = max(voxelSize * 4, 0.04)
        self.maxBlocks = maxBlocks
    }

    var blockCount: Int { blocks.count }
    var isFull: Bool { blocks.count >= maxBlocks }

    // MARK: - Integración de un fotograma

    func integrate(depth: [Float], confidence: [UInt8], width dw: Int, height dh: Int,
                   camera: CameraModel, rgb: RGBImage, maxDepth: Float, minConfidence: UInt8) {
        // 1. Reservar los bloques que atraviesa la banda de truncamiento alrededor de cada medición.
        let cameraPosition = camera.position
        let step = voxelSize * 4
        var touched: [Int64: Block] = [:]
        for y in stride(from: 2, to: dh, by: 4) {
            for x in stride(from: 2, to: dw, by: 4) {
                let i = y * dw + x
                let d = depth[i]
                guard d > 0.15, d <= maxDepth, confidence[i] >= minConfidence else { continue }
                let p = camera.unproject(u: Float(x) + 0.5, v: Float(y) + 0.5, depth: d)
                let ray = p - cameraPosition
                let length = simd_length(ray)
                guard length > 0 else { continue }
                let dir = ray / length
                var t = -truncation
                while t <= truncation + step * 0.5 {
                    let bc = blockCoord(of: p + dir * t)
                    let k = Self.key(bc)
                    if touched[k] == nil {
                        if let existing = blocks[k] {
                            touched[k] = existing
                        } else if blocks.count < maxBlocks {
                            let block = Block(origin: bc)
                            blocks[k] = block
                            touched[k] = block
                        }
                    }
                    t += step
                }
            }
        }

        // 2. Actualizar cada vóxel de esos bloques proyectándolo en el mapa de profundidad (en paralelo).
        let list = Array(touched.values)
        guard !list.isEmpty else { return }
        let vs = voxelSize, trunc = truncation, maxW = maxWeight
        let toRGBx = Float(rgb.width) / Float(dw)
        let toRGBy = Float(rgb.height) / Float(dh)

        depth.withUnsafeBufferPointer { dep in
            confidence.withUnsafeBufferPointer { conf in
                DispatchQueue.concurrentPerform(iterations: list.count) { n in
                    let block = list[n]
                    let bx = Int(block.origin.x) * 8, by = Int(block.origin.y) * 8, bz = Int(block.origin.z) * 8
                    block.voxels.withUnsafeMutableBufferPointer { vox in
                        for idx in 0..<512 {
                            let lx = idx & 7, ly = (idx >> 3) & 7, lz = idx >> 6
                            let center = SIMD3<Float>(Float(bx + lx) + 0.5, Float(by + ly) + 0.5, Float(bz + lz) + 0.5) * vs
                            guard let pr = camera.project(center), pr.u >= 0, pr.v >= 0 else { continue }
                            let ui = Int(pr.u), vi = Int(pr.v)
                            guard ui < dw, vi < dh else { continue }
                            let di = vi * dw + ui
                            let d = dep[di]
                            let c = conf[di]
                            guard d > 0.15, d <= maxDepth, c >= minConfidence else { continue }
                            let sdf = d - pr.z
                            guard sdf > -trunc else { continue }   // muy detrás de la superficie: no observado

                            // Más peso a lecturas de alta confianza y cercanas: el error del LiDAR crece con la distancia².
                            let w: Float = (c >= 2 ? 1 : 0.4) * max(0.15, min(1, 1 / (d * d)))
                            var v = vox[idx]
                            let oldW = Float(v.weight)
                            let total = oldW + w
                            v.sdf = Float16((Float(v.sdf) * oldW + min(1, sdf / trunc) * w) / total)
                            v.weight = Float16(min(total, maxW))

                            if abs(sdf) < trunc * 0.5,
                               let color = rgb.color(atX: pr.u * toRGBx, y: pr.v * toRGBy) {
                                let a = 1 / (Float(v.colorSamples) + 1)
                                let old = SIMD3<Float>(Float(v.r), Float(v.g), Float(v.b))
                                let mixed = rgb8(old + (color - old) * a)
                                v.r = mixed.x; v.g = mixed.y; v.b = mixed.z
                                if v.colorSamples < 32 { v.colorSamples += 1 }
                            }
                            vox[idx] = v
                        }
                    }
                }
            }
        }
    }

    // MARK: - Muestreo (para alinear fotogramas con el modelo)

    /// Distancia con signo (metros) y su gradiente en un punto, por interpolación trilineal de los 8 vóxeles
    /// vecinos. `nil` si alguno no está observado con al menos `minWeight`.
    func sample(_ p: SIMD3<Float>, minWeight: Float = 2) -> (distance: Float, gradient: SIMD3<Float>)? {
        let g = p / voxelSize - 0.5               // centros de vóxel en coordenadas enteras
        let base = g.rounded(.down)
        let f = g - base
        let i0 = SIMD3<Int32>(Int32(base.x), Int32(base.y), Int32(base.z))

        var s = [Float](repeating: 0, count: 8)
        for c in 0..<8 {
            let idx = i0 &+ SIMD3<Int32>(Int32(c & 1), Int32((c >> 1) & 1), Int32((c >> 2) & 1))
            guard let block = blocks[Self.key(SIMD3(idx.x >> 3, idx.y >> 3, idx.z >> 3))] else { return nil }
            let v = block.voxels[Int(idx.x & 7) + Int(idx.y & 7) * 8 + Int(idx.z & 7) * 64]
            guard Float(v.weight) >= minWeight else { return nil }
            s[c] = Float(v.sdf) * truncation
        }
        let (fx, fy, fz) = (f.x, f.y, f.z)
        // Interpolación trilineal: primero en x, luego en y, luego en z.
        let x00 = s[0] + (s[1] - s[0]) * fx, x10 = s[2] + (s[3] - s[2]) * fx
        let x01 = s[4] + (s[5] - s[4]) * fx, x11 = s[6] + (s[7] - s[6]) * fx
        let y0 = x00 + (x10 - x00) * fy, y1 = x01 + (x11 - x01) * fy
        let value = y0 + (y1 - y0) * fz
        // Derivadas analíticas de la misma interpolación (por metro).
        let dx = ((s[1] - s[0]) * (1 - fy) * (1 - fz) + (s[3] - s[2]) * fy * (1 - fz)
                  + (s[5] - s[4]) * (1 - fy) * fz + (s[7] - s[6]) * fy * fz)
        let dy = ((x10 - x00) * (1 - fz) + (x11 - x01) * fz)
        let dz = y1 - y0
        return (value, SIMD3(dx, dy, dz) / voxelSize)
    }

    // MARK: - Relleno de huecos

    /// Extiende la distancia con signo a los vóxeles no observados que están rodeados de vóxeles observados.
    /// Cada iteración cierra ~1 vóxel de borde, así que `iterations` = 4 cierra huecos de hasta ~8 vóxeles.
    /// Solo prolonga superficies existentes: no inventa geometría donde no se vio nada.
    func fillHoles(iterations: Int, progress: (Double) -> Void) {
        for iteration in 0..<iterations {
            let list = Array(blocks.values)
            let hoods = list.map { neighborhood(of: $0) }
            var pending = [[(Int, Voxel)]](repeating: [], count: list.count)

            pending.withUnsafeMutableBufferPointer { out in
                DispatchQueue.concurrentPerform(iterations: list.count) { n in
                    let block = list[n]
                    let hood = hoods[n]
                    var updates: [(Int, Voxel)] = []
                    for idx in 0..<512 where block.voxels[idx].weight == 0 {
                        let x = idx & 7, y = (idx >> 3) & 7, z = idx >> 6
                        var sdfSum: Float = 0, count: Float = 0
                        var colorSum = SIMD3<Float>(repeating: 0), colorCount: Float = 0
                        for (dx, dy, dz) in Self.faceOffsets {
                            guard let v = hood.voxel(x + dx, y + dy, z + dz) else { continue }
                            sdfSum += Float(v.sdf)
                            count += 1
                            if v.colorSamples > 0 {
                                colorSum += SIMD3<Float>(Float(v.r), Float(v.g), Float(v.b))
                                colorCount += 1
                            }
                        }
                        guard count >= 3 else { continue }
                        var filled = Voxel()
                        filled.sdf = Float16(sdfSum / count)
                        filled.weight = Self.filledWeight
                        if colorCount > 0 {
                            let c = rgb8(colorSum / colorCount)
                            filled.r = c.x; filled.g = c.y; filled.b = c.z
                            filled.colorSamples = 1
                        }
                        updates.append((idx, filled))
                    }
                    out[n] = updates
                }
            }

            for (n, updates) in pending.enumerated() {
                for (idx, voxel) in updates { list[n].voxels[idx] = voxel }
            }
            progress(Double(iteration + 1) / Double(iterations))
        }
    }

    // MARK: - Extracción de la superficie (Surface Nets)

    /// Genera una malla triangular con color y normales en la isosuperficie sdf = 0.
    /// Surface Nets coloca un vértice por celda que cruza la superficie y une celdas vecinas con quads:
    /// produce mallas suaves, sin agujeros entre bloques y sin las tablas de Marching Cubes.
    /// `minWeight`: peso acumulado mínimo para fiarse de un vóxel. Lo que solo se vio de pasada
    /// (reflejos, lecturas sueltas, bordes) no llega al mínimo y no genera superficie.
    func extractMesh(minWeight: Float = 1) -> ColoredMesh {
        var mesh = ColoredMesh()
        let list = Array(blocks.values)
        guard !list.isEmpty else { return mesh }
        let hoods = list.map { neighborhood(of: $0) }
        for block in list { block.vertexIndex = [Int32](repeating: -1, count: 512) }

        let cornerOffsets: [SIMD3<Float>] = (0..<8).map { SIMD3(Float($0 & 1), Float(($0 >> 1) & 1), Float(($0 >> 2) & 1)) }
        let edges: [(Int, Int)] = [(0, 1), (2, 3), (4, 5), (6, 7),
                                   (0, 2), (1, 3), (4, 6), (5, 7),
                                   (0, 4), (1, 5), (2, 6), (3, 7)]
        var corners = [Voxel](repeating: Voxel(), count: 8)

        // Paso 1: un vértice por celda con cambio de signo.
        for (n, block) in list.enumerated() {
            let hood = hoods[n]
            let base = SIMD3<Float>(Float(block.origin.x) * 8, Float(block.origin.y) * 8, Float(block.origin.z) * 8)
            for idx in 0..<512 {
                let x = idx & 7, y = (idx >> 3) & 7, z = idx >> 6
                var valid = true, anyInside = false, anyOutside = false
                for c in 0..<8 {
                    guard let v = hood.voxel(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1), minWeight: minWeight) else { valid = false; break }
                    corners[c] = v
                    if v.sdf < 0 { anyInside = true } else { anyOutside = true }
                }
                guard valid, anyInside, anyOutside else { continue }

                var sum = SIMD3<Float>(repeating: 0)
                var crossings: Float = 0
                var spurious = false
                for (a, b) in edges {
                    let sa = Float(corners[a].sdf), sb = Float(corners[b].sdf)
                    guard (sa < 0) != (sb < 0) else { continue }
                    // Salto entre espacio libre lejano y "detrás de la superficie": borde de silueta, no superficie real.
                    if abs(sa) > 0.95 && abs(sb) > 0.95 { spurious = true; break }
                    let t = sa / (sa - sb)
                    sum += cornerOffsets[a] + (cornerOffsets[b] - cornerOffsets[a]) * t
                    crossings += 1
                }
                guard !spurious, crossings > 0 else { continue }

                let s = (0..<8).map { Float(corners[$0].sdf) }
                var normal = SIMD3<Float>((s[1] + s[3] + s[5] + s[7]) - (s[0] + s[2] + s[4] + s[6]),
                                          (s[2] + s[3] + s[6] + s[7]) - (s[0] + s[1] + s[4] + s[5]),
                                          (s[4] + s[5] + s[6] + s[7]) - (s[0] + s[1] + s[2] + s[3]))
                let len = simd_length(normal)
                normal = len > 0 ? normal / len : SIMD3(0, 1, 0)

                var colorSum = SIMD3<Float>(repeating: 0), colorWeight: Float = 0
                for c in 0..<8 where corners[c].colorSamples > 0 {
                    let w = max(0.05, 1 - abs(s[c]))
                    colorSum += SIMD3<Float>(Float(corners[c].r), Float(corners[c].g), Float(corners[c].b)) * w
                    colorWeight += w
                }

                let local = SIMD3<Float>(Float(x), Float(y), Float(z)) + 0.5 + sum / crossings
                mesh.positions.append((base + local) * voxelSize)
                mesh.normals.append(normal)
                mesh.colors.append(colorWeight > 0 ? rgb8(colorSum / colorWeight) : SIMD3(160, 160, 160))
                block.vertexIndex[idx] = Int32(mesh.positions.count - 1)
            }
        }

        // Paso 2: por cada arista de vóxeles con cambio de signo, un quad entre las 4 celdas que la comparten.
        let units = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for (n, block) in list.enumerated() {
            let hood = hoods[n]
            for idx in 0..<512 {
                let x = idx & 7, y = (idx >> 3) & 7, z = idx >> 6
                guard let v = hood.voxel(x, y, z, minWeight: minWeight) else { continue }
                let inside = v.sdf < 0
                for axis in 0..<3 {
                    let ea = units[axis], eb = units[(axis + 1) % 3], ec = units[(axis + 2) % 3]
                    guard let neighbor = hood.voxel(x + ea.0, y + ea.1, z + ea.2, minWeight: minWeight),
                          (neighbor.sdf < 0) != inside else { continue }
                    let q00 = hood.vertex(x - eb.0 - ec.0, y - eb.1 - ec.1, z - eb.2 - ec.2)
                    let q10 = hood.vertex(x - ec.0, y - ec.1, z - ec.2)
                    let q11 = block.vertexIndex[idx]
                    let q01 = hood.vertex(x - eb.0, y - eb.1, z - eb.2)
                    guard q00 >= 0, q10 >= 0, q11 >= 0, q01 >= 0 else { continue }
                    let quad = inside ? [q00, q10, q11, q00, q11, q01] : [q00, q11, q10, q00, q01, q11]
                    mesh.indices.append(contentsOf: quad.map { UInt32($0) })
                }
            }
        }

        for block in list { block.vertexIndex = [] }
        return mesh
    }

    // MARK: - Utilidades

    /// Peso con el que se marcan los vóxeles rellenados (no medidos). Las lecturas reales nunca suman
    /// exactamente esto (la mínima es 0,4 × 0,15 = 0,06).
    static let filledWeight: Float16 = 0.05

    private static let faceOffsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    @inline(__always)
    private static func key(_ b: SIMD3<Int32>) -> Int64 {
        let mask: Int64 = 0x1FFFFF
        return (Int64(b.x) & mask) | ((Int64(b.y) & mask) << 21) | ((Int64(b.z) & mask) << 42)
    }

    @inline(__always)
    private func blockCoord(of p: SIMD3<Float>) -> SIMD3<Int32> {
        let q = (p / voxelSize).rounded(.down)
        return SIMD3<Int32>(Int32(q.x) >> 3, Int32(q.y) >> 3, Int32(q.z) >> 3)
    }

    /// Acceso rápido a los vóxeles de un bloque y sus 26 vecinos con coordenadas locales de −1 a 8.
    private struct Neighborhood {
        let blocks: [Block?]

        @inline(__always)
        func locate(_ x: Int, _ y: Int, _ z: Int) -> (Block?, Int) {
            let bx = x < 0 ? 0 : (x > 7 ? 2 : 1)
            let by = y < 0 ? 0 : (y > 7 ? 2 : 1)
            let bz = z < 0 ? 0 : (z > 7 ? 2 : 1)
            return (blocks[bx + by * 3 + bz * 9], (x & 7) + (y & 7) * 8 + (z & 7) * 64)
        }

        /// Vóxel observado con al menos `minWeight` de peso (o rellenado como hueco), o `nil`.
        @inline(__always)
        func voxel(_ x: Int, _ y: Int, _ z: Int, minWeight: Float = 0) -> Voxel? {
            let (block, i) = locate(x, y, z)
            guard let block else { return nil }
            let v = block.voxels[i]
            guard v.weight > 0 else { return nil }
            return (Float(v.weight) >= minWeight || v.weight == TSDFVolume.filledWeight) ? v : nil
        }

        @inline(__always)
        func vertex(_ x: Int, _ y: Int, _ z: Int) -> Int32 {
            let (block, i) = locate(x, y, z)
            guard let block, !block.vertexIndex.isEmpty else { return -1 }
            return block.vertexIndex[i]
        }
    }

    private func neighborhood(of block: Block) -> Neighborhood {
        var list = [Block?](repeating: nil, count: 27)
        for dz in -1...1 {
            for dy in -1...1 {
                for dx in -1...1 {
                    let o = SIMD3<Int32>(block.origin.x + Int32(dx), block.origin.y + Int32(dy), block.origin.z + Int32(dz))
                    list[(dx + 1) + (dy + 1) * 3 + (dz + 1) * 9] = blocks[Self.key(o)]
                }
            }
        }
        return Neighborhood(blocks: list)
    }
}
