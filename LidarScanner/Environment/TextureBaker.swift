import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import simd

/// Texturizado fotográfico de la malla (como en fotogrametría):
///  1. Elige para cada triángulo la mejor foto de alta resolución: de frente, cerca y sin nada delante.
///  2. Suaviza esa elección para que zonas vecinas usen la misma foto (menos costuras).
///  3. Agrupa triángulos contiguos de la misma foto en "parches".
///  4. Empaqueta los parches en una sola imagen (atlas) y copia los píxeles de las fotos originales.
///  5. Duplica los vértices de los bordes entre parches y les asigna coordenadas de textura.
/// Los triángulos que ninguna foto vio bien se pintan con su color por vértice (paleta en el atlas).
enum TextureBaker {
    private static let padding = 2

    static func bake(_ mesh: ColoredMesh, keyframes: [Keyframe], atlasSize S: Int, outputURL: URL,
                     progress: (Double) -> Void) -> ColoredMesh? {
        let views = keyframes.filter { $0.fullImageURL != nil }
        let n = mesh.positions.count
        let triCount = mesh.triangleCount
        guard n > 0, triCount > 0, !views.isEmpty else { return nil }
        let indices = mesh.indices
        let positions = mesh.positions

        // Normal y centro de cada triángulo (orientada hacia fuera, como las normales de los vértices).
        var faceNormal = [SIMD3<Float>](repeating: .zero, count: triCount)
        var centroid = [SIMD3<Float>](repeating: .zero, count: triCount)
        for t in 0..<triCount {
            let a = Int(indices[t * 3]), b = Int(indices[t * 3 + 1]), c = Int(indices[t * 3 + 2])
            var normal = simd_cross(positions[b] - positions[a], positions[c] - positions[a])
            let len = simd_length(normal)
            normal = len > 0 ? normal / len : .zero
            if simd_dot(normal, mesh.normals[a] + mesh.normals[b] + mesh.normals[c]) < 0 { normal = -normal }
            faceNormal[t] = normal
            centroid[t] = (positions[a] + positions[b] + positions[c]) / 3
        }

        /// Calidad de la vista `k` para el triángulo `t` (0 = no sirve).
        func score(_ t: Int, _ k: Int) -> Float {
            let view = views[k]
            let camera = view.fullCamera
            let toCamera = camera.position - centroid[t]
            let distance = simd_length(toCamera)
            guard distance > 0.1 else { return 0 }
            let cosAngle = simd_dot(faceNormal[t], toCamera / distance)
            guard cosAngle > 0.3 else { return 0 }
            let margin: Float = 8
            let w = Float(view.fullWidth) - margin, h = Float(view.fullHeight) - margin
            for k in 0..<3 {
                guard let p = camera.project(positions[Int(indices[t * 3 + k])]),
                      p.u > margin, p.v > margin, p.u < w, p.v < h else { return 0 }
            }
            if let dp = view.depthCamera.project(centroid[t]), let sensed = view.depth(atU: dp.u, v: dp.v),
               dp.z > sensed + 0.04 + 0.02 * sensed { return 0 }   // tapado
            // Frontal y cercana = más píxeles de la foto por cm² de superficie.
            return cosAngle * cosAngle / (distance * distance)
        }

        // 1. Mejor foto por triángulo.
        var labels = [Int32](repeating: -1, count: triCount)
        let chunk = 4096
        let chunks = (triCount + chunk - 1) / chunk
        labels.withUnsafeMutableBufferPointer { out in
            DispatchQueue.concurrentPerform(iterations: chunks) { c in
                for t in (c * chunk)..<min(triCount, (c + 1) * chunk) {
                    var best: Float = 0
                    for k in views.indices {
                        let s = score(t, k)
                        if s > best { best = s; out[t] = Int32(k) }
                    }
                }
            }
        }
        progress(0.35)

        // Triángulos que tocan cada vértice (CSR).
        var offsets = [Int](repeating: 0, count: n + 1)
        for idx in indices { offsets[Int(idx) + 1] += 1 }
        for i in 0..<n { offsets[i + 1] += offsets[i] }
        var fill = offsets
        var vertexTriangles = [Int32](repeating: 0, count: indices.count)
        for t in 0..<triCount {
            for k in 0..<3 {
                let v = Int(indices[t * 3 + k])
                vertexTriangles[fill[v]] = Int32(t)
                fill[v] += 1
            }
        }

        // 2. Suavizado de la elección: cada triángulo adopta la foto mayoritaria de sus vecinos si también le sirve.
        for _ in 0..<2 {
            var updated = labels
            updated.withUnsafeMutableBufferPointer { out in
                DispatchQueue.concurrentPerform(iterations: chunks) { c in
                    var candidates = [Int32](repeating: -1, count: 32)
                    var votes = [Int32](repeating: 0, count: 32)
                    for t in (c * chunk)..<min(triCount, (c + 1) * chunk) {
                        var used = 0
                        for k in 0..<3 {
                            let v = Int(indices[t * 3 + k])
                            for j in offsets[v]..<offsets[v + 1] {
                                let label = labels[Int(vertexTriangles[j])]
                                guard label >= 0 else { continue }
                                if let slot = candidates[0..<used].firstIndex(of: label) {
                                    votes[slot] += 1
                                } else if used < 32 {
                                    candidates[used] = label
                                    votes[used] = 1
                                    used += 1
                                }
                            }
                        }
                        guard used > 0 else { continue }
                        var bestSlot = 0
                        for s in 1..<max(used, 1) where votes[s] > votes[bestSlot] { bestSlot = s }
                        let winner = candidates[bestSlot]
                        if winner != labels[t], score(t, Int(winner)) > 0 { out[t] = winner }
                    }
                }
            }
            labels = updated
        }
        progress(0.5)

        // 3. Parches: triángulos contiguos (comparten vértice) con la misma foto.
        var parent = (0..<triCount).map { Int32($0) }
        func find(_ x: Int32) -> Int32 {
            var x = x
            while parent[Int(x)] != x {
                parent[Int(x)] = parent[Int(parent[Int(x)])]
                x = parent[Int(x)]
            }
            return x
        }
        for v in 0..<n {
            let start = offsets[v], end = offsets[v + 1]
            guard end - start > 1 else { continue }
            for i in start..<end {
                let ti = vertexTriangles[i]
                guard labels[Int(ti)] >= 0 else { continue }
                for j in (i + 1)..<end {
                    let tj = vertexTriangles[j]
                    guard labels[Int(tj)] == labels[Int(ti)] else { continue }
                    let ri = find(ti), rj = find(tj)
                    if ri != rj { parent[Int(ri)] = rj }
                }
            }
        }

        // Coordenadas en la foto (px) de cada esquina de triángulo y rectángulo de cada parche.
        var cornerUV = [SIMD2<Float>](repeating: .zero, count: indices.count)
        var chartOf = [Int32](repeating: -1, count: triCount)
        var chartIndexByRoot: [Int32: Int32] = [:]
        var chartView: [Int] = []
        var chartMin: [SIMD2<Float>] = []
        var chartMax: [SIMD2<Float>] = []
        var trianglesByView = [[Int]](repeating: [], count: views.count)
        for t in 0..<triCount where labels[t] >= 0 {
            trianglesByView[Int(labels[t])].append(t)
            let root = find(Int32(t))
            let chart: Int32
            if let existing = chartIndexByRoot[root] {
                chart = existing
            } else {
                chart = Int32(chartView.count)
                chartIndexByRoot[root] = chart
                chartView.append(Int(labels[t]))
                chartMin.append(SIMD2(repeating: .greatestFiniteMagnitude))
                chartMax.append(SIMD2(repeating: -.greatestFiniteMagnitude))
            }
            chartOf[t] = chart
            let camera = views[Int(labels[t])].fullCamera
            for k in 0..<3 {
                let p = camera.project(positions[Int(indices[t * 3 + k])])
                let uv = SIMD2<Float>(p?.u ?? 0, p?.v ?? 0)
                cornerUV[t * 3 + k] = uv
                chartMin[Int(chart)] = simd_min(chartMin[Int(chart)], uv)
                chartMax[Int(chart)] = simd_max(chartMax[Int(chart)], uv)
            }
        }
        let chartCount = chartView.count

        // Paleta para los triángulos sin foto: su color medio, cuantizado a 4096 tonos (celdas de 4×4 px).
        var paletteCell: [Int: Int] = [:]
        var paletteColors: [SIMD3<UInt8>] = []
        var trianglePalette = [Int32](repeating: -1, count: triCount)
        for t in 0..<triCount where labels[t] < 0 {
            var sum = SIMD3<Int>(repeating: 0)
            for k in 0..<3 {
                let c = mesh.colors[Int(indices[t * 3 + k])]
                sum &+= SIMD3<Int>(Int(c.x), Int(c.y), Int(c.z))
            }
            let avg = sum / 3
            let key = (avg.x >> 4) << 8 | (avg.y >> 4) << 4 | (avg.z >> 4)
            if let cell = paletteCell[key] {
                trianglePalette[t] = Int32(cell)
            } else {
                paletteCell[key] = paletteColors.count
                trianglePalette[t] = Int32(paletteColors.count)
                paletteColors.append(SIMD3(UInt8(avg.x), UInt8(avg.y), UInt8(avg.z)))
            }
        }
        let paletteColumns = 64, cellSize = 4
        let paletteRows = (paletteColors.count + paletteColumns - 1) / paletteColumns

        // 4. Empaquetado en estanterías. Si no cabe, se reduce la escala de todos los parches.
        var sizes = (0..<chartCount).map { chartMax[$0] - chartMin[$0] + Float(padding * 2) }
        if paletteRows > 0 { sizes.append(SIMD2(Float(paletteColumns * cellSize), Float(paletteRows * cellSize))) }
        let paletteItem = paletteRows > 0 ? sizes.count - 1 : -1
        let order = sizes.indices.sorted { sizes[$0].y > sizes[$1].y }
        let totalArea = sizes.reduce(Float(0)) { $0 + $1.x * $1.y }

        func pack(scale: Float) -> [SIMD2<Int>]? {
            var placed = [SIMD2<Int>](repeating: .zero, count: sizes.count)
            var x = 0, y = 0, shelf = 0
            for i in order {
                let s = i == paletteItem ? 1 : scale
                let w = Int((sizes[i].x * s).rounded(.up)) + padding * 2
                let h = Int((sizes[i].y * s).rounded(.up)) + padding * 2
                guard w <= S, h <= S else { return nil }
                if x + w > S { x = 0; y += shelf; shelf = 0 }
                if y + h > S { return nil }
                placed[i] = SIMD2(x + padding, y + padding)
                x += w
                shelf = max(shelf, h)
            }
            return placed
        }
        var scale = min(1, sqrt(0.8 * Float(S * S) / max(totalArea, 1)))
        var placement = pack(scale: scale)
        while placement == nil, scale > 0.01 {
            scale *= 0.9
            placement = pack(scale: scale)
        }
        guard let placement else { return nil }
        progress(0.6)

        // Copia de píxeles: cada foto se decodifica una sola vez, para todos sus parches.
        var atlas = [UInt8](repeating: 0, count: S * S * 3)
        var chartsByView = [[Int]](repeating: [], count: views.count)
        for c in 0..<chartCount { chartsByView[chartView[c]].append(c) }
        let pad = padding
        for (k, charts) in chartsByView.enumerated() where !charts.isEmpty {
            guard let url = views[k].fullImageURL, let photo = decodeRGBA(url) else { continue }

            // Igualado de exposición: la cámara ajusta la exposición foto a foto y eso crea costuras de color.
            // Se compara la foto con el color de consenso de la malla (mezcla de todas las vistas) en sus
            // propios triángulos y se corrige con una ganancia por canal.
            var reference = SIMD3<Float>(repeating: 0), observed = SIMD3<Float>(repeating: 0)
            let own = trianglesByView[k]
            let sampleStep = max(1, own.count / 1500)
            for i in stride(from: 0, to: own.count, by: sampleStep) {
                let t = own[i]
                for c in 0..<3 {
                    let color = mesh.colors[Int(indices[t * 3 + c])]
                    reference += SIMD3<Float>(Float(color.x), Float(color.y), Float(color.z))
                    observed += photo.sample(cornerUV[t * 3 + c].x, cornerUV[t * 3 + c].y)
                }
            }
            let gain = simd_clamp(reference / simd_max(observed, SIMD3(repeating: 1)),
                                  SIMD3(repeating: 0.7), SIMD3(repeating: 1.4))

            atlas.withUnsafeMutableBufferPointer { out in
                DispatchQueue.concurrentPerform(iterations: charts.count) { i in
                    let c = charts[i]
                    let origin = placement[c]
                    let src = chartMin[c] - Float(pad)
                    let w = Int((sizes[c].x * scale).rounded(.up)), h = Int((sizes[c].y * scale).rounded(.up))
                    for y in -pad..<(h + pad) {
                        let ay = origin.y + y
                        guard ay >= 0, ay < S else { continue }
                        for x in -pad..<(w + pad) {
                            let ax = origin.x + x
                            guard ax >= 0, ax < S else { continue }
                            let su = src.x + (Float(x) + 0.5) / scale
                            let sv = src.y + (Float(y) + 0.5) / scale
                            let color = rgb8(photo.sample(su, sv) * gain)
                            let o = (ay * S + ax) * 3
                            out[o] = color.x; out[o + 1] = color.y; out[o + 2] = color.z
                        }
                    }
                }
            }
            progress(0.6 + 0.3 * Double(k + 1) / Double(views.count))
        }
        if paletteItem >= 0 {
            let origin = placement[paletteItem]
            for (i, color) in paletteColors.enumerated() {
                let cx = origin.x + (i % paletteColumns) * cellSize
                let cy = origin.y + (i / paletteColumns) * cellSize
                for y in 0..<cellSize {
                    for x in 0..<cellSize {
                        let o = ((cy + y) * S + cx + x) * 3
                        atlas[o] = color.x; atlas[o + 1] = color.y; atlas[o + 2] = color.z
                    }
                }
            }
        }
        guard writeJPEG(rgb: atlas, width: S, height: S, to: outputURL) else { return nil }
        atlas = []

        // 5. Malla final con coordenadas de textura (mismo orden de triángulos que la original).
        var out = ColoredMesh()
        var newIndex: [Int64: UInt32] = [:]
        newIndex.reserveCapacity(n * 2)
        let invS = 1 / Float(S)
        for t in 0..<triCount {
            for k in 0..<3 {
                let v = Int(indices[t * 3 + k])
                let group: Int64
                let uv: SIMD2<Float>
                if chartOf[t] >= 0 {
                    let c = Int(chartOf[t])
                    group = Int64(c)
                    let local = (cornerUV[t * 3 + k] - chartMin[c] + Float(padding)) * scale
                    uv = (SIMD2<Float>(Float(placement[c].x), Float(placement[c].y)) + local) * invS
                } else {
                    let cell = Int(trianglePalette[t])
                    group = Int64(chartCount + cell)
                    let origin = placement[paletteItem]
                    let px = Float(origin.x + (cell % paletteColumns) * cellSize) + Float(cellSize) / 2
                    let py = Float(origin.y + (cell / paletteColumns) * cellSize) + Float(cellSize) / 2
                    uv = SIMD2(px, py) * invS
                }
                let key = group << 32 | Int64(v)
                if let existing = newIndex[key] {
                    out.indices.append(existing)
                } else {
                    let index = UInt32(out.positions.count)
                    newIndex[key] = index
                    out.positions.append(positions[v])
                    out.normals.append(mesh.normals[v])
                    out.colors.append(mesh.colors[v])
                    out.uvs.append(uv)
                    out.indices.append(index)
                }
            }
        }
        progress(1)
        return out
    }

    // MARK: - Imágenes

    private struct Photo {
        let width: Int
        let height: Int
        let pixels: [UInt8]   // RGBA

        /// Muestreo bilineal (coordenadas en píxeles, se recortan al borde).
        @inline(__always)
        func sample(_ u: Float, _ v: Float) -> SIMD3<Float> {
            let x = max(0, min(Float(width) - 1.001, u - 0.5))
            let y = max(0, min(Float(height) - 1.001, v - 0.5))
            let x0 = Int(x), y0 = Int(y)
            let fx = x - Float(x0), fy = y - Float(y0)
            func px(_ xi: Int, _ yi: Int) -> SIMD3<Float> {
                let o = (yi * width + xi) * 4
                return SIMD3(Float(pixels[o]), Float(pixels[o + 1]), Float(pixels[o + 2]))
            }
            let top = px(x0, y0) * (1 - fx) + px(x0 + 1, y0) * fx
            let bottom = px(x0, y0 + 1) * (1 - fx) + px(x0 + 1, y0 + 1) * fx
            return top * (1 - fy) + bottom * fy
        }
    }

    private static func decodeRGBA(_ url: URL) -> Photo? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { return nil }
        let w = image.width, h = image.height
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: w, height: h,
                                          bitsPerComponent: 8, bytesPerRow: w * 4,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return drawn ? Photo(width: w, height: h, pixels: pixels) : nil
    }

    static func writeJPEG(rgb: [UInt8], width: Int, height: Int, to url: URL, quality: Double = 0.9) -> Bool {
        guard let provider = CGDataProvider(data: Data(rgb) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                                  bytesPerRow: width * 3, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
        else { return false }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        return CGImageDestinationFinalize(destination)
    }
}
