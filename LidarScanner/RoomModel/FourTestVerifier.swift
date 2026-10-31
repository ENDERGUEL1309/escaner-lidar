import Foundation
import simd

/// Bucle de las 4 pruebas: proyecta el modelo en las cámaras reales y lo compara con lo que no sale del modelo
/// (máscaras de Vision, lecturas LiDAR, fotos). Lo que ninguna cámara midió devuelve `nil`: no cuenta a favor ni en contra.
enum FourTestVerifier {
    static let maxRMSE: Float = 0.010
    static let minViews = 3
    /// Píxeles que deben atravesar una pieza para contar la vista como violación: separa objetos reales de lecturas sueltas.
    static let minViolationPixels = 12
    /// Vecindad del mueble donde se toman puntos LiDAR.
    private static let margin: Float = 0.05

    static func verify(_ input: ParametricFurniture, keyframes: [Keyframe], masks: [InstanceSignatures.Mask?],
                       metal: MetalFusion?) -> ParametricFurniture {
        var f = input
        var aspect: Bool?
        let views = keyframes.indices.filter { sees(f, keyframes[$0]) }

        // El mueble se ajusta a las lecturas antes de juzgarlo: RoomPlan da una caja, no las medidas finas.
        for _ in 0..<2 { refineSize(&f, keyframes: views.map { keyframes[$0] }) }
        let appliesAspect = f.kind == .desk || f.kind == .storage
        if appliesAspect {
            let decision = resolveVariant(f, keyframes: views.map { keyframes[$0] })
            if let variant = decision.variant, variant != f.variant {
                f.variant = variant
                f.rebuildParts()
            }
            aspect = decision.consistent
        }

        var report = VerificationReport()
        report.views = views.count
        report.silhouetteIoU = silhouetteIoU(f, views: views, keyframes: keyframes, masks: masks)
        report.geometryRMSE = geometryRMSE(f, keyframes: views.map { keyframes[$0] })
        report.freeSpaceViolations = freeSpaceViolations(f, keyframes: views.map { keyframes[$0] }, metal: metal)
        report.aspectConsistent = aspect
        report.aspectApplies = appliesAspect
        f.report = report
        f.state = state(of: f.kind, report: report)
        return f
    }

    static func state(of kind: FurnitureKind, report: VerificationReport) -> VerificationState {
        guard report.views >= minViews else { return .red }
        guard let iou = report.silhouetteIoU, iou >= (kind.isComplex ? 0.75 : 0.90),
              let rmse = report.geometryRMSE, rmse <= maxRMSE,
              report.freeSpaceViolations == 0,
              !report.aspectApplies || report.aspectConsistent == true else { return .yellow }
        return .green
    }

    private static func sees(_ f: ParametricFurniture, _ kf: Keyframe) -> Bool {
        let center = f.position + SIMD3(0, f.size.y / 2, 0)
        guard let p = kf.imageCamera.project(center) else { return false }
        return p.u >= 0 && p.v >= 0 && p.u < Float(kf.image.width) && p.v < Float(kf.image.height) && p.z < 6
    }

    // MARK: - Puntos LiDAR del mueble

    /// Puntos medidos cerca del mueble, en sus coordenadas. Excluye suelo, y lo que hay encima (monitores, almohadas, tazas).
    private static func localPoints(_ f: ParametricFurniture, keyframes: [Keyframe], step: Int = 2) -> [SIMD3<Float>] {
        let toLocal = f.worldToLocal
        let half = SIMD3(f.size.x / 2 + margin, 0, f.size.z / 2 + margin)
        var points: [SIMD3<Float>] = []
        for kf in keyframes where kf.depthWidth > 0 {
            for y in stride(from: 0, to: kf.depthHeight, by: step) {
                for x in stride(from: 0, to: kf.depthWidth, by: step) {
                    let d = kf.depth[y * kf.depthWidth + x]
                    guard d > 0.15, d < 4.5 else { continue }
                    let world = kf.depthCamera.unproject(u: Float(x) + 0.5, v: Float(y) + 0.5, depth: d)
                    let p = toLocal * SIMD4<Float>(world, 1)
                    guard abs(p.x) < half.x, abs(p.z) < half.z, p.y > 0.03, p.y < f.size.y + 0.03 else { continue }
                    points.append(SIMD3(p.x, p.y, p.z))
                }
            }
        }
        return points
    }

    /// Acerca ancho, fondo y alto a las lecturas (como mucho 8 cm por pasada). Sin lecturas no se toca nada.
    private static func refineSize(_ f: inout ParametricFurniture, keyframes: [Keyframe]) {
        let points = localPoints(f, keyframes: keyframes, step: 3)
        guard points.count >= 200 else { return }
        func spread(_ values: [Float]) -> (low: Float, high: Float) {
            let sorted = values.sorted()
            return (sorted[sorted.count / 100], sorted[sorted.count - 1 - sorted.count / 100])
        }
        let x = spread(points.map(\.x)), z = spread(points.map(\.z))
        let top = spread(points.map(\.y)).high
        let limit: Float = 0.08
        func clampChange(_ new: Float, _ old: Float) -> Float { max(old - limit, min(old + limit, new)) }
        f.size = SIMD3(clampChange(x.high - x.low, f.size.x), clampChange(top, f.size.y), clampChange(z.high - z.low, f.size.z))
        let shift = SIMD4<Float>((x.low + x.high) / 2, 0, (z.low + z.high) / 2, 0)
        let world = f.localToWorld * SIMD4<Float>(max(-limit, min(limit, shift.x)), 0, max(-limit, min(limit, shift.z)), 0)
        f.position += SIMD3(world.x, 0, world.z)
        f.rebuildParts()
    }

    // MARK: - Prueba de geometría

    private static func surfaceDistance(_ p: SIMD3<Float>, part: FurniturePart) -> Float {
        let q = simd_abs(p - part.center) - part.size / 2
        let outside = simd_length(simd_max(q, SIMD3(repeating: 0)))
        let inside = min(max(q.x, max(q.y, q.z)), 0)
        return outside > 0 ? outside : -inside
    }

    private static func geometryRMSE(_ f: ParametricFurniture, keyframes: [Keyframe]) -> Float? {
        let points = localPoints(f, keyframes: keyframes)
        guard points.count >= 50 else { return nil }
        var sum: Float = 0
        for p in points {
            let d = f.parts.map { surfaceDistance(p, part: $0) }.min() ?? 0
            sum += d * d
        }
        return (sum / Float(points.count)).squareRoot()
    }

    // MARK: - Prueba de espacio libre

    private static func freeSpaceViolations(_ f: ParametricFurniture, keyframes: [Keyframe], metal: MetalFusion?) -> Int? {
        guard let metal else { return nil }
        var total = 0, measured = false
        for kf in keyframes where kf.depthWidth > 0 {
            guard let count = metal.freeSpaceViolations(depth: kf.depth, width: kf.depthWidth, height: kf.depthHeight,
                                                        camera: kf.depthCamera, worldToLocal: f.worldToLocal,
                                                        parts: f.parts) else { continue }
            measured = true
            if count >= minViolationPixels { total += count }
        }
        return measured ? total : nil
    }

    // MARK: - Prueba de silueta

    private static func corners(of part: FurniturePart) -> [SIMD3<Float>] {
        let h = part.size / 2
        return (0..<8).map { i in
            part.center + SIMD3(i & 1 == 0 ? -h.x : h.x, i & 2 == 0 ? -h.y : h.y, i & 4 == 0 ? -h.z : h.z)
        }
    }

    private static func cross(_ o: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }

    private static func convexHull(_ points: [SIMD2<Float>]) -> [SIMD2<Float>] {
        let sorted = points.sorted { $0.x != $1.x ? $0.x < $1.x : $0.y < $1.y }
        var lower: [SIMD2<Float>] = [], upper: [SIMD2<Float>] = []
        for p in sorted {
            while lower.count >= 2, cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 { lower.removeLast() }
            lower.append(p)
        }
        for p in sorted.reversed() {
            while upper.count >= 2, cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 { upper.removeLast() }
            upper.append(p)
        }
        return Array(lower.dropLast() + upper.dropLast())
    }

    private static func fill(_ hull: [SIMD2<Float>], into mask: inout [Bool], width: Int, height: Int) {
        guard hull.count >= 3 else { return }
        let lowY = max(0, Int(hull.map(\.y).min()!)), highY = min(height - 1, Int(hull.map(\.y).max()!))
        guard lowY <= highY else { return }
        for y in lowY...highY {
            let line = Float(y) + 0.5
            var left = Float.infinity, right = -Float.infinity
            for i in hull.indices {
                let a = hull[i], b = hull[(i + 1) % hull.count]
                guard (a.y <= line) != (b.y <= line) else { continue }
                let x = a.x + (line - a.y) / (b.y - a.y) * (b.x - a.x)
                left = min(left, x)
                right = max(right, x)
            }
            guard left <= right else { continue }
            let from = max(0, Int(left)), to = min(width - 1, Int(right))
            if from <= to { for x in from...to { mask[y * width + x] = true } }
        }
    }

    /// Silueta del modelo en la vista, a la resolución de `camera`. También devuelve la distancia a su punto más cercano.
    private static func rasterize(_ parts: [FurniturePart], toWorld: simd_float4x4, camera: CameraModel,
                                  width: Int, height: Int) -> (mask: [Bool], nearest: Float)? {
        var mask = [Bool](repeating: false, count: width * height)
        var nearest = Float.infinity
        for part in parts {
            var projected: [SIMD2<Float>] = []
            for corner in corners(of: part) {
                let w = toWorld * SIMD4<Float>(corner, 1)
                guard let p = camera.project(SIMD3(w.x, w.y, w.z)) else { return nil }
                projected.append(SIMD2(p.u, p.v))
                nearest = min(nearest, p.z)
            }
            fill(convexHull(projected), into: &mask, width: width, height: height)
        }
        return (mask, nearest)
    }

    private static func silhouetteIoU(_ f: ParametricFurniture, views: [Int], keyframes: [Keyframe],
                                      masks: [InstanceSignatures.Mask?]) -> Float? {
        var sum: Float = 0, count = 0
        for k in views {
            guard k < masks.count, let segmentation = masks[k] else { continue }
            let kf = keyframes[k]
            let w = segmentation.width, h = segmentation.height
            let camera = kf.imageCamera.scaled(sx: Float(w) / Float(kf.image.width), sy: Float(h) / Float(kf.image.height))
            guard let model = rasterize(f.parts, toWorld: f.localToWorld, camera: camera, width: w, height: h) else { continue }

            // Lo que tiene delante otro objeto no es comparable en esta vista.
            let sx = Float(kf.depthWidth) / Float(w), sy = Float(kf.depthHeight) / Float(h)
            func isOccluded(_ x: Int, _ y: Int) -> Bool {
                guard let sensed = kf.depth(atU: (Float(x) + 0.5) * sx, v: (Float(y) + 0.5) * sy) else { return false }
                return sensed < model.nearest - 0.08
            }

            // La máscara independiente es la instancia que más solapa con el modelo.
            var votes = [Int](repeating: 0, count: 256)
            for i in model.mask.indices where model.mask[i] { votes[Int(segmentation.data[i])] += 1 }
            votes[0] = 0
            guard let instance = votes.indices.max(by: { votes[$0] < votes[$1] }), votes[instance] > 0 else { continue }

            var both = 0, either = 0
            for y in 0..<h {
                for x in 0..<w {
                    let i = y * w + x
                    let inModel = model.mask[i], inMask = Int(segmentation.data[i]) == instance
                    guard inModel || inMask, !isOccluded(x, y) else { continue }
                    either += 1
                    if inModel && inMask { both += 1 }
                }
            }
            guard either > 100 else { continue }
            sum += Float(both) / Float(either)
            count += 1
        }
        return count > 0 ? sum / Float(count) : nil
    }

    // MARK: - Prueba de aspecto

    /// Brillo medio y energía de bordes (Sobel) de una imagen dentro de una máscara.
    private static func photometry(_ image: RGBImage, mask: [Bool]) -> (luma: Float, edges: Float)? {
        let w = image.width, h = image.height
        guard w > 2, h > 2 else { return nil }
        func luma(_ x: Int, _ y: Int) -> Float {
            let i = (y * w + x) * 3
            return 0.299 * Float(image.pixels[i]) + 0.587 * Float(image.pixels[i + 1]) + 0.114 * Float(image.pixels[i + 2])
        }
        var sum: Float = 0, edges: Float = 0, n: Float = 0
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) where mask[y * w + x] {
                sum += luma(x, y)
                let gx = luma(x + 1, y - 1) + 2 * luma(x + 1, y) + luma(x + 1, y + 1)
                    - luma(x - 1, y - 1) - 2 * luma(x - 1, y) - luma(x - 1, y + 1)
                let gy = luma(x - 1, y + 1) + 2 * luma(x, y + 1) + luma(x + 1, y + 1)
                    - luma(x - 1, y - 1) - 2 * luma(x, y - 1) - luma(x + 1, y - 1)
                edges += abs(gx) + abs(gy)
                n += 1
            }
        }
        return n > 50 ? (sum / n, edges / n) : nil
    }

    /// Resuelve lo que la silueta no distingue. Escritorio: un frente de cajones iluminado es más claro que el hueco de
    /// las piernas. Armario: unas baldas abiertas tienen más bordes que unas puertas lisas. `consistent` exige que al
    /// menos dos tercios de las vistas coincidan; si no hay vistas válidas, no hay decisión.
    private static func resolveVariant(_ f: ParametricFurniture, keyframes: [Keyframe]) -> (variant: FurnitureVariant?, consistent: Bool?) {
        var yes = 0, no = 0
        let w = f.size.x / 2, h = f.size.y, d = f.size.z / 2
        let block = min(0.45, f.size.x * 0.4)
        let drawers = FurniturePart(name: "", center: SIMD3(w - block / 2, h * 0.5, d), size: SIMD3(block, h * 0.8, 0.01))
        let hollow = FurniturePart(name: "", center: SIMD3(-block / 2, h * 0.45, d), size: SIMD3(f.size.x - block - 0.1, h * 0.7, 0.01))
        let front = FurniturePart(name: "", center: SIMD3(0, h * 0.5, d), size: SIMD3(f.size.x * 0.9, h * 0.9, 0.01))

        for kf in keyframes {
            let width = kf.image.width, height = kf.image.height
            func read(_ part: FurniturePart) -> (luma: Float, edges: Float)? {
                guard let model = rasterize([part], toWorld: f.localToWorld, camera: kf.imageCamera, width: width, height: height) else { return nil }
                return photometry(kf.image, mask: model.mask)
            }
            if f.kind == .desk {
                guard let a = read(drawers), let b = read(hollow) else { continue }
                if a.luma - b.luma > 12 { yes += 1 } else { no += 1 }
            } else {
                guard let a = read(front) else { continue }
                if a.edges > 60 { yes += 1 } else { no += 1 }
            }
        }
        let total = yes + no
        guard total >= 2 else { return (nil, nil) }
        let positive = yes >= no
        let variant: FurnitureVariant = f.kind == .desk
            ? (positive ? .deskWithDrawers : .deskOpen)
            : (positive ? .storageOpen : .storageDoors)
        return (variant, max(yes, no) * 3 >= total * 2)
    }
}
