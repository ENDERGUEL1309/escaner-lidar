import CoreVideo
import Foundation
import Vision
import simd

/// Separación de objetos con IA: en cada foto clave, Vision (`VNGenerateForegroundInstanceMaskRequest`)
/// recorta los "sujetos" (cada objeto en primer plano recibe un número de instancia; 0 es el fondo).
/// Proyectando la malla en esas fotos, cada vértice acumula una "firma": qué instancia es en cada vista.
/// Dos vértices vecinos que la IA ve de forma consistente como objetos distintos (p. ej. una taza y la
/// mesa sobre la que está) no se unen en el mismo objeto aunque se toquen.
struct InstanceSignatures {
    private let viewCount: Int
    private let rowOf: [Int32]     // vértice → fila de la tabla (-1 si no se analiza)
    private let labels: [UInt8]    // fila × vista → instancia (0 = fondo, 255 = no visible en esa vista)

    private static let unseen: UInt8 = 255

    /// `include`: vértices a analizar (los que no son estructura).
    static func compute(mesh: ColoredMesh, include: [Bool], keyframes: [Keyframe]) -> InstanceSignatures? {
        let masks = keyframes.map { instanceMask(for: $0.image) }
        guard masks.contains(where: { $0 != nil }) else { return nil }

        let n = mesh.positions.count
        var rowOf = [Int32](repeating: -1, count: n)
        var rows = 0
        for i in 0..<n where include[i] {
            rowOf[i] = Int32(rows)
            rows += 1
        }
        guard rows > 0 else { return nil }

        let views = keyframes.count
        var labels = [UInt8](repeating: unseen, count: rows * views)
        let positions = mesh.positions
        let chunk = 4096
        labels.withUnsafeMutableBufferPointer { out in
            DispatchQueue.concurrentPerform(iterations: (n + chunk - 1) / chunk) { c in
                for v in (c * chunk)..<min(n, (c + 1) * chunk) where rowOf[v] >= 0 {
                    let row = Int(rowOf[v]) * views
                    let p = positions[v]
                    for (k, kf) in keyframes.enumerated() {
                        guard let mask = masks[k],
                              let pr = kf.imageCamera.project(p),
                              pr.u >= 0, pr.v >= 0,
                              pr.u < Float(kf.image.width), pr.v < Float(kf.image.height) else { continue }
                        if let dp = kf.depthCamera.project(p), let sensed = kf.depth(atU: dp.u, v: dp.v),
                           dp.z > sensed + 0.04 + 0.02 * sensed { continue }   // tapado en esta vista
                        let mx = min(mask.width - 1, Int(pr.u * Float(mask.width) / Float(kf.image.width)))
                        let my = min(mask.height - 1, Int(pr.v * Float(mask.height) / Float(kf.image.height)))
                        out[row + k] = mask.data[my * mask.width + mx]
                    }
                }
            }
        }
        return InstanceSignatures(viewCount: views, rowOf: rowOf, labels: labels)
    }

    /// ¿La IA ve a los vértices `a` y `b` como objetos distintos en la mayoría de las vistas comunes?
    func separates(_ a: Int, _ b: Int) -> Bool {
        let ra = Int(rowOf[a]), rb = Int(rowOf[b])
        guard ra >= 0, rb >= 0 else { return false }
        var both = 0, differ = 0
        for k in 0..<viewCount {
            let la = labels[ra * viewCount + k], lb = labels[rb * viewCount + k]
            guard la != Self.unseen, lb != Self.unseen else { continue }
            both += 1
            if la != lb { differ += 1 }
        }
        return both >= 2 && differ * 2 > both
    }

    private struct Mask {
        let width: Int
        let height: Int
        let data: [UInt8]
    }

    private static func instanceMask(for image: RGBImage) -> Mask? {
        guard let cgImage = ObjectRecognizer.cgImage(from: image,
                                                     rect: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        else { return nil }
        // Orientación del sensor tal cual: así la máscara coincide píxel a píxel con la foto.
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up)
        guard (try? handler.perform([request])) != nil,
              let observation = request.results?.first else { return nil }

        let buffer = observation.instanceMask
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        var data = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            let row = (base + y * rowBytes).assumingMemoryBound(to: UInt8.self)
            for x in 0..<w { data[y * w + x] = row[x] }
        }
        return Mask(width: w, height: h, data: data)
    }
}
