import CoreML
import Foundation
import Vision

/// Profundidad deducida solo del color de la foto (Depth Anything V2 Small de Apple, en el Neural Engine).
///
/// Es una fuente independiente del LiDAR: la red mira sombras, perspectiva y texturas. Devuelve profundidad
/// relativa (inversa y sin escala), así que se ajusta al LiDAR en cada foto (1/z ≈ a·d + b) y lo que queda es
/// el desacuerdo: donde color y LiDAR discrepan hay cristal, espejo, superficie negra o brillante, o una pose mala.
///
/// El modelo (DepthAnythingV2SmallF16.mlmodelc) lo añade la compilación de GitHub Actions; sin él todo sigue
/// funcionando y esta validación se marca como no disponible.
final class ColorDepth {
    private let request: VNCoreMLRequest

    init?() {
        guard let url = Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlmodelc") else { return nil }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        guard let model = try? MLModel(contentsOf: url, configuration: config),
              let vision = try? VNCoreMLModel(for: model) else { return nil }
        request = VNCoreMLRequest(model: vision)
        request.imageCropAndScaleOption = .scaleFill     // la imagen entera, para que píxel a píxel case con el LiDAR
    }

    /// Mapa de la red reducido a `width`×`height` (mismo encuadre que la imagen de la cámara). Más alto = más cerca.
    func disparity(_ buffer: CVPixelBuffer, width: Int, height: Int) -> [Float]? {
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up)
        guard (try? handler.perform([request])) != nil else { return nil }
        if let pixels = request.results?.first as? VNPixelBufferObservation {
            return Self.sample(pixels.pixelBuffer, width: width, height: height)
        }
        if let feature = request.results?.first as? VNCoreMLFeatureValueObservation,
           let array = feature.featureValue.multiArrayValue {
            return Self.sample(array, width: width, height: height)
        }
        return nil
    }

    private static func sample(_ buffer: CVPixelBuffer, width: Int, height: Int) -> [Float]? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let row = CVPixelBufferGetBytesPerRow(buffer)
        let format = CVPixelBufferGetPixelFormatType(buffer)
        func value(_ x: Int, _ y: Int) -> Float {
            let p = base + y * row
            switch format {
            case kCVPixelFormatType_OneComponent16Half:
                return Float(p.assumingMemoryBound(to: Float16.self)[x])
            case kCVPixelFormatType_OneComponent32Float, kCVPixelFormatType_DepthFloat32, kCVPixelFormatType_DisparityFloat32:
                return p.assumingMemoryBound(to: Float.self)[x]
            case kCVPixelFormatType_OneComponent8:
                return Float(p.assumingMemoryBound(to: UInt8.self)[x])
            default:
                return Float(p.assumingMemoryBound(to: UInt8.self)[x * 4])
            }
        }
        var out = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                out[y * width + x] = value(min(w - 1, (x * w + w / 2) / width), min(h - 1, (y * h + h / 2) / height))
            }
        }
        return out
    }

    private static func sample(_ array: MLMultiArray, width: Int, height: Int) -> [Float]? {
        let shape = array.shape.map(\.intValue)
        guard shape.count >= 2 else { return nil }
        let h = shape[shape.count - 2], w = shape[shape.count - 1]
        let strides = array.strides.map(\.intValue)
        let sy = strides[strides.count - 2], sx = strides[strides.count - 1]
        var out = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let i = min(h - 1, (y * h + h / 2) / height) * sy + min(w - 1, (x * w + w / 2) / width) * sx
                out[y * width + x] = array[i].floatValue
            }
        }
        return out
    }

    /// Ajusta la red al LiDAR (1/z = a·d + b, mínimos cuadrados) y devuelve la profundidad del color en metros
    /// y la fracción de píxeles donde ambas fuentes coinciden a menos del 5 %. `nil` si hay muy pocos datos.
    static func align(disparity d: [Float], lidar z: [Float]) -> (depth: [Float], agreement: Float)? {
        var n: Double = 0, sx: Double = 0, sy: Double = 0, sxx: Double = 0, sxy: Double = 0
        for i in d.indices where z[i] > 0.1 {
            let x = Double(d[i]), y = 1 / Double(z[i])
            n += 1; sx += x; sy += y; sxx += x * x; sxy += x * y
        }
        guard n >= 50 else { return nil }
        let denominator = n * sxx - sx * sx
        guard abs(denominator) > 1e-12 else { return nil }
        let a = (n * sxy - sx * sy) / denominator, b = (sy - a * sx) / n
        guard a > 0 else { return nil }      // al revés: la red y el LiDAR no ven la misma escena
        var depth = [Float](repeating: 0, count: d.count)
        var agree = 0, compared = 0
        for i in d.indices {
            let inverse = a * Double(d[i]) + b
            guard inverse > 0.05 else { continue }
            depth[i] = Float(1 / inverse)
            if z[i] > 0.1 {
                compared += 1
                if abs(depth[i] - z[i]) / z[i] < 0.05 { agree += 1 }
            }
        }
        return (depth, compared > 0 ? Float(agree) / Float(compared) : 0)
    }
}
