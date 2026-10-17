import CoreVideo
import simd

/// Imagen RGB compacta (3 bytes por píxel) en memoria de CPU.
struct RGBImage {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    /// Color (0–255) del píxel más cercano, o `nil` si está fuera de la imagen.
    @inline(__always)
    func color(atX x: Float, y: Float) -> SIMD3<Float>? {
        let xi = Int(x), yi = Int(y)
        guard x >= 0, y >= 0, xi < width, yi < height else { return nil }
        let i = (yi * width + xi) * 3
        return SIMD3(Float(pixels[i]), Float(pixels[i + 1]), Float(pixels[i + 2]))
    }
}

enum ImageSampler {
    /// Convierte `ARFrame.capturedImage` (YCbCr 4:2:0 bi-planar) a RGB, reduciendo la resolución `step` veces.
    static func downsampledRGB(from buffer: CVPixelBuffer, step: Int) -> RGBImage? {
        guard CVPixelBufferGetPlaneCount(buffer) >= 2 else { return nil }
        let fullRange = CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let yRaw = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
              let cRaw = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else { return nil }
        let yBase = yRaw.assumingMemoryBound(to: UInt8.self)
        let cBase = cRaw.assumingMemoryBound(to: UInt8.self)
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let cStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let width = CVPixelBufferGetWidthOfPlane(buffer, 0) / step
        let height = CVPixelBufferGetHeightOfPlane(buffer, 0) / step

        var out = [UInt8](repeating: 0, count: width * height * 3)
        out.withUnsafeMutableBufferPointer { dst in
            for j in 0..<height {
                let sy = j * step
                for i in 0..<width {
                    let sx = i * step
                    var luma = Float(yBase[sy * yStride + sx])
                    let ci = (sy / 2) * cStride + (sx / 2) * 2
                    var cb = Float(cBase[ci]) - 128
                    var cr = Float(cBase[ci + 1]) - 128
                    if !fullRange {
                        luma = (luma - 16) * 1.164
                        cb *= 1.138
                        cr *= 1.138
                    }
                    let r = luma + 1.402 * cr
                    let g = luma - 0.344136 * cb - 0.714136 * cr
                    let b = luma + 1.772 * cb
                    let o = (j * width + i) * 3
                    dst[o] = UInt8(max(0, min(255, r)))
                    dst[o + 1] = UInt8(max(0, min(255, g)))
                    dst[o + 2] = UInt8(max(0, min(255, b)))
                }
            }
        }
        return RGBImage(width: width, height: height, pixels: out)
    }
}

@inline(__always)
func rgb8(_ c: SIMD3<Float>) -> SIMD3<UInt8> {
    SIMD3(UInt8(max(0, min(255, c.x.rounded()))),
          UInt8(max(0, min(255, c.y.rounded()))),
          UInt8(max(0, min(255, c.z.rounded()))))
}
