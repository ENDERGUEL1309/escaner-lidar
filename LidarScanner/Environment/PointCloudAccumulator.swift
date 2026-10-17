import simd

struct ColoredPoint {
    var position: SIMD3<Float>
    var color: SIMD3<UInt8>
}

/// Nube de puntos a color fusionada en una rejilla de vóxeles.
/// Cada vóxel promedia posición y color de todas las mediciones que caen en él, lo que
/// reduce el ruido del LiDAR y evita duplicados al pasar varias veces por la misma zona.
/// No es thread-safe: úsalo siempre desde la misma cola serie.
final class PointCloudAccumulator {
    private struct Voxel {
        var position: SIMD3<Float>
        var color: SIMD3<Float>
        var count: Float
    }

    private(set) var voxelSize: Float
    private var voxels: [Int64: Voxel] = [:]
    private let maxVoxels: Int
    /// A partir de aquí el vóxel deja de promediar: ya es estable.
    private let maxSamplesPerVoxel: Float = 32

    init(voxelSize: Float, maxVoxels: Int = 4_000_000) {
        self.voxelSize = voxelSize
        self.maxVoxels = maxVoxels
    }

    var count: Int { voxels.count }

    func reset(voxelSize: Float) {
        self.voxelSize = voxelSize
        voxels.removeAll(keepingCapacity: false)
    }

    func add(_ p: SIMD3<Float>, color: SIMD3<Float>) {
        let k = key(for: p)
        if var v = voxels[k] {
            guard v.count < maxSamplesPerVoxel else { return }
            v.count += 1
            let t = 1 / v.count
            v.position += (p - v.position) * t
            v.color += (color - v.color) * t
            voxels[k] = v
        } else if voxels.count < maxVoxels {
            voxels[k] = Voxel(position: p, color: color, count: 1)
        }
    }

    func points() -> [ColoredPoint] {
        voxels.values.map { ColoredPoint(position: $0.position, color: rgb8($0.color)) }
    }

    /// Empaqueta las coordenadas enteras del vóxel (21 bits con signo por eje) en un Int64.
    @inline(__always)
    private func key(for p: SIMD3<Float>) -> Int64 {
        let q = (p / voxelSize).rounded(.down)
        let mask: Int64 = 0x1FFFFF
        let x = Int64(q.x) & mask
        let y = Int64(q.y) & mask
        let z = Int64(q.z) & mask
        return x | (y << 21) | (z << 42)
    }
}
