import simd

/// Modelo de cámara pinhole para una resolución concreta (imagen RGB o mapa de profundidad).
///
/// Convención: los intrínsecos de ARKit usan coordenadas tipo OpenCV (x derecha, y abajo, z hacia delante),
/// mientras que `ARCamera.transform` usa x derecha, y arriba, z hacia atrás. Se convierte con un flip de Y y Z.
struct CameraModel {
    var fx: Float
    var fy: Float
    var cx: Float
    var cy: Float
    let cameraToWorld: simd_float4x4
    let worldToCamera: simd_float4x4

    init(intrinsics K: simd_float3x3, cameraToWorld: simd_float4x4) {
        fx = K[0][0]
        fy = K[1][1]
        cx = K[2][0]
        cy = K[2][1]
        self.cameraToWorld = cameraToWorld
        self.worldToCamera = cameraToWorld.inverse
    }

    /// Reescala los intrínsecos a otra resolución (p. ej. de 1920x1440 a 256x192).
    func scaled(sx: Float, sy: Float) -> CameraModel {
        var c = self
        c.fx *= sx; c.cx *= sx
        c.fy *= sy; c.cy *= sy
        return c
    }

    var position: SIMD3<Float> {
        SIMD3(cameraToWorld.columns.3.x, cameraToWorld.columns.3.y, cameraToWorld.columns.3.z)
    }

    var forward: SIMD3<Float> {
        -SIMD3(cameraToWorld.columns.2.x, cameraToWorld.columns.2.y, cameraToWorld.columns.2.z)
    }

    /// Píxel (u, v) + profundidad z (metros) → punto en el mundo.
    @inline(__always)
    func unproject(u: Float, v: Float, depth: Float) -> SIMD3<Float> {
        let x = (u - cx) / fx * depth
        let y = (v - cy) / fy * depth
        let p = cameraToWorld * SIMD4<Float>(x, -y, -depth, 1)
        return SIMD3(p.x, p.y, p.z)
    }

    /// Punto del mundo → píxel (u, v) y profundidad z. `nil` si queda detrás de la cámara.
    @inline(__always)
    func project(_ world: SIMD3<Float>) -> (u: Float, v: Float, z: Float)? {
        let p = worldToCamera * SIMD4<Float>(world, 1)
        let z = -p.z
        guard z > 0.05 else { return nil }
        return (fx * p.x / z + cx, fy * (-p.y) / z + cy, z)
    }
}
