import ARKit

struct ScanSettings {
    enum Quality: String, CaseIterable, Identifiable {
        case fast = "Rápido"
        case balanced = "Equilibrado"
        case precise = "Preciso"
        var id: String { rawValue }
    }

    var quality: Quality = .balanced
    /// El LiDAR del iPhone es fiable hasta ~5 m; más lejos el ruido crece mucho.
    var maxDepth: Float = 4.0
    var minConfidence: ARConfidenceLevel = .high

    /// Tamaño del vóxel de la nube de puntos (metros).
    var voxelSize: Float {
        switch quality {
        case .fast: 0.02
        case .balanced: 0.01
        case .precise: 0.005
        }
    }

    /// Cada cuántos píxeles del mapa de profundidad (256x192) se toma una muestra.
    var depthStride: Int {
        switch quality {
        case .fast: 4
        case .balanced: 2
        case .precise: 1
        }
    }

    /// Cada cuántos fotogramas de ARKit (60 fps) se integra uno.
    var frameInterval: Int {
        switch quality {
        case .fast: 10
        case .balanced: 6
        case .precise: 4
        }
    }
}
