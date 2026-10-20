import ARKit

struct ScanSettings {
    enum Quality: String, CaseIterable, Identifiable {
        case fast = "Rápido"
        case balanced = "Equilibrado"
        case precise = "Preciso"
        var id: String { rawValue }
    }

    var quality: Quality = .balanced
    /// Cierra huecos pequeños de la superficie prolongando la geometría vecina.
    var fillHoles = true
    /// Separa la escena en estructura (paredes, suelo…) y objetos editables.
    var detectObjects = true
    /// Escanear desde ubicaciones fijas: se registra quieto en cada sitio y se pausa al caminar entre ellos.
    var stationMode = true
    /// Guarda las fotos de 12 MP con sus cámaras en `fotos_clave/` (unos 3–5 MB por foto) para la IA en la nube.
    var saveCloudPhotos = true
    /// Guarda además ~2 fotogramas por segundo (1280 px, ~200 KB) en `fotogramas/` para la vista
    /// fotorrealista (Gaussian Splatting) en la nube, que necesita muchas más vistas que la detección.
    var captureSplatFrames = true
    static let maxSplatFrames = 400
    /// Las lecturas de confianza media se usan con menos peso; las bajas se descartan.
    var minConfidence: ARConfidenceLevel = .medium

    /// Tamaño del vóxel de la fusión TSDF (metros): la resolución de la superficie final.
    var voxelSize: Float {
        switch quality {
        case .fast: 0.02
        case .balanced: 0.015
        case .precise: 0.01
        }
    }

    /// Límite de bloques de 8×8×8 vóxeles (~4 KB cada uno) para no agotar la memoria.
    var maxBlocks: Int {
        switch quality {
        case .fast: 60_000
        case .balanced: 80_000
        case .precise: 100_000
        }
    }

    /// Lado (px) de la imagen de textura. 8192² son 64 Mpx: el máximo detalle sin agotar la memoria.
    var atlasSize: Int {
        quality == .precise ? 8192 : 4096
    }

    /// El LiDAR del iPhone es fiable hasta ~5 m; más lejos el ruido crece mucho.
    var maxDepth: Float {
        quality == .precise ? 3.0 : 4.0
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
