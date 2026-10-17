import ARKit
import Observation

/// Escaneo de entornos: combina
///  • la malla de reconstrucción de escena de ARKit (LiDAR),
///  • el mapa de profundidad por píxel (`sceneDepth`) para una nube de puntos densa,
///  • la cámara RGB para el color,
///  • el seguimiento visual-inercial de ARKit para situar todo en un mismo sistema de coordenadas.
@Observable
final class SceneScanner: NSObject, ARSessionDelegate {
    enum Phase: Equatable {
        case ready, scanning, paused, processing, finished
        case failed(String)
    }

    var phase: Phase = .ready
    var settings = ScanSettings()
    var processingStep = ""
    var progress: Double = 0
    var pointCount = 0
    var keyframeCount = 0
    var meshAnchorCount = 0
    var trackingHint: String?
    var exportedFiles: [URL] = []

    @ObservationIgnored private weak var session: ARSession?
    @ObservationIgnored private let queue = DispatchQueue(label: "lidar.scan.processing", qos: .userInitiated)
    @ObservationIgnored private let accumulator = PointCloudAccumulator(voxelSize: 0.01)
    @ObservationIgnored private let keyframes = KeyframeStore()
    @ObservationIgnored private var isIntegrating = false   // solo desde main
    @ObservationIgnored private var frameCounter = 0         // solo desde main

    static var isSupported: Bool {
        ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
            && ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    // MARK: - Control

    func attach(_ session: ARSession) {
        self.session = session
        session.delegate = self
        session.run(makeConfiguration())
    }

    func start() {
        guard let session else { return }
        let voxelSize = settings.voxelSize
        queue.async { [accumulator, keyframes] in
            accumulator.reset(voxelSize: voxelSize)
            keyframes.reset()
        }
        pointCount = 0
        keyframeCount = 0
        exportedFiles = []
        // Descarta la malla que ARKit haya construido antes de pulsar «Iniciar», sin perder el seguimiento.
        session.run(makeConfiguration(), options: [.resetSceneReconstruction])
        phase = .scanning
    }

    func pause() { if phase == .scanning { phase = .paused } }
    func resume() { if phase == .paused { phase = .scanning } }

    func stop() { session?.pause() }

    func prepareNewScan() {
        phase = .ready
        if let session { session.run(makeConfiguration(), options: [.resetSceneReconstruction]) }
    }

    func finish() {
        guard let session, phase == .scanning || phase == .paused else { return }
        let meshAnchors = session.currentFrame?.anchors.compactMap { $0 as? ARMeshAnchor } ?? []
        session.pause()
        phase = .processing
        progress = 0

        // La cola es serie: esto se ejecuta después del último fotograma en curso.
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let files = try self.exportScan(meshAnchors: meshAnchors)
                DispatchQueue.main.async {
                    self.exportedFiles = files
                    self.phase = .finished
                }
            } catch {
                DispatchQueue.main.async { self.phase = .failed(error.localizedDescription) }
            }
        }
    }

    private func makeConfiguration() -> ARWorldTrackingConfiguration {
        let config = ARWorldTrackingConfiguration()
        config.sceneReconstruction = ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification)
            ? .meshWithClassification : .mesh
        var semantics: ARConfiguration.FrameSemantics = [.sceneDepth]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) {
            semantics.insert(.smoothedSceneDepth)
        }
        config.frameSemantics = semantics
        config.environmentTexturing = .none
        config.planeDetection = []
        return config
    }

    // MARK: - ARSessionDelegate (hilo principal)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        updateTrackingHint(frame.camera.trackingState)

        frameCounter += 1
        if frameCounter % 30 == 0 {
            meshAnchorCount = frame.anchors.reduce(0) { $0 + ($1 is ARMeshAnchor ? 1 : 0) }
        }

        guard phase == .scanning, !isIntegrating,
              frameCounter % settings.frameInterval == 0 else { return }

        // Solo un fotograma en vuelo: si el procesado va lento se saltan fotogramas,
        // en lugar de retener ARFrames (lo que bloquearía la cámara).
        isIntegrating = true
        let settings = self.settings
        queue.async { [weak self] in
            guard let self else { return }
            self.integrate(frame, settings: settings)
            let points = self.accumulator.count
            let views = self.keyframes.count
            DispatchQueue.main.async {
                self.isIntegrating = false
                self.pointCount = points
                self.keyframeCount = views
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        phase = .failed(error.localizedDescription)
    }

    private func updateTrackingHint(_ state: ARCamera.TrackingState) {
        let hint: String?
        switch state {
        case .normal: hint = nil
        case .notAvailable: hint = "Seguimiento no disponible"
        case .limited(.initializing): hint = "Inicializando… mueve el iPhone despacio"
        case .limited(.excessiveMotion): hint = "Muévete más despacio"
        case .limited(.insufficientFeatures): hint = "Poca textura: apunta a zonas con más detalle"
        case .limited(.relocalizing): hint = "Relocalizando…"
        case .limited: hint = "Seguimiento limitado"
        }
        if hint != trackingHint { trackingHint = hint }
    }

    // MARK: - Integración (cola de procesado)

    private func integrate(_ frame: ARFrame, settings: ScanSettings) {
        guard case .normal = frame.camera.trackingState,
              let sceneDepth = frame.smoothedSceneDepth ?? frame.sceneDepth,
              let rgb = ImageSampler.downsampledRGB(from: frame.capturedImage, step: 3) else { return }

        let imageSize = frame.camera.imageResolution
        let baseCamera = CameraModel(intrinsics: frame.camera.intrinsics, cameraToWorld: frame.camera.transform)

        let depthMap = sceneDepth.depthMap
        let dw = CVPixelBufferGetWidth(depthMap)
        let dh = CVPixelBufferGetHeight(depthMap)
        let depthCamera = baseCamera.scaled(sx: Float(dw) / Float(imageSize.width),
                                            sy: Float(dh) / Float(imageSize.height))
        let rgbCamera = baseCamera.scaled(sx: Float(rgb.width) / Float(imageSize.width),
                                          sy: Float(rgb.height) / Float(imageSize.height))

        // Copia profundidad y confianza a memoria propia para no retener el ARFrame.
        var depth = [Float](repeating: 0, count: dw * dh)
        var confidence = [UInt8](repeating: UInt8(ARConfidenceLevel.high.rawValue), count: dw * dh)

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(depthMap) {
            let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
            for y in 0..<dh {
                let row = (base + y * rowBytes).assumingMemoryBound(to: Float32.self)
                for x in 0..<dw { depth[y * dw + x] = row[x] }
            }
        }
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)

        if let confMap = sceneDepth.confidenceMap {
            CVPixelBufferLockBaseAddress(confMap, .readOnly)
            if let base = CVPixelBufferGetBaseAddress(confMap) {
                let rowBytes = CVPixelBufferGetBytesPerRow(confMap)
                for y in 0..<dh {
                    let row = (base + y * rowBytes).assumingMemoryBound(to: UInt8.self)
                    for x in 0..<dw { confidence[y * dw + x] = row[x] }
                }
            }
            CVPixelBufferUnlockBaseAddress(confMap, .readOnly)
        }

        // Nube de puntos: cada píxel de profundidad fiable → punto 3D con el color de la cámara.
        let minConf = UInt8(settings.minConfidence.rawValue)
        let toRGBx = Float(rgb.width) / Float(dw)
        let toRGBy = Float(rgb.height) / Float(dh)
        for y in stride(from: 0, to: dh, by: settings.depthStride) {
            for x in stride(from: 0, to: dw, by: settings.depthStride) {
                let i = y * dw + x
                let d = depth[i]
                guard d > 0.15, d <= settings.maxDepth, confidence[i] >= minConf else { continue }
                let u = Float(x) + 0.5, v = Float(y) + 0.5
                guard let color = rgb.color(atX: u * toRGBx, y: v * toRGBy) else { continue }
                accumulator.add(depthCamera.unproject(u: u, v: v, depth: d), color: color)
            }
        }

        // Fotogramas clave para texturizar la malla al final.
        if keyframes.shouldAdd(camera: rgbCamera) {
            // Los píxeles poco fiables se marcan como 0 para no usarlos en la prueba de oclusión.
            for i in 0..<depth.count where confidence[i] < UInt8(ARConfidenceLevel.medium.rawValue) { depth[i] = 0 }
            keyframes.add(Keyframe(image: rgb, imageCamera: rgbCamera,
                                   depth: depth, depthWidth: dw, depthHeight: dh,
                                   depthCamera: depthCamera))
        }
    }

    // MARK: - Exportación (cola de procesado)

    private func exportScan(meshAnchors: [ARMeshAnchor]) throws -> [URL] {
        let folder = try ScanStorage.newScanFolder(prefix: "Entorno")
        var files: [URL] = []

        report("Extrayendo malla LiDAR", 0.02)
        var mesh = MeshBuilder.build(from: meshAnchors)

        if !mesh.isEmpty {
            report("Coloreando malla con \(keyframes.count) vistas", 0.05)
            MeshColorizer.colorize(&mesh, keyframes: keyframes.frames) { [weak self] p in
                self?.report(nil, 0.05 + 0.65 * p)
            }

            report("Exportando PLY", 0.72)
            let ply = folder.appendingPathComponent("malla_color.ply")
            try ModelExporter.writePLY(mesh: mesh, to: ply)
            files.append(ply)

            report("Exportando OBJ", 0.78)
            let obj = folder.appendingPathComponent("malla_color.obj")
            try ModelExporter.writeOBJ(mesh: mesh, to: obj)
            files.append(obj)

            report("Exportando USDZ", 0.86)
            let usdz = folder.appendingPathComponent("malla_color.usdz")
            if (try? ModelExporter.writeUSDZ(mesh: mesh, to: usdz)) != nil { files.append(usdz) }
        }

        report("Exportando nube de puntos", 0.94)
        let points = accumulator.points()
        if !points.isEmpty {
            let cloud = folder.appendingPathComponent("nube_puntos.ply")
            try ModelExporter.writePLY(points: points, to: cloud)
            files.append(cloud)
        }

        guard !files.isEmpty else {
            try? FileManager.default.removeItem(at: folder)
            throw ScanError.nothingCaptured
        }
        report("Listo", 1)
        return files
    }

    private func report(_ step: String?, _ value: Double) {
        DispatchQueue.main.async {
            if let step { self.processingStep = step }
            self.progress = value
        }
    }
}

enum ScanError: LocalizedError {
    case nothingCaptured
    case exportFailed(String)

    var errorDescription: String? {
        switch self {
        case .nothingCaptured: "No se capturó geometría. Escanea más tiempo y con buena luz."
        case .exportFailed(let format): "No se pudo exportar \(format)."
        }
    }
}
