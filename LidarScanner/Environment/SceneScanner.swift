import ARKit
import Observation

/// Escaneo de entornos: combina
///  • el mapa de profundidad LiDAR por píxel (`sceneDepth`), fusionado en un volumen TSDF,
///  • la cámara RGB para el color (en el volumen y proyectando fotos clave sobre la malla),
///  • el seguimiento visual-inercial de ARKit para situar todo en un mismo sistema de coordenadas,
///  • la malla clasificada de ARKit (red neuronal en el dispositivo) para reconocer paredes, suelo y muebles.
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
    var voxelCount = 0
    var keyframeCount = 0
    var meshAnchorCount = 0
    var isVolumeFull = false
    var trackingHint: String?
    var exportedFiles: [URL] = []
    var detectedObjects: [SceneObjectInfo] = []

    @ObservationIgnored private weak var session: ARSession?
    @ObservationIgnored private let queue = DispatchQueue(label: "lidar.scan.processing", qos: .userInitiated)
    @ObservationIgnored private var volume = TSDFVolume(voxelSize: 0.015, maxBlocks: 80_000)   // solo desde `queue`
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
        let maxBlocks = settings.maxBlocks
        queue.async { [weak self, keyframes] in
            self?.volume = TSDFVolume(voxelSize: voxelSize, maxBlocks: maxBlocks)
            keyframes.reset()
        }
        voxelCount = 0
        keyframeCount = 0
        isVolumeFull = false
        exportedFiles = []
        detectedObjects = []
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
        let settings = self.settings

        // La cola es serie: esto se ejecuta después del último fotograma en curso.
        queue.async { [weak self] in
            guard let self else { return }
            do {
                let (files, objects) = try self.exportScan(meshAnchors: meshAnchors, settings: settings)
                DispatchQueue.main.async {
                    self.exportedFiles = files
                    self.detectedObjects = objects
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
            let voxels = self.volume.blockCount * 512
            let full = self.volume.isFull
            let views = self.keyframes.count
            DispatchQueue.main.async {
                self.isIntegrating = false
                self.voxelCount = voxels
                self.isVolumeFull = full
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

        // Fusión volumétrica: cada lectura LiDAR refina la distancia a la superficie y su color.
        volume.integrate(depth: depth, confidence: confidence, width: dw, height: dh,
                         camera: depthCamera, rgb: rgb,
                         maxDepth: settings.maxDepth,
                         minConfidence: UInt8(settings.minConfidence.rawValue))

        // Fotogramas clave para dar a la malla el color más nítido al final.
        if keyframes.shouldAdd(camera: rgbCamera) {
            // Los píxeles poco fiables se marcan como 0 para no usarlos en la prueba de oclusión.
            for i in 0..<depth.count where confidence[i] < UInt8(ARConfidenceLevel.medium.rawValue) { depth[i] = 0 }
            keyframes.add(Keyframe(image: rgb, imageCamera: rgbCamera,
                                   depth: depth, depthWidth: dw, depthHeight: dh,
                                   depthCamera: depthCamera))
        }
    }

    // MARK: - Post-proceso y exportación (cola de procesado)

    private func exportScan(meshAnchors: [ARMeshAnchor], settings: ScanSettings) throws -> ([URL], [SceneObjectInfo]) {
        let folder = try ScanStorage.newScanFolder(prefix: "Entorno")
        var files: [URL] = []
        var objects: [SceneObjectInfo] = []

        if settings.fillHoles {
            report("Rellenando huecos", 0.02)
            volume.fillHoles(iterations: 4) { [weak self] p in self?.report(nil, 0.02 + 0.10 * p) }
        }

        report("Generando superficie", 0.12)
        var mesh = volume.extractMesh()
        if mesh.isEmpty {
            // Sin datos de profundidad útiles: se recurre a la malla de ARKit.
            mesh = MeshBuilder.build(from: meshAnchors)
        }
        guard !mesh.isEmpty else {
            try? FileManager.default.removeItem(at: folder)
            throw ScanError.nothingCaptured
        }

        report("Coloreando con \(keyframes.count) vistas", 0.2)
        MeshColorizer.colorize(&mesh, keyframes: keyframes.frames) { [weak self] p in
            self?.report(nil, 0.2 + 0.4 * p)
        }

        report("Exportando malla", 0.62)
        let ply = folder.appendingPathComponent("malla_color.ply")
        try ModelExporter.writePLY(mesh: mesh, to: ply)
        files.append(ply)

        let obj = folder.appendingPathComponent("malla_color.obj")
        try ModelExporter.writeOBJ(mesh: mesh, to: obj)
        files.append(obj)

        report("Exportando USDZ", 0.68)
        let usdz = folder.appendingPathComponent("malla_color.usdz")
        if (try? ModelExporter.writeUSDZ(mesh: mesh, to: usdz)) != nil { files.append(usdz) }

        report("Exportando nube de puntos", 0.74)
        let cloud = folder.appendingPathComponent("nube_puntos.ply")
        try ModelExporter.writePointCloudPLY(mesh: mesh, to: cloud)
        files.append(cloud)

        if settings.detectObjects {
            report("Detectando objetos", 0.78)
            let classifier = SurfaceClassifier(anchors: meshAnchors)
            if !classifier.isEmpty {
                let segments = SceneSegmenter.segment(mesh, classifier: classifier)
                let parts = segments.map { ModelExporter.Part(name: $0.name, mesh: $0.mesh) }

                report("Exportando escena por objetos", 0.88)
                let segmentedOBJ = folder.appendingPathComponent("escena_objetos.obj")
                try ModelExporter.writeOBJ(parts: parts, to: segmentedOBJ)
                files.append(segmentedOBJ)

                let segmentedUSDZ = folder.appendingPathComponent("escena_objetos.usdz")
                if (try? ModelExporter.writeUSDZ(parts: parts, to: segmentedUSDZ)) != nil { files.append(segmentedUSDZ) }

                objects = segments.map { SceneSegmenter.info(for: $0, file: ModelExporter.objectName($0.name)) }
                let json = folder.appendingPathComponent("escena.json")
                try ModelExporter.writeJSON(SceneDescription(objetos: objects), to: json)
                files.append(json)
            }
        }

        report("Listo", 1)
        return (files, objects)
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
