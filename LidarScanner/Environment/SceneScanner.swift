import ARKit
import CoreImage
import ImageIO
import Observation
import UIKit

/// Escaneo de entornos: combina
///  • el mapa de profundidad LiDAR por píxel (`sceneDepth`), filtrado y fusionado en un volumen TSDF,
///  • fotos de alta resolución (hasta 12 MP) tomadas cuando la cámara está quieta, para la textura,
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
    @ObservationIgnored private let keyframes = KeyframeStore()                                // solo desde `queue`
    @ObservationIgnored private let ciContext = CIContext()
    @ObservationIgnored private let keyframeFolder = FileManager.default.temporaryDirectory
        .appendingPathComponent("FotosClave", isDirectory: true)

    // Solo desde el hilo principal.
    @ObservationIgnored private var isIntegrating = false
    @ObservationIgnored private var frameCounter = 0
    @ObservationIgnored private var isCapturingKeyframe = false
    @ObservationIgnored private var lastKeyframeTransform: simd_float4x4?
    @ObservationIgnored private var previousTransform: simd_float4x4?
    @ObservationIgnored private var previousTimestamp: TimeInterval = 0
    @ObservationIgnored private var keyframeSpacing: Float = 1

    static var isSupported: Bool {
        ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
            && ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    private static var supportsHighResolutionPhotos: Bool {
        ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing != nil
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
        let folder = keyframeFolder
        queue.async { [weak self, keyframes] in
            self?.volume = TSDFVolume(voxelSize: voxelSize, maxBlocks: maxBlocks)
            keyframes.reset()
            try? FileManager.default.removeItem(at: folder)
            try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        }
        voxelCount = 0
        keyframeCount = 0
        isVolumeFull = false
        exportedFiles = []
        detectedObjects = []
        lastKeyframeTransform = nil
        keyframeSpacing = 1
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

        // La cola es serie: esto se ejecuta después del último fotograma y foto clave en curso.
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
        // Formato de vídeo que permite tomar fotos a resolución completa del sensor durante la sesión.
        if let format = ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing {
            config.videoFormat = format
        }
        return config
    }

    // MARK: - ARSessionDelegate (hilo principal)

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        updateTrackingHint(frame.camera.trackingState)

        frameCounter += 1
        if frameCounter % 30 == 0 {
            meshAnchorCount = frame.anchors.reduce(0) { $0 + ($1 is ARMeshAnchor ? 1 : 0) }
        }

        if phase == .scanning {
            considerKeyframe(frame, session: session)
        } else {
            previousTransform = nil
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
            DispatchQueue.main.async {
                self.isIntegrating = false
                self.voxelCount = voxels
                self.isVolumeFull = full
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

    // MARK: - Fotos clave (hilo principal → cola)

    /// Toma una foto de alta resolución cuando la cámara ha cambiado lo suficiente de posición o ángulo
    /// y está casi quieta en ese momento (sin desenfoque de movimiento).
    private func considerKeyframe(_ frame: ARFrame, session: ARSession) {
        let transform = frame.camera.transform
        let timestamp = frame.timestamp
        defer {
            previousTransform = transform
            previousTimestamp = timestamp
        }
        guard !isCapturingKeyframe, case .normal = frame.camera.trackingState,
              let previous = previousTransform, timestamp > previousTimestamp else { return }

        let dt = Float(timestamp - previousTimestamp)
        let speed = simd_distance(Self.position(of: transform), Self.position(of: previous)) / dt
        let angularSpeed = Self.rotationAngle(from: previous, to: transform) / dt
        guard speed < 0.35, angularSpeed < 0.5 else { return }   // 35 cm/s y ~30°/s

        if let last = lastKeyframeTransform {
            let moved = simd_distance(Self.position(of: transform), Self.position(of: last)) > 0.2 * keyframeSpacing
            let turned = Self.rotationAngle(from: last, to: transform) > (.pi / 12) * keyframeSpacing
            guard moved || turned else { return }
        }
        guard let sceneDepth = frame.smoothedSceneDepth ?? frame.sceneDepth else { return }
        let snapshot = Self.copyDepth(sceneDepth)
        let depthCamera = CameraModel(intrinsics: frame.camera.intrinsics, cameraToWorld: transform)
            .scaled(sx: Float(snapshot.width) / Float(frame.camera.imageResolution.width),
                    sy: Float(snapshot.height) / Float(frame.camera.imageResolution.height))

        lastKeyframeTransform = transform
        isCapturingKeyframe = true

        let store: (CVPixelBuffer, ARCamera) -> Void = { [weak self] buffer, camera in
            guard let self else { return }
            self.queue.async {
                self.storeKeyframe(buffer: buffer, camera: camera, depth: snapshot, depthCamera: depthCamera)
                let count = self.keyframes.count
                let spacing = pow(1.5, Float(self.keyframes.decimations))
                DispatchQueue.main.async {
                    self.isCapturingKeyframe = false
                    self.keyframeCount = count
                    self.keyframeSpacing = spacing
                }
            }
        }

        if Self.supportsHighResolutionPhotos {
            session.captureHighResolutionFrame { highRes, _ in
                if let highRes {
                    store(highRes.capturedImage, highRes.camera)
                } else {
                    store(frame.capturedImage, frame.camera)
                }
            }
        } else {
            store(frame.capturedImage, frame.camera)
        }
    }

    private func storeKeyframe(buffer: CVPixelBuffer, camera: ARCamera,
                               depth: (depth: [Float], confidence: [UInt8], width: Int, height: Int),
                               depthCamera: CameraModel) {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let resolution = camera.imageResolution
        let fullCamera = CameraModel(intrinsics: camera.intrinsics, cameraToWorld: camera.transform)
            .scaled(sx: Float(width) / Float(resolution.width), sy: Float(height) / Float(resolution.height))

        let step = max(1, Int((Float(width) / 1000).rounded()))
        guard let small = ImageSampler.downsampledRGB(from: buffer, step: step) else { return }
        let smallCamera = fullCamera.scaled(sx: Float(small.width) / Float(width), sy: Float(small.height) / Float(height))

        // Foto completa a JPEG en disco: en memoria ocuparía ~36 MB cada una.
        var url: URL? = keyframeFolder.appendingPathComponent("\(UUID().uuidString).jpg")
        let quality = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        do {
            try ciContext.writeJPEGRepresentation(of: CIImage(cvPixelBuffer: buffer), to: url!,
                                                  colorSpace: CGColorSpaceCreateDeviceRGB(),
                                                  options: [quality: 0.92])
        } catch {
            url = nil
        }

        // Los píxeles de profundidad poco fiables se marcan como 0 para no usarlos en la prueba de oclusión.
        var occlusionDepth = depth.depth
        for i in occlusionDepth.indices where depth.confidence[i] < UInt8(ARConfidenceLevel.medium.rawValue) {
            occlusionDepth[i] = 0
        }
        keyframes.add(Keyframe(image: small, imageCamera: smallCamera,
                               depth: occlusionDepth, depthWidth: depth.width, depthHeight: depth.height,
                               depthCamera: depthCamera,
                               fullImageURL: url, fullCamera: fullCamera, fullWidth: width, fullHeight: height))
    }

    private static func position(of t: simd_float4x4) -> SIMD3<Float> {
        SIMD3(t.columns.3.x, t.columns.3.y, t.columns.3.z)
    }

    /// Ángulo (radianes) de la rotación relativa entre dos poses.
    private static func rotationAngle(from a: simd_float4x4, to b: simd_float4x4) -> Float {
        let r = simd_mul(a.inverse, b)
        let trace = r.columns.0.x + r.columns.1.y + r.columns.2.z
        return acos(max(-1, min(1, (trace - 1) / 2)))
    }

    // MARK: - Integración (cola de procesado)

    private static func copyDepth(_ sceneDepth: ARDepthData) -> (depth: [Float], confidence: [UInt8], width: Int, height: Int) {
        let depthMap = sceneDepth.depthMap
        let w = CVPixelBufferGetWidth(depthMap)
        let h = CVPixelBufferGetHeight(depthMap)
        var depth = [Float](repeating: 0, count: w * h)
        var confidence = [UInt8](repeating: UInt8(ARConfidenceLevel.high.rawValue), count: w * h)

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(depthMap) {
            let rowBytes = CVPixelBufferGetBytesPerRow(depthMap)
            for y in 0..<h {
                let row = (base + y * rowBytes).assumingMemoryBound(to: Float32.self)
                for x in 0..<w { depth[y * w + x] = row[x] }
            }
        }
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)

        if let confMap = sceneDepth.confidenceMap {
            CVPixelBufferLockBaseAddress(confMap, .readOnly)
            if let base = CVPixelBufferGetBaseAddress(confMap) {
                let rowBytes = CVPixelBufferGetBytesPerRow(confMap)
                for y in 0..<h {
                    let row = (base + y * rowBytes).assumingMemoryBound(to: UInt8.self)
                    for x in 0..<w { confidence[y * w + x] = row[x] }
                }
            }
            CVPixelBufferUnlockBaseAddress(confMap, .readOnly)
        }
        return (depth, confidence, w, h)
    }

    /// Descarta los "píxeles volantes": en los bordes de los objetos el LiDAR mezcla primer plano y fondo
    /// y devuelve distancias intermedias que no existen. Se detectan por el salto con sus vecinos.
    private static func removeFlyingPixels(_ depth: [Float], width w: Int, height h: Int) -> [Float] {
        var out = [Float](repeating: 0, count: depth.count)
        guard w > 2, h > 2 else { return out }
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = y * w + x
                let d = depth[i]
                guard d > 0 else { continue }
                let jump = max(max(abs(d - depth[i - 1]), abs(d - depth[i + 1])),
                               max(abs(d - depth[i - w]), abs(d - depth[i + w])))
                if jump < 0.02 + 0.03 * d { out[i] = d }
            }
        }
        return out
    }

    private func integrate(_ frame: ARFrame, settings: ScanSettings) {
        guard case .normal = frame.camera.trackingState,
              let sceneDepth = frame.smoothedSceneDepth ?? frame.sceneDepth,
              let rgb = ImageSampler.downsampledRGB(from: frame.capturedImage, step: 3) else { return }

        let imageSize = frame.camera.imageResolution
        let baseCamera = CameraModel(intrinsics: frame.camera.intrinsics, cameraToWorld: frame.camera.transform)
        let raw = Self.copyDepth(sceneDepth)
        let depthCamera = baseCamera.scaled(sx: Float(raw.width) / Float(imageSize.width),
                                            sy: Float(raw.height) / Float(imageSize.height))
        let depth = Self.removeFlyingPixels(raw.depth, width: raw.width, height: raw.height)

        // Fusión volumétrica: cada lectura LiDAR refina la distancia a la superficie y su color.
        volume.integrate(depth: depth, confidence: raw.confidence, width: raw.width, height: raw.height,
                         camera: depthCamera, rgb: rgb,
                         maxDepth: settings.maxDepth,
                         minConfidence: UInt8(settings.minConfidence.rawValue))
    }

    // MARK: - Post-proceso y exportación (cola de procesado)

    private func exportScan(meshAnchors: [ARMeshAnchor], settings: ScanSettings) throws -> ([URL], [SceneObjectInfo]) {
        let folder = try ScanStorage.newScanFolder(prefix: "Entorno")
        var files: [URL] = []
        var objects: [SceneObjectInfo] = []

        if settings.fillHoles {
            report("Rellenando huecos", 0.02)
            volume.fillHoles(iterations: 4) { [weak self] p in self?.report(nil, 0.02 + 0.06 * p) }
        }

        report("Generando superficie", 0.08)
        var mesh = volume.extractMesh(minWeight: 1)
        if mesh.isEmpty {
            // Sin datos de profundidad útiles: se recurre a la malla de ARKit.
            mesh = MeshBuilder.build(from: meshAnchors)
        }

        report("Eliminando ruido", 0.12)
        MeshCleaner.removeSmallComponents(&mesh, minTriangles: 300)
        MeshCleaner.smooth(&mesh, iterations: 4)
        MeshCleaner.recomputeNormals(&mesh)
        guard !mesh.isEmpty else {
            try? FileManager.default.removeItem(at: folder)
            throw ScanError.nothingCaptured
        }

        report("Coloreando con \(keyframes.count) fotos", 0.15)
        MeshColorizer.colorize(&mesh, keyframes: keyframes.frames) { [weak self] p in
            self?.report(nil, 0.15 + 0.2 * p)
        }

        report("Exportando malla", 0.35)
        let ply = folder.appendingPathComponent("malla_color.ply")
        try ModelExporter.writePLY(mesh: mesh, to: ply)
        files.append(ply)
        let cloud = folder.appendingPathComponent("nube_puntos.ply")
        try ModelExporter.writePointCloudPLY(mesh: mesh, to: cloud)
        files.append(cloud)

        // Textura fotográfica: cada zona de la malla se recubre con la mejor foto de 12 MP.
        report("Creando textura de alta resolución", 0.38)
        let textureURL = folder.appendingPathComponent("textura.jpg")
        let textured = TextureBaker.bake(mesh, keyframes: keyframes.frames, atlasSize: settings.atlasSize,
                                         outputURL: textureURL) { [weak self] p in
            self?.report(nil, 0.38 + 0.37 * p)
        }
        let textureImage = textured != nil ? UIImage(contentsOfFile: textureURL.path) : nil
        let materialFile = textured != nil ? "modelo.mtl" : nil

        report("Exportando modelo", 0.76)
        if textured != nil {
            try ModelExporter.writeMTL(to: folder.appendingPathComponent("modelo.mtl"), textureFile: "textura.jpg")
            files.append(textureURL)
        }
        let model = textured ?? mesh
        let obj = folder.appendingPathComponent("modelo.obj")
        try ModelExporter.writeOBJ(parts: [.init(name: "Escaneo", mesh: model)], to: obj, materialFile: materialFile)
        files.append(obj)
        let usdz = folder.appendingPathComponent("modelo.usdz")
        if (try? ModelExporter.writeUSDZ(parts: [.init(name: "Escaneo", mesh: model)], texture: textureImage, to: usdz)) != nil {
            files.append(usdz)
        }

        if settings.detectObjects {
            report("Detectando objetos", 0.8)
            let classifier = SurfaceClassifier(anchors: meshAnchors)
            if !classifier.isEmpty {
                var segments = SceneSegmenter.segment(mesh, classifier: classifier)

                // IA de visión: identifica cada objeto en la foto clave donde mejor se ve y lo renombra.
                report("Identificando objetos con IA", 0.84)
                var recognitions: [Int: ObjectRecognizer.Result] = [:]
                var counters: [String: Int] = [:]
                for i in segments.indices where segments[i].kind == .object {
                    let result = ObjectRecognizer.recognize(segments[i].mesh, keyframes: keyframes.frames)
                    recognitions[i] = result
                    let base = result?.label ?? segments[i].label.displayName
                    counters[base, default: 0] += 1
                    segments[i].name = "\(base) \(counters[base]!)"
                }

                // Cada parte (con textura si la hay) como PLY propio, para el visor y para editarla por separado.
                report("Exportando escena por objetos", 0.9)
                let parts = segments.map { segment in
                    ModelExporter.Part(name: segment.name,
                                       mesh: textured?.submesh(triangles: segment.triangles) ?? segment.mesh)
                }
                let objectsFolder = folder.appendingPathComponent("objetos", isDirectory: true)
                try FileManager.default.createDirectory(at: objectsFolder, withIntermediateDirectories: true)
                for part in parts {
                    let url = objectsFolder.appendingPathComponent("\(ModelExporter.objectName(part.name)).ply")
                    try ModelExporter.writePLY(mesh: part.mesh, to: url, textureFile: textured != nil ? "../textura.jpg" : nil)
                }

                let segmentedOBJ = folder.appendingPathComponent("escena_objetos.obj")
                try ModelExporter.writeOBJ(parts: parts, to: segmentedOBJ, materialFile: materialFile)
                files.append(segmentedOBJ)

                let segmentedUSDZ = folder.appendingPathComponent("escena_objetos.usdz")
                if (try? ModelExporter.writeUSDZ(parts: parts, texture: textureImage, to: segmentedUSDZ)) != nil {
                    files.append(segmentedUSDZ)
                }

                objects = segments.enumerated().map { i, segment in
                    SceneSegmenter.info(for: segment,
                                        file: "objetos/\(ModelExporter.objectName(segment.name)).ply",
                                        recognition: recognitions[i])
                }
                let json = folder.appendingPathComponent("escena.json")
                try ModelExporter.writeJSON(SceneDescription(objetos: objects), to: json)
                files.append(json)
            }
        }

        keyframes.reset()   // borra las fotos temporales
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
