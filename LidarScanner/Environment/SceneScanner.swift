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
        /// Modo por ubicaciones: caminando hacia la siguiente ubicación (no se registra, pero se sigue la posición).
        case moving
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
    /// Ubicación actual en el modo por ubicaciones (1, 2, 3…).
    var stationCount = 0
    /// La cámara se mueve demasiado rápido: no se registra hasta que vaya más despacio.
    var isTooFast = false

    /// Se llama con la posición de cada nueva ubicación, para dibujar su marcador en la cámara.
    @ObservationIgnored var onStationAdded: ((SIMD3<Float>) -> Void)?
    /// Se llama al empezar un escaneo nuevo, para quitar los marcadores del anterior.
    @ObservationIgnored var onStationsReset: (() -> Void)?

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
    @ObservationIgnored private var motionTransform: simd_float4x4?
    @ObservationIgnored private var motionTimestamp: TimeInterval = 0
    @ObservationIgnored private var lastSplatFrameTime: TimeInterval = 0
    @ObservationIgnored private var lastSplatFrameTransform: simd_float4x4?
    @ObservationIgnored private var isSavingSplatFrame = false
    @ObservationIgnored private var splatFrameCount = 0
    // Fotogramas para Gaussian Splatting: se codifican en su propia cola para no frenar la fusión.
    @ObservationIgnored private let splatQueue = DispatchQueue(label: "lidar.scan.fotogramas", qos: .utility)
    @ObservationIgnored private var splatRecords: [CloudExport.Photo] = []          // solo desde `splatQueue`
    @ObservationIgnored private let splatFolder = FileManager.default.temporaryDirectory
        .appendingPathComponent("Fotogramas", isDirectory: true)
    @ObservationIgnored private var linearSpeed: Float = 0     // m/s, suavizada
    @ObservationIgnored private var angularSpeed: Float = 0    // rad/s, suavizada

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
        splatFrameCount = 0
        lastSplatFrameTransform = nil
        let frames = splatFolder
        splatQueue.async { [weak self] in
            self?.splatRecords = []
            try? FileManager.default.removeItem(at: frames)
            try? FileManager.default.createDirectory(at: frames, withIntermediateDirectories: true)
        }
        // Descarta la malla que ARKit haya construido antes de pulsar «Iniciar», sin perder el seguimiento.
        session.run(makeConfiguration(), options: [.resetSceneReconstruction])
        phase = .scanning
        stationCount = 0
        onStationsReset?()
        if settings.stationMode { markStation() }
    }

    func pause() { if phase == .scanning { phase = .paused } }
    func resume() { if phase == .paused { phase = .scanning } }

    /// Modo por ubicaciones: deja de registrar para caminar hasta la siguiente ubicación.
    func goToNextStation() {
        if phase == .scanning { phase = .moving }
    }

    /// Modo por ubicaciones: empieza a registrar desde la ubicación actual.
    func scanHere() {
        guard phase == .moving else { return }
        phase = .scanning
        markStation()
    }

    private func markStation() {
        stationCount += 1
        if let transform = session?.currentFrame?.camera.transform {
            onStationAdded?(SIMD3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z))
        }
    }

    func stop() { session?.pause() }

    func prepareNewScan() {
        phase = .ready
        if let session { session.run(makeConfiguration(), options: [.resetSceneReconstruction]) }
    }

    func finish() {
        guard let session, phase == .scanning || phase == .paused || phase == .moving else { return }
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

        let tooFast = updateMotion(frame)
        if tooFast != isTooFast { isTooFast = tooFast }

        if phase == .scanning {
            considerKeyframe(frame, session: session)
            if settings.captureSplatFrames, !tooFast { considerSplatFrame(frame) }
        } else {
            previousTransform = nil
        }

        // Moverse rápido emborrona las fotos y desalinea la profundidad: esos fotogramas no se registran.
        guard phase == .scanning, !tooFast, !isIntegrating,
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

    /// Fotograma para Gaussian Splatting: cada ~0,5 s si la cámara se ha movido o girado un poco desde el anterior.
    private func considerSplatFrame(_ frame: ARFrame) {
        guard !isSavingSplatFrame, splatFrameCount < ScanSettings.maxSplatFrames,
              case .normal = frame.camera.trackingState,
              frame.timestamp - lastSplatFrameTime >= 0.5 else { return }
        let transform = frame.camera.transform
        if let last = lastSplatFrameTransform {
            let moved = simd_distance(Self.position(of: transform), Self.position(of: last)) > 0.03
            let turned = Self.rotationAngle(from: last, to: transform) > 4 * .pi / 180
            guard moved || turned else { return }
        }
        isSavingSplatFrame = true
        lastSplatFrameTime = frame.timestamp
        lastSplatFrameTransform = transform

        let buffer = frame.capturedImage
        let camera = frame.camera
        let timestamp = frame.timestamp
        let index = splatFrameCount
        splatQueue.async { [weak self] in
            guard let self else { return }
            let saved = self.saveSplatFrame(buffer: buffer, camera: camera, timestamp: timestamp, index: index)
            DispatchQueue.main.async {
                self.isSavingSplatFrame = false
                if saved { self.splatFrameCount += 1 }
            }
        }
    }

    /// Reduce el fotograma a 1280 px de ancho y lo guarda en JPEG con su cámara (intrínsecos escalados).
    private func saveSplatFrame(buffer: CVPixelBuffer, camera: ARCamera, timestamp: TimeInterval, index: Int) -> Bool {
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let scale = min(1, 1280 / Float(width))
        let name = String(format: "%04d.jpg", index)
        let image = CIImage(cvPixelBuffer: buffer)
            .transformed(by: CGAffineTransform(scaleX: CGFloat(scale), y: CGFloat(scale)))
        let quality = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        do {
            try ciContext.writeJPEGRepresentation(of: image, to: splatFolder.appendingPathComponent(name),
                                                  colorSpace: CGColorSpaceCreateDeviceRGB(), options: [quality: 0.8])
        } catch {
            return false
        }
        let resolution = camera.imageResolution
        let model = CameraModel(intrinsics: camera.intrinsics, cameraToWorld: camera.transform)
            .scaled(sx: Float(width) / Float(resolution.width) * scale, sy: Float(height) / Float(resolution.height) * scale)
        let m = model.cameraToWorld
        splatRecords.append(CloudExport.Photo(
            archivo: name,
            ancho: Int((Float(width) * scale).rounded()), alto: Int((Float(height) * scale).rounded()),
            fx: model.fx, fy: model.fy, cx: model.cx, cy: model.cy,
            cameraToWorld: [m.columns.0, m.columns.1, m.columns.2, m.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] },
            marcaTiempo: timestamp, profundidad: nil))
        return true
    }

    /// Velocidad de la cámara (suavizada para que el aviso no parpadee). Devuelve si va demasiado rápido:
    /// más de 0,5 m/s caminando o más de ~80°/s girando.
    private func updateMotion(_ frame: ARFrame) -> Bool {
        let transform = frame.camera.transform
        defer {
            motionTransform = transform
            motionTimestamp = frame.timestamp
        }
        guard case .normal = frame.camera.trackingState, let previous = motionTransform,
              frame.timestamp > motionTimestamp else { return false }
        let dt = Float(frame.timestamp - motionTimestamp)
        let linear = simd_distance(Self.position(of: transform), Self.position(of: previous)) / dt
        let angular = Self.rotationAngle(from: previous, to: transform) / dt
        linearSpeed = linearSpeed * 0.85 + linear * 0.15
        angularSpeed = angularSpeed * 0.85 + angular * 0.15
        return linearSpeed > 0.5 || angularSpeed > 1.4
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

        let store: (CVPixelBuffer, ARCamera, TimeInterval) -> Void = { [weak self] buffer, camera, time in
            guard let self else { return }
            self.queue.async {
                self.storeKeyframe(buffer: buffer, camera: camera, depth: snapshot, depthCamera: depthCamera, timestamp: time)
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
                    store(highRes.capturedImage, highRes.camera, highRes.timestamp)
                } else {
                    store(frame.capturedImage, frame.camera, timestamp)
                }
            }
        } else {
            store(frame.capturedImage, frame.camera, timestamp)
        }
    }

    private func storeKeyframe(buffer: CVPixelBuffer, camera: ARCamera,
                               depth: (depth: [Float], confidence: [UInt8], width: Int, height: Int),
                               depthCamera: CameraModel, timestamp: TimeInterval) {
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
                               fullImageURL: url, fullCamera: fullCamera, fullWidth: width, fullHeight: height, timestamp: timestamp))
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

        report("Eliminando ruido", 0.1)
        MeshCleaner.removeSmallComponents(&mesh, minTriangles: 300)
        MeshCleaner.smooth(&mesh, iterations: 4)
        guard !mesh.isEmpty else {
            try? FileManager.default.removeItem(at: folder)
            throw ScanError.nothingCaptured
        }

        // Pulido con la clasificación neuronal de ARKit: suelo, techo y paredes a planos perfectos.
        report("Enderezando paredes y suelo", 0.12)
        let classifier = SurfaceClassifier(anchors: meshAnchors)
        var labels: [SurfaceLabel] = []
        if !classifier.isEmpty {
            labels = SceneSegmenter.labels(for: mesh, classifier: classifier)
            MeshCleaner.flattenStructures(&mesh, labels: labels)
        }
        MeshCleaner.recomputeNormals(&mesh)

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
            if !labels.isEmpty {
                // IA de visión: separa en las fotos los objetos que se tocan entre sí.
                report("Separando objetos con IA", 0.78)
                let signatures = InstanceSignatures.compute(mesh: mesh, include: labels.map { !$0.isStructure },
                                                            keyframes: keyframes.frames)
                report("Detectando objetos", 0.82)
                var segments = SceneSegmenter.segment(mesh, labels: labels, signatures: signatures)

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

        // Fotos de 12 MP con sus cámaras y profundidad, para procesar el escaneo con la IA en la nube.
        if settings.saveCloudPhotos {
            report("Guardando fotos para la IA en la nube", 0.97)
            try? CloudExport.saveKeyframes(keyframes.frames, to: folder)
        }
        if settings.captureSplatFrames {
            report("Guardando fotogramas para la vista fotorrealista", 0.98)
            let records = splatQueue.sync { splatRecords }   // espera a que termine el último fotograma
            try? CloudExport.saveFrames(records, from: splatFolder, to: folder)
        }

        keyframes.reset()   // borra las fotos temporales que queden
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
