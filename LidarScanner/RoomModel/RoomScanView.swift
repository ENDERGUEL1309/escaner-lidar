import ARKit
import RoomPlan
import SwiftUI
import simd

/// Convierte el resultado de RoomPlan (cajas) en muebles paramétricos por piezas, pegados a sus paredes.
enum RoomModelBuilder {
    private static func kind(for category: CapturedRoom.Object.Category) -> FurnitureKind {
        switch category {
        case .table: .table
        case .bed: .bed
        case .sofa: .sofa
        case .chair: .chair
        case .storage: .storage
        default: .appliance
        }
    }

    private static func position(_ m: simd_float4x4) -> SIMD3<Float> {
        SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
    }

    static func build(from room: CapturedRoom) -> RoomModel {
        var model = RoomModel()
        let wallCenters = room.walls.map { position($0.transform) }
        let middle = wallCenters.isEmpty ? SIMD3<Float>(repeating: 0)
            : wallCenters.reduce(SIMD3<Float>(repeating: 0), +) / Float(wallCenters.count)

        // RoomPlan no da el suelo directamente: es la base más baja entre paredes y objetos.
        let bases = room.walls.map { position($0.transform).y - $0.dimensions.y / 2 }
            + room.objects.map { position($0.transform).y - $0.dimensions.y / 2 }
        model.floorY = bases.min() ?? 0

        func inwardNormal(_ surface: CapturedRoom.Surface) -> SIMD3<Float> {
            let center = position(surface.transform)
            var normal = simd_normalize(SIMD3(surface.transform.columns.2.x, 0, surface.transform.columns.2.z))
            if simd_dot(middle - center, normal) < 0 { normal = -normal }
            return normal
        }

        model.walls = room.walls.map { wall in
            RoomWall(center: position(wall.transform), width: wall.dimensions.x, height: wall.dimensions.y,
                     normal: inwardNormal(wall))
        }
        let openings: [(OpeningKind, [CapturedRoom.Surface])] = [(.door, room.doors), (.window, room.windows),
                                                                  (.opening, room.openings)]
        model.openings = openings.flatMap { kind, surfaces in
            surfaces.map { s in
                RoomOpening(kind: kind, center: position(s.transform), width: s.dimensions.x, height: s.dimensions.y,
                            normal: inwardNormal(s))
            }
        }
        // Área del suelo: fórmula del polígono con las esquinas que da RoomPlan (en el plano local del suelo).
        if let floor = room.floors.first, floor.polygonCorners.count >= 3 {
            let c = floor.polygonCorners
            var twice = SIMD3<Float>(repeating: 0)      // vale en cualquier plano: |Σ a×b| / 2
            for i in c.indices { twice += simd_cross(c[i], c[(i + 1) % c.count]) }
            model.floorArea = simd_length(twice) / 2
        }

        var counters: [FurnitureKind: Int] = [:]
        for object in room.objects {
            let c = position(object.transform)
            let yaw = atan2(object.transform.columns.2.x, object.transform.columns.2.z)
            var kind = Self.kind(for: object.category)
            // RoomPlan no distingue escritorios: una mesa estrecha contra una pared lo es.
            if kind == .table, object.dimensions.z < 0.9,
               model.walls.contains(where: { abs(simd_dot(c - $0.center, $0.normal)) < object.dimensions.z / 2 + 0.2 }) {
                kind = .desk
            }
            counters[kind, default: 0] += 1
            let furniture = ParametricFurniture(name: "\(kind.displayName) \(counters[kind]!)", kind: kind,
                                                position: SIMD3(c.x, model.floorY, c.z), yaw: yaw, size: object.dimensions)
            model.furniture.append(model.snappedToWall(furniture))
        }
        return model
    }
}

/// Toma fotogramas de la sesión de RoomPlan solo si aportan: paralaje suficiente (≥ 12° o ≥ 0,35 m) y nitidez.
private final class RoomKeyframeSampler {
    private let queue = DispatchQueue(label: "lidar.room.fotogramas", qos: .utility)
    private let store = KeyframeStore(maxCount: 30)      // solo desde `queue`
    private var sharpnessMean: Float = 0                 // solo desde `queue`
    private var lastTransform: simd_float4x4?            // solo desde main
    private var isBusy = false                           // solo desde main

    private static func angle(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
        let r = simd_mul(a.inverse, b)
        return acos(max(-1, min(1, (r.columns.0.x + r.columns.1.y + r.columns.2.z - 1) / 2)))
    }

    func consider(_ frame: ARFrame) {
        guard !isBusy, case .normal = frame.camera.trackingState else { return }
        let transform = frame.camera.transform
        if let last = lastTransform {
            let moved = simd_distance(transform.columns.3, last.columns.3) >= 0.35
            guard moved || Self.angle(last, transform) >= 12 * .pi / 180 else { return }
        }
        isBusy = true
        let buffer = frame.capturedImage
        let camera = frame.camera
        let timestamp = frame.timestamp
        let depth = (frame.smoothedSceneDepth ?? frame.sceneDepth).map { SceneScanner.copyDepth($0) }
        queue.async { [weak self] in
            guard let self else { return }
            let accepted = self.add(buffer: buffer, camera: camera, depth: depth, timestamp: timestamp)
            DispatchQueue.main.async {
                self.isBusy = false
                if accepted { self.lastTransform = transform }
            }
        }
    }

    /// Varianza del laplaciano: una foto movida tiene mucho menos detalle que la media de las anteriores.
    private static func sharpness(_ image: RGBImage) -> Float {
        let w = image.width, h = image.height
        guard w > 2, h > 2 else { return 0 }
        func g(_ x: Int, _ y: Int) -> Float { Float(image.pixels[(y * w + x) * 3 + 1]) }
        var sum: Float = 0, squares: Float = 0, n: Float = 0
        for y in stride(from: 1, to: h - 1, by: 3) {
            for x in stride(from: 1, to: w - 1, by: 3) {
                let l = g(x - 1, y) + g(x + 1, y) + g(x, y - 1) + g(x, y + 1) - 4 * g(x, y)
                sum += l
                squares += l * l
                n += 1
            }
        }
        let mean = sum / n
        return squares / n - mean * mean
    }

    private func add(buffer: CVPixelBuffer, camera: ARCamera,
                     depth: (depth: [Float], confidence: [UInt8], width: Int, height: Int)?, timestamp: TimeInterval) -> Bool {
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        guard let image = ImageSampler.downsampledRGB(from: buffer, step: max(1, width / 1000)) else { return false }
        let sharp = Self.sharpness(image)
        guard store.count < 3 || sharp >= sharpnessMean * 0.5 else { return false }
        let count = Float(store.count)
        sharpnessMean = (sharpnessMean * count + sharp) / (count + 1)

        let resolution = camera.imageResolution
        let full = CameraModel(intrinsics: camera.intrinsics, cameraToWorld: camera.transform)
            .scaled(sx: Float(width) / Float(resolution.width), sy: Float(height) / Float(resolution.height))
        let small = full.scaled(sx: Float(image.width) / Float(width), sy: Float(image.height) / Float(height))
        var measured: [Float] = []
        if let depth {
            measured = depth.depth
            for i in measured.indices where depth.confidence[i] < UInt8(ARConfidenceLevel.medium.rawValue) { measured[i] = 0 }
        }
        let depthWidth = depth?.width ?? 0, depthHeight = depth?.height ?? 0
        let depthCamera = depthWidth > 0
            ? full.scaled(sx: Float(depthWidth) / Float(width), sy: Float(depthHeight) / Float(height)) : full
        store.add(Keyframe(image: image, imageCamera: small, depth: measured, depthWidth: depthWidth, depthHeight: depthHeight,
                           depthCamera: depthCamera, fullImageURL: nil, fullCamera: full,
                           fullWidth: width, fullHeight: height, timestamp: timestamp))
        return true
    }

    func collected() -> [Keyframe] { queue.sync { store.frames } }
}

/// Controlador de RoomPlan: su propia sesión de ARKit, su guía en pantalla y su resultado procesado.
final class RoomCaptureViewController: UIViewController, RoomCaptureViewDelegate {
    var onFinished: ((CapturedRoom, [Keyframe], [ValidationFrame]) -> Void)?
    var onFailed: ((String) -> Void)?
    var onQuality: ((LiveQuality) -> Void)?

    // RoomCaptureViewDelegate hereda de la codificación de objetos, aunque aquí nunca se archiva.
    static var supportsSecureCoding: Bool { false }

    private var captureView: RoomCaptureView!
    private var timer: Timer?
    private var isStopped = false
    private let sampler = RoomKeyframeSampler()
    private let validator = CaptureValidator()

    override func viewDidLoad() {
        super.viewDidLoad()
        captureView = RoomCaptureView(frame: view.bounds)
        captureView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        captureView.delegate = self
        view.addSubview(captureView)
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        captureView.captureSession.run(configuration: RoomCaptureSession.Configuration())
        validator.onQuality = { [weak self] quality in self?.onQuality?(quality) }
        validator.start()
        timer = Timer.scheduledTimer(timeInterval: 0.3, target: self, selector: #selector(sampleFrame), userInfo: nil, repeats: true)
    }

    @objc private func sampleFrame() {
        if let frame = captureView.captureSession.arSession.currentFrame {
            sampler.consider(frame)
            validator.consider(frame)
        }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        stopCapture()
    }

    func stopCapture() {
        guard !isStopped else { return }
        isStopped = true
        timer?.invalidate()
        timer = nil
        validator.stop()
        captureView?.captureSession.stop()
    }

    func captureView(shouldPresent roomDataForProcessing: CapturedRoomData, error: Error?) -> Bool {
        if let error { onFailed?(error.localizedDescription) }
        return error == nil
    }

    func captureView(didPresent processedResult: CapturedRoom, error: Error?) {
        if let error {
            onFailed?(error.localizedDescription)
        } else {
            onFinished?(processedResult, sampler.collected(), validator.collected())
        }
    }
}

private struct RoomCaptureContainer: UIViewControllerRepresentable {
    @Binding var stopRequested: Bool
    let onFinished: (CapturedRoom, [Keyframe], [ValidationFrame]) -> Void
    let onFailed: (String) -> Void
    let onQuality: (LiveQuality) -> Void

    func makeUIViewController(context: Context) -> RoomCaptureViewController {
        let controller = RoomCaptureViewController()
        controller.onFinished = onFinished
        controller.onFailed = onFailed
        controller.onQuality = onQuality
        return controller
    }

    func updateUIViewController(_ controller: RoomCaptureViewController, context: Context) {
        if stopRequested { controller.stopCapture() }
    }
}

/// Semáforo en vivo de cada fuente: lo que está verde ya está aportando datos fiables en este instante.
private struct LiveQualityPanel: View {
    let quality: LiveQuality

    private func chip(_ title: String, _ icon: String, _ ok: Bool) -> some View {
        Label(title, systemImage: icon)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 8).padding(.vertical, 5)
            .background((ok ? Color.green : Color.red).opacity(0.85), in: Capsule())
            .foregroundStyle(.white)
    }

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 4) {
                chip("Cámara", "camera", quality.sharp)
                chip("Luz", "sun.max", quality.light)
                chip("LiDAR \(Int(quality.lidar * 100))%", "dot.radiowaves.left.and.right", quality.lidarOK)
            }
            HStack(spacing: 4) {
                chip(String(format: "%.1f m", quality.distance), "ruler", quality.distanceOK)
                chip("Giro", "gyroscope", quality.steady)
                chip("Posición", "location", quality.tracking && quality.imuAgrees)
            }
            HStack(spacing: 4) {
                chip(quality.colorAgreement.map { "Color \(Int($0 * 100))%" } ?? "Color —", "paintpalette", quality.colorOK)
                chip("Reflejos", "sun.haze", quality.glare <= 0.08)
            }
            if let hint = quality.hint {
                Text(hint)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(.yellow.opacity(0.9), in: Capsule())
                    .foregroundStyle(.black)
            }
        }
        .padding(.horizontal)
        .allowsHitTesting(false)
    }
}

/// Pantalla «Modelar habitación»: captura con RoomPlan y pasa al editor con los muebles por piezas.
struct RoomScanView: View {
    static var isSupported: Bool { RoomCaptureSession.isSupported }

    @State private var stopRequested = false
    @State private var editor: RoomEditorModel?
    @State private var failure: String?
    @State private var quality = LiveQuality()
    @State private var isValidating = false

    var body: some View {
        ZStack(alignment: .bottom) {
            if let editor {
                RoomEditorView(model: editor)
            } else if let failure {
                ContentUnavailableView("No se pudo capturar la habitación", systemImage: "exclamationmark.triangle",
                                       description: Text(failure))
            } else {
                RoomCaptureContainer(stopRequested: $stopRequested) { room, keyframes, frames in
                    isValidating = true
                    let model = RoomModelBuilder.build(from: room)
                    Task {
                        let validated = await Task.detached(priority: .userInitiated) {
                            CaptureValidator.validate(room: model, frames: frames)
                        }.value
                        editor = RoomEditorModel(room: validated, keyframes: keyframes, validationFrames: frames)
                        isValidating = false
                    }
                } onFailed: { message in
                    failure = message
                } onQuality: { q in
                    quality = q
                }
                .ignoresSafeArea()
                if isValidating {
                    ProgressView("Validando paredes y muebles con todos los sensores…")
                        .padding()
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                        .frame(maxHeight: .infinity)
                }
                if !stopRequested {
                    LiveQualityPanel(quality: quality)
                        .frame(maxHeight: .infinity, alignment: .top)
                        .padding(.top, 8)
                    Button { stopRequested = true } label: {
                        Label("Terminar habitación", systemImage: "checkmark.circle.fill").font(.headline)
                    }
                    .buttonStyle(.borderedProminent)
                    .padding(.bottom, 30)
                }
            }
        }
        .navigationTitle("Modelar habitación")
        .navigationBarTitleDisplayMode(.inline)
    }
}
