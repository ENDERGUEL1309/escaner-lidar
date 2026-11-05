import ARKit
import CoreImage
import CoreMotion
import Foundation
import simd

/// Lo que dice cada fuente durante la grabación, para el panel en vivo.
struct LiveQuality: Equatable {
    var sharp = false          // cámara: foto nítida
    var light = false          // cámara: exposición útil
    var lidar: Float = 0       // LiDAR: fracción de píxeles con confianza media o alta
    var distance: Float = 0    // sensor de profundidad: distancia mediana a lo que se mira
    var steady = false         // giroscopio: giro lento, sin foto movida
    var tracking = false       // posición: seguimiento visual-inercial de ARKit normal
    var imuAgrees = true       // posición frente a giroscopio: el giro de ARKit coincide con el del giroscopio
    var glare: Float = 0       // filtro de color: fracción de píxeles quemados (reflejos)
    /// Profundidad del color frente al LiDAR: fracción que coincide a menos del 5 %. `nil` sin el modelo.
    var colorAgreement: Float?
    var hint: String?

    var colorOK: Bool { (colorAgreement ?? 1) >= 0.6 }

    var lidarOK: Bool { lidar >= 0.6 }
    var distanceOK: Bool { distance >= 0.3 && distance <= 4 }
}

/// Una comprobación independiente sobre una pared o un mueble.
struct ValidationCheck: Codable, Identifiable {
    var id: String { name }
    var name: String
    var passed: Bool
    var detail: String
}

/// Validaciones de un elemento. Cada una sale de una fuente distinta; cuantas más se cumplan, más fiable es.
struct ElementValidation: Codable {
    var checks: [ValidationCheck] = []
    /// Desde dónde volver a grabarlo (posición del iPhone y punto al que mirar), si alguna validación falla.
    var suggestedEye: SIMD3<Float>?
    var suggestedTarget: SIMD3<Float>?

    var passed: Int { checks.filter(\.passed).count }
    var state: VerificationState {
        passed == checks.count ? .green : passed >= checks.count - 2 ? .yellow : .red
    }
    /// Qué hacer para la primera validación que falla.
    var advice: String? { checks.first { !$0.passed }.map { "\($0.name): \($0.detail)" } }
}

/// Registro ligero de un instante de la grabación: pose, profundidad reducida y la nota de cada sensor.
struct ValidationFrame {
    var camera: CameraModel          // intrínsecos ya escalados al mapa `depth`
    var depth: [Float]               // metros, 0 = sin dato o confianza baja
    var width: Int
    var height: Int
    var sharp: Bool
    var light: Bool
    var steady: Bool
    var tracking: Bool
    var imuAgrees: Bool
    /// Profundidad deducida del color, ya en metros (vacía sin el modelo).
    var colorDepth: [Float] = []

    var good: Bool { sharp && light && steady && tracking && imuAgrees }
}

/// Valida lo escaneado cruzando cámara, LiDAR, profundidad, giroscopio, posición y ángulo de cada vista.
///
/// En vivo: cada 0,3 s puntúa el fotograma (nitidez y luz de la cámara, confianza del LiDAR, distancia, giro del
/// giroscopio, seguimiento de ARKit y si el giro de ARKit coincide con el del giroscopio) y guarda un registro.
/// Al terminar: proyecta cada pared y mueble en todos los registros y comprueba vistas, nitidez, ángulos,
/// distancia, confianza del LiDAR, que el LiDAR mida la superficie donde el modelo dice y la estabilidad.
final class CaptureValidator {
    private let queue = DispatchQueue(label: "lidar.validacion", qos: .utility)
    private let motion = CMMotionManager()
    private var frames: [ValidationFrame] = []                 // solo desde `queue`
    private var sharpnessMean: Float = 0                        // solo desde `queue`
    private var sharpnessCount: Float = 0                       // solo desde `queue`
    private var lastPose: simd_quatf?                           // solo desde `queue`
    private var lastAttitude: simd_quatf?                       // solo desde `queue`
    private var isBusy = false                                  // solo desde main
    private lazy var colorDepth = ColorDepth()                  // solo desde `queue`

    // Grabación: fotogramas buenos (foto + LiDAR + pose) y bitácora de sensores, en el formato de los escaneos de
    // entorno (fotogramas/camaras.json), para poder reprocesar la habitación en el PC.
    private var recordingFolder: URL?                           // solo desde `queue`
    private var records: [CloudExport.Photo] = []               // solo desde `queue`
    private var lastSavedPose: simd_float4x4?                   // solo desde `queue`
    private var lastSavedTime: TimeInterval = 0                 // solo desde `queue`
    private var sensorLog: [SensorSample] = []                  // solo desde `queue`
    private let ciContext = CIContext()

    struct SensorSample: Codable {
        var t: Double
        /// Pose de ARKit (posición y cuaternión x, y, z, w).
        var posicion: [Float]
        var orientacion: [Float]
        /// Giroscopio (rad/s), gravedad (g), aceleración del usuario (g) y actitud del sensor de movimiento.
        var giro: [Double]
        var gravedad: [Double]
        var aceleracion: [Double]
        var actitud: [Double]
        var nitida: Bool
        var luz: Bool
        var estable: Bool
        var seguimiento: Bool
        var giroscopioCoincide: Bool
        var lidar: Float
        var distancia: Float
        var reflejos: Float
        var color: Float?
        /// Fotograma guardado en este instante, si lo hubo.
        var fotograma: String?
    }

    func startRecording(in folder: URL) {
        let directory = folder.appendingPathComponent("fotogramas", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        queue.async { self.recordingFolder = directory }
    }

    /// Escribe camaras.json y sensores.json. Devuelve cuántos fotogramas se guardaron.
    @discardableResult
    func finishRecording() -> Int {
        queue.sync {
            guard let directory = recordingFolder else { return 0 }
            recordingFolder = nil
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try? encoder.encode(CloudExport.Cameras(fotos: records))
                .write(to: directory.appendingPathComponent("camaras.json"), options: .atomic)
            try? encoder.encode(sensorLog)
                .write(to: directory.deletingLastPathComponent().appendingPathComponent("sensores.json"), options: .atomic)
            return records.count
        }
    }

    /// Guarda la foto (1280 px) y su LiDAR a resolución completa con la confianza, como SceneScanner.
    private func saveFrame(buffer: CVPixelBuffer, camera: ARCamera, timestamp: TimeInterval,
                           depth: (depth: [Float], confidence: [UInt8], width: Int, height: Int)?, in directory: URL) -> String? {
        let index = records.count
        let name = String(format: "%04d.jpg", index)
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let scale = min(1, 1280 / Float(width))
        let image = CIImage(cvPixelBuffer: buffer)
            .transformed(by: CGAffineTransform(scaleX: CGFloat(scale), y: CGFloat(scale)))
        let quality = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard (try? ciContext.writeJPEGRepresentation(of: image, to: directory.appendingPathComponent(name),
                                                      colorSpace: CGColorSpaceCreateDeviceRGB(),
                                                      options: [quality: 0.85])) != nil else { return nil }
        let resolution = camera.imageResolution
        let model = CameraModel(intrinsics: camera.intrinsics, cameraToWorld: camera.transform)
            .scaled(sx: Float(width) / Float(resolution.width) * scale, sy: Float(height) / Float(resolution.height) * scale)
        var depthInfo: CloudExport.DepthInfo?
        if let depth {
            let clean = SceneScanner.removeFlyingPixels(depth.depth, width: depth.width, height: depth.height)
            let depthName = String(format: "%04d.prof", index)
            let data = CloudExport.depthData(clean, width: depth.width, height: depth.height, confidence: depth.confidence)
            if (try? data.write(to: directory.appendingPathComponent(depthName), options: .atomic)) != nil {
                let d = CameraModel(intrinsics: camera.intrinsics, cameraToWorld: camera.transform)
                    .scaled(sx: Float(depth.width) / Float(resolution.width), sy: Float(depth.height) / Float(resolution.height))
                depthInfo = CloudExport.DepthInfo(archivo: depthName, ancho: depth.width, alto: depth.height,
                                                  fx: d.fx, fy: d.fy, cx: d.cx, cy: d.cy)
            }
        }
        let m = model.cameraToWorld
        records.append(CloudExport.Photo(
            archivo: name,
            ancho: Int((Float(width) * scale).rounded()), alto: Int((Float(height) * scale).rounded()),
            fx: model.fx, fy: model.fy, cx: model.cx, cy: model.cy,
            cameraToWorld: [m.columns.0, m.columns.1, m.columns.2, m.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] },
            marcaTiempo: timestamp, profundidad: depthInfo))
        return name
    }
    private static let maxFrames = 1500

    var onQuality: ((LiveQuality) -> Void)?

    func start() {
        guard motion.isDeviceMotionAvailable else { return }
        motion.deviceMotionUpdateInterval = 1 / 60
        motion.startDeviceMotionUpdates()
    }

    func stop() {
        motion.stopDeviceMotionUpdates()
    }

    /// Llamar desde el hilo principal con el fotograma actual de ARKit.
    func consider(_ frame: ARFrame) {
        guard !isBusy else { return }
        isBusy = true
        let buffer = frame.capturedImage
        let camera = frame.camera
        let depth = (frame.smoothedSceneDepth ?? frame.sceneDepth).map { SceneScanner.copyDepth($0) }
        let device = motion.deviceMotion
        let timestamp = frame.timestamp
        let rate = device.map { Float(simd_length(SIMD3($0.rotationRate.x, $0.rotationRate.y, $0.rotationRate.z))) } ?? 0
        let attitude = device.map { q in
            simd_quatf(ix: Float(q.attitude.quaternion.x), iy: Float(q.attitude.quaternion.y),
                       iz: Float(q.attitude.quaternion.z), r: Float(q.attitude.quaternion.w))
        }
        var tracking = false
        if case .normal = camera.trackingState { tracking = true }
        queue.async { [weak self] in
            guard let self else { return }
            let quality = self.record(buffer: buffer, camera: camera, depth: depth, rate: rate,
                                      attitude: attitude, tracking: tracking, timestamp: timestamp, device: device)
            DispatchQueue.main.async {
                self.isBusy = false
                self.onQuality?(quality)
            }
        }
    }

    /// Nitidez (varianza del laplaciano), brillo medio y reflejos (píxeles quemados) del plano de luminancia.
    private static func luma(_ buffer: CVPixelBuffer) -> (sharpness: Float, brightness: Float, glare: Float) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard CVPixelBufferGetPlaneCount(buffer) >= 1,
              let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else { return (0, 0, 0) }
        let w = CVPixelBufferGetWidthOfPlane(buffer, 0), h = CVPixelBufferGetHeightOfPlane(buffer, 0)
        let row = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
        let p = base.assumingMemoryBound(to: UInt8.self)
        func g(_ x: Int, _ y: Int) -> Float { Float(p[y * row + x]) }
        var sum: Float = 0, squares: Float = 0, bright: Float = 0, burnt: Float = 0, n: Float = 0
        for y in stride(from: h / 6, to: h * 5 / 6, by: 4) {
            for x in stride(from: w / 6, to: w * 5 / 6, by: 4) {
                let l = g(x - 1, y) + g(x + 1, y) + g(x, y - 1) + g(x, y + 1) - 4 * g(x, y)
                sum += l
                squares += l * l
                bright += g(x, y)
                if g(x, y) > 250 { burnt += 1 }
                n += 1
            }
        }
        guard n > 0 else { return (0, 0, 0) }
        let mean = sum / n
        return (squares / n - mean * mean, bright / n, burnt / n)
    }

    private static func angle(_ q: simd_quatf) -> Float {
        2 * acos(min(1, abs(q.real)))
    }

    private func record(buffer: CVPixelBuffer, camera: ARCamera,
                        depth: (depth: [Float], confidence: [UInt8], width: Int, height: Int)?,
                        rate: Float, attitude: simd_quatf?, tracking: Bool,
                        timestamp: TimeInterval, device: CMDeviceMotion?) -> LiveQuality {
        var q = LiveQuality()
        q.tracking = tracking

        // Cámara: nitidez relativa a la media de la grabación (depende de la escena) y exposición.
        let (sharpness, brightness, glare) = Self.luma(buffer)
        q.glare = glare
        q.sharp = sharpnessCount < 5 || sharpness >= sharpnessMean * 0.6
        sharpnessMean = (sharpnessMean * sharpnessCount + sharpness) / (sharpnessCount + 1)
        sharpnessCount = min(sharpnessCount + 1, 200)
        q.light = brightness > 35 && brightness < 225

        // Giroscopio: girar rápido emborrona la foto y hace saltar la profundidad.
        q.steady = rate < 0.8

        // Posición frente a giroscopio: el ángulo girado entre dos registros debe coincidir (no depende de los ejes).
        let pose = simd_quatf(camera.transform)
        if let lastPose, let lastAttitude, let attitude {
            let ar = Self.angle(pose * lastPose.inverse)
            let imu = Self.angle(attitude * lastAttitude.inverse)
            q.imuAgrees = abs(ar - imu) < 2 * .pi / 180
        }
        lastPose = pose
        lastAttitude = attitude

        // LiDAR: confianza y distancia. Se guarda reducido a 1/4 (64×48) sin las lecturas de confianza baja.
        var reduced: [Float] = []
        var rw = 0, rh = 0
        if let depth {
            rw = depth.width / 4
            rh = depth.height / 4
            reduced = [Float](repeating: 0, count: rw * rh)
            var confident = 0, values: [Float] = []
            values.reserveCapacity(rw * rh)
            for y in 0..<rh {
                for x in 0..<rw {
                    let i = (y * 4) * depth.width + x * 4
                    guard depth.confidence[i] >= UInt8(ARConfidenceLevel.medium.rawValue), depth.depth[i] > 0 else { continue }
                    confident += 1
                    reduced[y * rw + x] = depth.depth[i]
                    values.append(depth.depth[i])
                }
            }
            q.lidar = Float(confident) / Float(max(1, rw * rh))
            if !values.isEmpty { q.distance = values.sorted()[values.count / 2] }
        }

        // Color: profundidad deducida de la foto, ajustada al LiDAR; el desacuerdo marca lo dudoso.
        var colorMeters: [Float] = []
        if rw > 0, let colorDepth, let disparity = colorDepth.disparity(buffer, width: rw, height: rh),
           let aligned = ColorDepth.align(disparity: disparity, lidar: reduced) {
            colorMeters = aligned.depth
            q.colorAgreement = aligned.agreement
        }

        q.hint = !q.tracking ? "Muévete despacio y apunta a zonas con detalle para recuperar la posición"
            : !q.light ? "Luz insuficiente o excesiva: enciende luces o evita contraluces"
            : !q.steady ? "Gira más despacio"
            : !q.sharp ? "Foto movida: sostén el iPhone firme un momento"
            : q.distance > 4 ? "Acércate: a más de 4 m el LiDAR pierde precisión"
            : (q.distance > 0 && q.distance < 0.3) ? "Aléjate un poco: demasiado cerca para el LiDAR"
            : q.glare > 0.08 ? "Reflejos fuertes: cambia el ángulo para que la luz no rebote hacia la cámara"
            : !q.lidarOK ? "El LiDAR no mide bien aquí (cristal, espejo, negro o muy lejos): prueba otro ángulo"
            : !q.colorOK ? "Color y LiDAR no coinciden aquí (cristal, espejo o reflejo): grábalo desde otro ángulo"
            : !q.imuAgrees ? "El seguimiento se desvía del giroscopio: vuelve despacio a una zona ya grabada"
            : nil

        // Grabación: solo fotos que pasan los filtros, y solo si el iPhone se movió (≥ 8 cm o ≥ 6°) o pasó 1 s.
        var savedName: String?
        // Tope de 900 fotogramas (~350 MB): más no mejora la reconstrucción y llena el iPhone.
        if let directory = recordingFolder, records.count < 900, q.sharp, q.light, q.steady, q.tracking, q.glare <= 0.08 {
            let moved = lastSavedPose.map { last in
                simd_distance(last.columns.3, camera.transform.columns.3) >= 0.08
                    || Self.angle(simd_quatf(last) * simd_quatf(camera.transform).inverse) >= 6 * .pi / 180
            } ?? true
            if moved || timestamp - lastSavedTime >= 1, timestamp - lastSavedTime >= 0.25 {
                savedName = saveFrame(buffer: buffer, camera: camera, timestamp: timestamp, depth: depth, in: directory)
                if savedName != nil {
                    lastSavedPose = camera.transform
                    lastSavedTime = timestamp
                }
            }
        }
        if recordingFolder != nil {
            let t = camera.transform.columns.3
            let o = simd_quatf(camera.transform).vector
            sensorLog.append(SensorSample(
                t: timestamp, posicion: [t.x, t.y, t.z], orientacion: [o.x, o.y, o.z, o.w],
                giro: device.map { [$0.rotationRate.x, $0.rotationRate.y, $0.rotationRate.z] } ?? [],
                gravedad: device.map { [$0.gravity.x, $0.gravity.y, $0.gravity.z] } ?? [],
                aceleracion: device.map { [$0.userAcceleration.x, $0.userAcceleration.y, $0.userAcceleration.z] } ?? [],
                actitud: device.map { [$0.attitude.quaternion.x, $0.attitude.quaternion.y,
                                       $0.attitude.quaternion.z, $0.attitude.quaternion.w] } ?? [],
                nitida: q.sharp, luz: q.light, estable: q.steady, seguimiento: q.tracking, giroscopioCoincide: q.imuAgrees,
                lidar: q.lidar, distancia: q.distance, reflejos: q.glare, color: q.colorAgreement, fotograma: savedName))
        }

        if rw > 0, frames.count < Self.maxFrames {
            let resolution = camera.imageResolution
            let model = CameraModel(intrinsics: camera.intrinsics, cameraToWorld: camera.transform)
                .scaled(sx: Float(rw) / Float(resolution.width), sy: Float(rh) / Float(resolution.height))
            frames.append(ValidationFrame(camera: model, depth: reduced, width: rw, height: rh, sharp: q.sharp,
                                          light: q.light, steady: q.steady, tracking: q.tracking, imuAgrees: q.imuAgrees,
                                          colorDepth: colorMeters))
        }
        return q
    }

    func collected() -> [ValidationFrame] { queue.sync { frames } }

    // MARK: - Validación final

    /// Puntos de muestra en la superficie de una pared (rejilla 6×4, sin los bordes).
    static func samples(of wall: RoomWall, floorY: Float) -> (points: [SIMD3<Float>], normals: [SIMD3<Float>]) {
        var points: [SIMD3<Float>] = []
        for i in 0..<6 {
            for j in 0..<4 {
                let a = (Float(i) + 0.5) / 6 - 0.5, b = (Float(j) + 0.5) / 4 - 0.5
                points.append(wall.center + wall.tangent * (a * wall.width * 0.9) + SIMD3(0, b * wall.height * 0.9, 0))
            }
        }
        return (points, Array(repeating: wall.normal, count: points.count))
    }

    /// Puntos en las caras visibles de la caja de un mueble (tapa y cuatro lados).
    static func samples(of f: ParametricFurniture) -> (points: [SIMD3<Float>], normals: [SIMD3<Float>]) {
        let m = f.localToWorld
        let s = f.size
        var points: [SIMD3<Float>] = [], normals: [SIMD3<Float>] = []
        func add(_ local: SIMD3<Float>, _ n: SIMD3<Float>) {
            let p = m * SIMD4(local, 1), r = m * SIMD4(n, 0)
            points.append(SIMD3(p.x, p.y, p.z))
            normals.append(SIMD3(r.x, r.y, r.z))
        }
        for a in [-0.3, 0, 0.3] as [Float] {
            for b in [-0.3, 0, 0.3] as [Float] {
                add(SIMD3(a * s.x, s.y, b * s.z), SIMD3(0, 1, 0))
            }
            for h in [0.3, 0.7] as [Float] {
                add(SIMD3(a * s.x, h * s.y, s.z / 2), SIMD3(0, 0, 1))
                add(SIMD3(a * s.x, h * s.y, -s.z / 2), SIMD3(0, 0, -1))
                add(SIMD3(s.x / 2, h * s.y, a * s.z), SIMD3(1, 0, 0))
                add(SIMD3(-s.x / 2, h * s.y, a * s.z), SIMD3(-1, 0, 0))
            }
        }
        return (points, normals)
    }

    private static func depthAt(_ f: ValidationFrame, _ u: Float, _ v: Float) -> Float? {
        let x = Int(u), y = Int(v)
        guard x >= 0, y >= 0, x < f.width, y < f.height else { return nil }
        return f.depth[y * f.width + x]
    }

    /// `isWall`: la pared se compara punto a punto con el LiDAR (≤ 2 cm); el mueble, como caja que debe contener
    /// lo medido (nada medido por fuera de sus caras).
    static func validate(points: [SIMD3<Float>], normals: [SIMD3<Float>], isWall: Bool,
                         frames: [ValidationFrame], walls: [RoomWall] = []) -> ElementValidation {
        guard !points.isEmpty else { return ElementValidation() }
        let center = points.reduce(SIMD3<Float>(repeating: 0), +) / Float(points.count)
        var views = 0, goodViews = 0, closeViews = 0, frontalViews = 0, stableViews = 0
        var sectors = Set<Int>()
        var measured = 0, expected = 0
        var residuals: [Float] = []
        var outside = 0, compared = 0
        var colorCompared = 0, colorAgree = 0

        for f in frames {
            var visible = 0
            var frameResiduals: [Float] = []
            for (k, p) in points.enumerated() {
                // Solo caras que miran a la cámara.
                guard simd_dot(normals[k], f.camera.position - p) > 0,
                      let pr = f.camera.project(p),
                      pr.u >= 0, pr.v >= 0, pr.u < Float(f.width), pr.v < Float(f.height) else { continue }
                expected += 1
                guard let d = depthAt(f, pr.u, pr.v), d > 0 else { continue }
                if d < pr.z - 0.15 { continue }           // tapado por otra cosa
                visible += 1
                measured += 1
                frameResiduals.append(d - pr.z)
                if f.good, !f.colorDepth.isEmpty {
                    let c = f.colorDepth[Int(pr.v) * f.width + Int(pr.u)]
                    if c > 0 {
                        colorCompared += 1
                        if abs(c - d) / d < 0.05 { colorAgree += 1 }
                    }
                }
            }
            guard visible >= max(3, points.count / 4) else { continue }
            views += 1
            if f.tracking && f.imuAgrees { stableViews += 1 }
            guard f.good else { continue }
            goodViews += 1
            let toCamera = f.camera.position - center
            let distance = simd_length(toCamera)
            if distance >= 0.3 && distance <= 4 { closeViews += 1 }
            let direction = toCamera / max(distance, 1e-3)
            // Ángulo de la foto: se agrupan las direcciones de vista en sectores de 30° alrededor del elemento.
            sectors.insert(Int(((atan2(direction.x, direction.z) + .pi) / (.pi / 6)).rounded(.down)) % 12)
            let averageNormal = simd_normalize(normals.reduce(SIMD3<Float>(repeating: 0), +) + SIMD3(0, 1e-4, 0))
            if simd_dot(direction, averageNormal) > cos(Float.pi / 4) { frontalViews += 1 }
            for r in frameResiduals {
                residuals.append(r)
                compared += 1
                if r < -0.05 { outside += 1 }             // el LiDAR mide algo delante de la cara: la caja se queda corta
            }
        }

        let coverage = expected > 0 ? Float(measured) / Float(expected) : 0
        let rmse: Float? = residuals.isEmpty ? nil : {
            let deviations = residuals.map(abs).sorted()
            return deviations[deviations.count / 2] * 1.4826   // robusta: la mediana no se deja arrastrar por bordes
        }()
        let outsideFraction = compared > 0 ? Float(outside) / Float(compared) : 1
        let percent = { (v: Float) in "\(Int((v * 100).rounded())) %" }

        var checks: [ValidationCheck] = []
        checks.append(.init(name: "Vistas", passed: views >= 4,
                            detail: views >= 4 ? "\(views) vistas" : "solo \(views) vistas: grábalo más tiempo"))
        checks.append(.init(name: "Cámara", passed: goodViews >= 3,
                            detail: goodViews >= 3 ? "\(goodViews) fotos nítidas y bien expuestas"
                                : "\(goodViews) fotos nítidas: más luz y movimiento más lento"))
        let anglesOK = isWall ? (frontalViews >= 2 && sectors.count >= 2) : sectors.count >= 3
        checks.append(.init(name: "Ángulos", passed: anglesOK,
                            detail: anglesOK ? "\(sectors.count) direcciones distintas"
                                : isWall ? "grábala de frente y desde un lado" : "rodéalo: solo \(sectors.count) direcciones"))
        checks.append(.init(name: "Distancia", passed: closeViews >= 2,
                            detail: closeViews >= 2 ? "\(closeViews) vistas a 0,3–4 m" : "acércate a menos de 4 m"))
        checks.append(.init(name: "LiDAR", passed: coverage >= 0.6,
                            detail: coverage >= 0.6 ? "\(percent(coverage)) de lecturas fiables"
                                : "\(percent(coverage)) de lecturas fiables: superficie oscura, brillante o lejana"))
        if isWall {
            let ok = (rmse ?? 1) <= 0.02
            checks.append(.init(name: "Profundidad", passed: ok,
                                detail: rmse.map { "el LiDAR la mide a \(Int(($0 * 1000).rounded())) mm del modelo" } ?? "sin medir"))
        } else {
            let ok = compared > 0 && outsideFraction <= 0.1
            checks.append(.init(name: "Profundidad", passed: ok,
                                detail: compared == 0 ? "sin medir"
                                    : ok ? "lo medido cabe en sus medidas" : "\(percent(outsideFraction)) de lo medido sobresale: medidas cortas o mal colocado"))
        }
        if colorCompared >= 10 {
            let agreement = Float(colorAgree) / Float(colorCompared)
            let ok = agreement >= 0.7
            checks.append(.init(name: "Color frente a LiDAR", passed: ok,
                                detail: ok ? "la profundidad deducida del color coincide en un \(percent(agreement))"
                                    : "solo coincide en un \(percent(agreement)): posible cristal, reflejo o superficie difícil"))
        }
        let stableOK = views > 0 && Float(stableViews) / Float(views) >= 0.9
        checks.append(.init(name: "Posición y giroscopio", passed: stableOK,
                            detail: stableOK ? "seguimiento estable y coherente con el giroscopio"
                                : "el seguimiento se desvió: repasa la zona despacio"))
        var result = ElementValidation(checks: checks)
        if result.passed < checks.count {
            result.suggestedTarget = center
            result.suggestedEye = suggestedEye(center: center, normals: normals, isWall: isWall, seen: sectors, walls: walls)
        }
        return result
    }

    /// Como en la exploración guiada por anomalías: se propone la vista que más falta. Pared: de frente a 1,8 m.
    /// Mueble: el sector de 30° no visto más alejado de los vistos, a 1,5 m. Siempre a la altura de los ojos.
    /// Las vistas que caerían detrás de una pared no se proponen (un mueble pegado a la pared no se ve por detrás).
    static func suggestedEye(center: SIMD3<Float>, normals: [SIMD3<Float>], isWall: Bool, seen: Set<Int>,
                             walls: [RoomWall]) -> SIMD3<Float> {
        func place(_ direction: SIMD3<Float>, _ distance: Float) -> SIMD3<Float> {
            var eye = center + direction * distance
            eye.y = max(center.y + 0.3, 1.4)
            return eye
        }
        func inside(_ eye: SIMD3<Float>) -> Bool {
            !walls.contains { w in
                let offset = eye - w.center
                return abs(simd_dot(offset, w.tangent)) < w.width / 2 && simd_dot(offset, w.normal) < 0.3
                    && simd_dot(center - w.center, w.normal) > 0          // el elemento está delante de esa pared
            }
        }
        if isWall {
            let n = normals.first ?? SIMD3(0, 0, 1)
            return place(simd_normalize(SIMD3(n.x, 0, n.z) + SIMD3(1e-4, 0, 0)), 1.8)
        }
        func gap(_ s: Int) -> Int { seen.map { min(abs($0 - s), 12 - abs($0 - s)) }.min() ?? 6 }
        let candidates = (0..<12).filter { !seen.contains($0) }.sorted { gap($0) > gap($1) } + Array(0..<12)
        for distance in [1.5, 1.0] as [Float] {
            for sector in candidates {
                let azimuth = (Float(sector) + 0.5) * .pi / 6 - .pi
                let eye = place(SIMD3(sin(azimuth), 0, cos(azimuth)), distance)
                if inside(eye) { return eye }
            }
        }
        return place(SIMD3(0, 0, 1), 1.0)
    }

    static func validate(room: RoomModel, frames: [ValidationFrame]) -> RoomModel {
        var room = room
        for i in room.walls.indices {
            let s = samples(of: room.walls[i], floorY: room.floorY)
            room.walls[i].validation = validate(points: s.points, normals: s.normals, isWall: true, frames: frames)
        }
        for i in room.furniture.indices {
            let s = samples(of: room.furniture[i])
            room.furniture[i].validation = validate(points: s.points, normals: s.normals, isWall: false, frames: frames,
                                                        walls: room.walls)
        }
        return room
    }
}
