import ARKit
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
                                      attitude: attitude, tracking: tracking)
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
                        rate: Float, attitude: simd_quatf?, tracking: Bool) -> LiveQuality {
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
                         frames: [ValidationFrame]) -> ElementValidation {
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
        return ElementValidation(checks: checks)
    }

    static func validate(room: RoomModel, frames: [ValidationFrame]) -> RoomModel {
        var room = room
        for i in room.walls.indices {
            let s = samples(of: room.walls[i], floorY: room.floorY)
            room.walls[i].validation = validate(points: s.points, normals: s.normals, isWall: true, frames: frames)
        }
        for i in room.furniture.indices {
            let s = samples(of: room.furniture[i])
            room.furniture[i].validation = validate(points: s.points, normals: s.normals, isWall: false, frames: frames)
        }
        return room
    }
}
