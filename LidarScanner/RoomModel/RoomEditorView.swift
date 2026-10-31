import Observation
import SceneKit
import SwiftUI
import UIKit
import simd

/// Estado del editor. Los gestos llegan del `SCNView`; `revision` avisa de que hay que reconstruir los nodos.
@MainActor
@Observable
final class RoomEditorModel {
    var room: RoomModel
    var selectedID: UUID?
    var isMeasuring = false
    var measureA: SIMD3<Float>?
    var measureB: SIMD3<Float>?
    var isVerifying = false
    var exportedFiles: [URL] = []
    var exportError: String?
    private(set) var revision = 0

    @ObservationIgnored let keyframes: [Keyframe]

    init(room: RoomModel, keyframes: [Keyframe]) {
        self.room = room
        self.keyframes = keyframes
    }

    var selected: ParametricFurniture? { room.furniture.first { $0.id == selectedID } }
    var measuredDistance: Float? {
        guard let measureA, let measureB else { return nil }
        return simd_distance(measureA, measureB)
    }

    private func edit(_ change: (inout ParametricFurniture) -> Void) {
        guard let i = room.furniture.firstIndex(where: { $0.id == selectedID }) else { return }
        var f = room.furniture[i]
        change(&f)
        // Una edición a mano invalida lo verificado: vuelve a amarillo hasta repetir las pruebas.
        f.state = .yellow
        f.report = VerificationReport()
        room.furniture[i] = f
        revision += 1
    }

    func move(to position: SIMD3<Float>) {
        edit { f in
            f.position = SIMD3(position.x, room.floorY, position.z)
            f = room.keepInside(f)
        }
    }

    func rotate(by angle: Float) {
        edit { f in
            f.yaw += angle
            f = room.keepInside(f)
        }
    }

    /// Al soltar: si quedó junto a una pared, se alinea y se pega a ella.
    func snapSelected() {
        edit { f in f = room.snappedToWall(f) }
    }

    func deleteSelected() {
        room.furniture.removeAll { $0.id == selectedID }
        selectedID = nil
        revision += 1
    }

    func addMeasurePoint(_ point: SIMD3<Float>) {
        if measureA == nil || measureB != nil {
            measureA = point
            measureB = nil
        } else {
            measureB = point
        }
        revision += 1
    }

    func clearMeasure() {
        measureA = nil
        measureB = nil
        revision += 1
    }

    /// El verificador es pesado: corre fuera del hilo principal y aplica el resultado al terminar.
    func verifyAll() async {
        guard !isVerifying else { return }
        isVerifying = true
        defer { isVerifying = false }
        let furniture = room.furniture
        let keyframes = keyframes
        let room = self.room
        let results = await Task.detached(priority: .userInitiated) { () -> [ParametricFurniture] in
            let masks = keyframes.map { InstanceSignatures.instanceMask(for: $0.image) }
            let metal = MetalFusion()
            return furniture.map { room.keepInside(FourTestVerifier.verify($0, keyframes: keyframes, masks: masks, metal: metal)) }
        }.value
        self.room.furniture = results
        revision += 1
    }

    func export() {
        do {
            let folder = try ScanStorage.newScanFolder(prefix: "Habitacion")
            let json = folder.appendingPathComponent("habitacion.json")
            try ModelExporter.writeJSON(room, to: json)
            let parts = RoomMesh.parts(of: room)
            let obj = folder.appendingPathComponent("habitacion.obj")
            try ModelExporter.writeOBJ(parts: parts, to: obj)
            let usdz = folder.appendingPathComponent("habitacion.usdz")
            try ModelExporter.writeUSDZ(parts: parts, to: usdz)
            exportedFiles = [json, obj, usdz]
            exportError = nil
        } catch {
            exportError = error.localizedDescription
        }
    }
}

/// Mallas de cajas del modelo, en metros y coordenadas del mundo, para los exportadores de la app.
enum RoomMesh {
    private static let faces: [(normal: SIMD3<Float>, u: SIMD3<Float>, v: SIMD3<Float>)] = [
        (SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1)), (SIMD3(-1, 0, 0), SIMD3(0, 0, 1), SIMD3(0, 1, 0)),
        (SIMD3(0, 1, 0), SIMD3(0, 0, 1), SIMD3(1, 0, 0)), (SIMD3(0, -1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, 1)),
        (SIMD3(0, 0, 1), SIMD3(1, 0, 0), SIMD3(0, 1, 0)), (SIMD3(0, 0, -1), SIMD3(0, 1, 0), SIMD3(1, 0, 0)),
    ]

    private static func append(center: SIMD3<Float>, size: SIMD3<Float>, transform: simd_float4x4,
                               color: SIMD3<UInt8>, to mesh: inout ColoredMesh) {
        let half = size / 2
        let rotation = simd_float3x3(SIMD3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
                                     SIMD3(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
                                     SIMD3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z))
        for face in faces {
            let base = UInt32(mesh.positions.count)
            for (a, b) in [(Float(-1), Float(-1)), (1, -1), (1, 1), (-1, 1)] {
                let local = center + (face.normal + face.u * a + face.v * b) * half
                let world = transform * SIMD4<Float>(local, 1)
                mesh.positions.append(SIMD3(world.x, world.y, world.z))
                mesh.normals.append(rotation * face.normal)
                mesh.colors.append(color)
            }
            mesh.indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
        }
    }

    static func parts(of room: RoomModel) -> [ModelExporter.Part] {
        var result: [ModelExporter.Part] = []
        var walls = ColoredMesh()
        for wall in room.walls {
            let m = simd_float4x4(columns: (SIMD4<Float>(wall.tangent, 0), SIMD4<Float>(0, 1, 0, 0),
                                            SIMD4<Float>(wall.normal, 0), SIMD4<Float>(wall.center, 1)))
            append(center: SIMD3(0, 0, -0.05), size: SIMD3(wall.width, wall.height, 0.1), transform: m,
                   color: SIMD3(225, 225, 220), to: &walls)
        }
        if !walls.isEmpty { result.append(.init(name: "Paredes", mesh: walls)) }
        for furniture in room.furniture {
            var mesh = ColoredMesh()
            for part in furniture.parts {
                append(center: part.center, size: part.size, transform: furniture.localToWorld,
                       color: SIMD3(176, 142, 104), to: &mesh)
            }
            result.append(.init(name: furniture.name, mesh: mesh))
        }
        return result
    }
}

/// Vista 3D táctil: tocar selecciona o mide, arrastrar mueve sobre el suelo, dos dedos giran el mueble seleccionado.
@MainActor
private struct RoomSceneView: UIViewRepresentable {
    let model: RoomEditorModel
    /// Se pasa como valor para que cada edición vuelva a llamar a `updateUIView`.
    let revision: Int

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = context.coordinator.scene
        view.backgroundColor = .black
        view.autoenablesDefaultLighting = true
        view.allowsCameraControl = true
        context.coordinator.install(on: view)
        context.coordinator.rebuild()
        context.coordinator.placeCamera()
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        context.coordinator.rebuildIfNeeded()
    }

    @MainActor
    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        let scene = SCNScene()
        private let model: RoomEditorModel
        private let content = SCNNode()
        private weak var view: SCNView?
        private var builtRevision = -1
        private var grabOffset = SIMD3<Float>(repeating: 0)

        init(model: RoomEditorModel) {
            self.model = model
            super.init()
            scene.rootNode.addChildNode(content)
        }

        func install(on view: SCNView) {
            self.view = view
            let tap = UITapGestureRecognizer(target: self, action: #selector(tapped))
            let pan = UIPanGestureRecognizer(target: self, action: #selector(panned))
            pan.maximumNumberOfTouches = 1
            pan.delegate = self
            let rotation = UIRotationGestureRecognizer(target: self, action: #selector(rotated))
            rotation.delegate = self
            for recognizer in view.gestureRecognizers ?? [] where recognizer is UIPanGestureRecognizer {
                recognizer.require(toFail: pan)    // arrastrar un mueble no debe mover también la cámara
            }
            for recognizer in view.gestureRecognizers ?? [] where recognizer is UIRotationGestureRecognizer {
                recognizer.require(toFail: rotation)
            }
            view.addGestureRecognizer(tap)
            view.addGestureRecognizer(pan)
            view.addGestureRecognizer(rotation)
        }

        func gestureRecognizerShouldBegin(_ recognizer: UIGestureRecognizer) -> Bool {
            guard let view, model.selectedID != nil, !model.isMeasuring else { return false }
            if recognizer is UIRotationGestureRecognizer { return true }
            return furnitureID(at: recognizer.location(in: view)) == model.selectedID
        }

        private func furnitureID(at point: CGPoint) -> UUID? {
            guard let hit = view?.hitTest(point, options: nil).first else { return nil }
            var node: SCNNode? = hit.node
            while let current = node {
                if let name = current.name, let id = UUID(uuidString: name) { return id }
                node = current.parent
            }
            return nil
        }

        @objc private func tapped(_ recognizer: UITapGestureRecognizer) {
            guard let view else { return }
            let point = recognizer.location(in: view)
            if model.isMeasuring {
                guard let hit = view.hitTest(point, options: nil).first else { return }
                model.addMeasurePoint(simd_float3(hit.worldCoordinates))
            } else {
                model.selectedID = furnitureID(at: point)
                rebuild()
            }
        }

        /// Punto del suelo bajo el dedo.
        private func floorPoint(_ point: CGPoint) -> SIMD3<Float>? {
            guard let view else { return nil }
            let near = simd_float3(view.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), 0)))
            let far = simd_float3(view.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), 1)))
            let direction = far - near
            guard abs(direction.y) > 1e-5 else { return nil }
            let t = (model.room.floorY - near.y) / direction.y
            return t > 0 ? near + direction * t : nil
        }

        @objc private func panned(_ recognizer: UIPanGestureRecognizer) {
            guard let view, let selected = model.selected,
                  let point = floorPoint(recognizer.location(in: view)) else { return }
            switch recognizer.state {
            case .began:
                grabOffset = selected.position - point
            case .changed:
                model.move(to: point + grabOffset)
            case .ended:
                model.snapSelected()
            default:
                break
            }
        }

        @objc private func rotated(_ recognizer: UIRotationGestureRecognizer) {
            switch recognizer.state {
            case .changed:
                model.rotate(by: -Float(recognizer.rotation))
                recognizer.rotation = 0
            case .ended:
                model.snapSelected()
            default:
                break
            }
        }

        func rebuildIfNeeded() {
            if builtRevision != model.revision { rebuild() }
        }

        func rebuild() {
            builtRevision = model.revision
            content.childNodes.forEach { $0.removeFromParentNode() }

            for wall in model.room.walls {
                let box = SCNBox(width: CGFloat(wall.width), height: CGFloat(wall.height), length: 0.04, chamferRadius: 0)
                box.firstMaterial?.diffuse.contents = UIColor(white: 0.85, alpha: 0.35)
                box.firstMaterial?.isDoubleSided = true
                let node = SCNNode(geometry: box)
                node.simdPosition = wall.center - wall.normal * 0.02
                node.simdEulerAngles = SIMD3(0, atan2(wall.normal.x, wall.normal.z), 0)
                content.addChildNode(node)
            }

            for furniture in model.room.furniture {
                let group = SCNNode()
                group.name = furniture.id.uuidString
                group.simdTransform = furniture.localToWorld
                let color = Self.color(for: furniture.state)
                for part in furniture.parts {
                    let box = SCNBox(width: CGFloat(part.size.x), height: CGFloat(part.size.y), length: CGFloat(part.size.z),
                                     chamferRadius: 0.003)
                    box.firstMaterial?.diffuse.contents = color
                    box.firstMaterial?.emission.contents = furniture.id == model.selectedID ? UIColor(white: 0.25, alpha: 1) : UIColor.black
                    let node = SCNNode(geometry: box)
                    node.simdPosition = part.center
                    group.addChildNode(node)
                }
                content.addChildNode(group)
            }

            if let a = model.measureA {
                content.addChildNode(Self.marker(at: a))
                if let b = model.measureB {
                    content.addChildNode(Self.marker(at: b))
                    let source = SCNGeometrySource(vertices: [SCNVector3(a), SCNVector3(b)])
                    let element = SCNGeometryElement(indices: [UInt32(0), 1], primitiveType: .line)
                    let line = SCNGeometry(sources: [source], elements: [element])
                    line.firstMaterial?.diffuse.contents = UIColor.cyan
                    line.firstMaterial?.lightingModel = .constant
                    content.addChildNode(SCNNode(geometry: line))
                }
            }
        }

        private static func marker(at point: SIMD3<Float>) -> SCNNode {
            let sphere = SCNSphere(radius: 0.015)
            sphere.firstMaterial?.diffuse.contents = UIColor.cyan
            sphere.firstMaterial?.lightingModel = .constant
            let node = SCNNode(geometry: sphere)
            node.simdPosition = point
            return node
        }

        private static func color(for state: VerificationState) -> UIColor {
            switch state {
            case .red: UIColor.systemRed
            case .yellow: UIColor.systemYellow
            case .green: UIColor.systemGreen
            }
        }

        func placeCamera() {
            let positions = model.room.walls.map(\.center) + model.room.furniture.map(\.position)
            guard !positions.isEmpty else { return }
            let center = positions.reduce(SIMD3<Float>(repeating: 0), +) / Float(positions.count)
            let radius = max(positions.map { simd_distance($0, center) }.max() ?? 2, 1.5)
            let camera = SCNNode()
            camera.camera = SCNCamera()
            camera.camera?.zNear = 0.02
            camera.camera?.zFar = Double(radius * 12)
            camera.simdPosition = center + SIMD3(radius * 0.8, radius * 1.6, radius * 0.8)
            camera.look(at: SCNVector3(center))
            scene.rootNode.addChildNode(camera)
            view?.pointOfView = camera
        }
    }
}

/// Editor de la habitación: estado Rojo/Amarillo/Verde de cada mueble y mover, girar, borrar, medir, verificar y exportar.
@MainActor
struct RoomEditorView: View {
    @Bindable var model: RoomEditorModel

    private func percent(_ value: Float?) -> String {
        value.map { $0.formatted(.number.precision(.fractionLength(2))) } ?? "sin medir"
    }

    var body: some View {
        VStack(spacing: 0) {
            RoomSceneView(model: model, revision: model.revision)
                .overlay(alignment: .top) { measureBanner }

            VStack(alignment: .leading, spacing: 10) {
                if let f = model.selected { details(of: f) } else { summary }
                HStack {
                    Toggle(isOn: $model.isMeasuring) { Label("Medir", systemImage: "ruler") }
                        .toggleStyle(.button)
                        .onChange(of: model.isMeasuring) { _, measuring in if !measuring { model.clearMeasure() } }
                    Spacer()
                    Button { Task { await model.verifyAll() } } label: {
                        Label(model.isVerifying ? "Verificando…" : "Verificar", systemImage: "checkmark.seal")
                    }
                    .disabled(model.isVerifying || model.keyframes.isEmpty)
                    Button { model.export() } label: {
                        Label("Exportar", systemImage: "square.and.arrow.up")
                    }
                }
                .buttonStyle(.bordered)
                if !model.exportedFiles.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack {
                            ForEach(model.exportedFiles, id: \.self) { url in
                                ShareLink(item: url) { Label(url.pathExtension.uppercased(), systemImage: "doc") }
                            }
                        }
                    }
                }
                if let error = model.exportError {
                    Text(error).font(.caption).foregroundStyle(.orange)
                }
            }
            .padding()
            .background(Color(.secondarySystemBackground))
        }
        .task { if !model.keyframes.isEmpty { await model.verifyAll() } }
    }

    @ViewBuilder private var measureBanner: some View {
        if model.isMeasuring {
            Text(measureText)
                .font(.headline.monospacedDigit())
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(.top, 8)
        }
    }

    private var measureText: String {
        if let distance = model.measuredDistance {
            return distance.formatted(.number.precision(.fractionLength(3))) + " m"
        }
        return model.measureA == nil ? "Toca el primer punto" : "Toca el segundo punto"
    }

    private func reportText(_ f: ParametricFurniture) -> String {
        let rmse = f.report.geometryRMSE.map { "\(Int($0 * 1000)) mm" } ?? "sin medir"
        let free = f.report.freeSpaceViolations.map { String($0) } ?? "sin medir"
        return "Vistas \(f.report.views) · IoU \(percent(f.report.silhouetteIoU)) · RMSE \(rmse) · Espacio libre \(free)"
    }

    private var summary: some View {
        let count = { (state: VerificationState) in model.room.furniture.filter { $0.state == state }.count }
        return VStack(alignment: .leading, spacing: 4) {
            Text("\(model.room.furniture.count) muebles · \(model.room.walls.count) paredes").font(.headline)
            Text("Verdes \(count(.green)) · Amarillos \(count(.yellow)) · Rojos \(count(.red)). Toca un mueble para editarlo.")
                .font(.caption).foregroundStyle(.secondary)
            if model.keyframes.isEmpty {
                Text("Sin fotos de esta captura: no se puede verificar.").font(.caption).foregroundStyle(.orange)
            }
        }
    }

    private func details(of f: ParametricFurniture) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(f.name).font(.headline)
                Text(f.state.title).font(.caption.bold())
                    .foregroundStyle(f.state == .green ? .green : f.state == .yellow ? .yellow : .red)
                Spacer()
                Button { model.rotate(by: -.pi / 12); model.snapSelected() } label: { Image(systemName: "rotate.left") }
                Button { model.rotate(by: .pi / 12); model.snapSelected() } label: { Image(systemName: "rotate.right") }
                Button(role: .destructive) { model.deleteSelected() } label: { Image(systemName: "trash") }
            }
            .buttonStyle(.bordered)
            Text("\(f.parts.count) piezas · \(f.size.x * 100, specifier: "%.0f") × \(f.size.y * 100, specifier: "%.0f") × \(f.size.z * 100, specifier: "%.0f") cm")
                .font(.caption.monospacedDigit())
            Text(reportText(f))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }
}
