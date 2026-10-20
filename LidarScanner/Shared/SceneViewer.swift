import SceneKit
import SwiftUI
import Observation

/// Cambio hecho a un objeto en el editor (respecto a su posición escaneada).
struct ObjectEdit: Codable, Equatable {
    var dx: Float = 0
    var dy: Float = 0
    var dz: Float = 0
    /// Giro alrededor del eje vertical, en radianes.
    var yaw: Float = 0
    var borrado = false

    var offset: SIMD3<Float> { SIMD3(dx, dy, dz) }
}

/// Visor y editor 3D de un escaneo de entorno, con su textura o color real.
/// Cada objeto detectado es un nodo independiente: se puede seleccionar, aislar, ocultar, mover, girar,
/// borrar y exportar. Las ediciones se guardan en `edicion.json` y se pueden deshacer.
@MainActor
@Observable
final class SceneViewerModel {
    struct Item: Identifiable {
        let id: String          // nombre del nodo
        let title: String
        let subtitle: String
        let isStructure: Bool
        let file: URL
    }

    private struct EditFile: Codable {
        var objetos: [String: ObjectEdit]
    }

    var items: [Item] = []
    var hidden: Set<String> = []
    var edits: [String: ObjectEdit] = [:]
    var selectedID: String? {
        didSet { if selectedID != oldValue { isMoving = false } }
    }
    var isMoving = false
    var isLoading = true
    var errorMessage: String?
    var shaded = false { didSet { applyAppearance() } }
    var cameraNode: SCNNode?
    var target = SCNVector3Zero
    private(set) var undoStack: [(id: String, edit: ObjectEdit)] = []

    let scene = SCNScene()
    @ObservationIgnored private var nodes: [String: SCNNode] = [:]
    @ObservationIgnored private var meshes: [String: ColoredMesh] = [:]
    @ObservationIgnored private var centers: [String: SIMD3<Float>] = [:]
    @ObservationIgnored private var folder: URL?
    @ObservationIgnored private var pendingEdit: (id: String, edit: ObjectEdit)?

    var selectedItem: Item? { items.first { $0.id == selectedID } }
    var canUndo: Bool { !undoStack.isEmpty }
    var deletedItems: [Item] { items.filter { edits[$0.id]?.borrado == true } }
    var activeItems: [Item] { items.filter { edits[$0.id]?.borrado != true } }

    private var editsURL: URL? { folder?.appendingPathComponent("edicion.json") }

    // MARK: - Carga

    func load(folder: URL) async {
        guard nodes.isEmpty else { return }
        self.folder = folder

        // Objetos de escena.json; si el escaneo no se separó, la malla completa como un único objeto.
        var sources: [(Item, URL)] = []
        for object in ScanStorage.sceneObjects(in: folder) {
            let url = folder.appendingPathComponent(object.archivo)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            let item = Item(id: ModelExporter.objectName(object.nombre), title: object.nombre,
                            subtitle: object.detailText, isStructure: object.estructura, file: url)
            sources.append((item, url))
        }
        if sources.isEmpty {
            let url = folder.appendingPathComponent("malla_color.ply")
            sources = [(Item(id: "Escaneo", title: "Escaneo completo", subtitle: "Malla con color",
                             isStructure: false, file: url), url)]
        }

        let loaded: [(Item, ColoredMesh)] = await Task.detached(priority: .userInitiated) {
            sources.compactMap { item, url in
                guard let mesh = try? PLYReader.readMesh(from: url), !mesh.isEmpty else { return nil }
                return (item, mesh)
            }
        }.value

        guard !loaded.isEmpty else {
            errorMessage = "No se encontró una malla con color en este escaneo."
            isLoading = false
            return
        }

        // Textura fotográfica compartida por todos los objetos (si el escaneo la tiene).
        let texture = UIImage(contentsOfFile: folder.appendingPathComponent("textura.jpg").path)

        if let url = editsURL, let data = try? Data(contentsOf: url),
           let file = try? JSONDecoder().decode(EditFile.self, from: data) {
            edits = file.objetos
        }

        var minP = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxP = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for (item, mesh) in loaded {
            var objectMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
            var objectMax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
            for p in mesh.positions {
                objectMin = simd_min(objectMin, p)
                objectMax = simd_max(objectMax, p)
            }
            minP = simd_min(minP, objectMin)
            maxP = simd_max(maxP, objectMax)
            let center = (objectMin + objectMax) / 2

            // El pivote en el centro del objeto: así gira sobre sí mismo.
            let node = SCNNode(geometry: ModelExporter.makeGeometry(mesh, texture: texture))
            node.name = item.id
            node.pivot = SCNMatrix4MakeTranslation(center.x, center.y, center.z)
            scene.rootNode.addChildNode(node)
            nodes[item.id] = node
            meshes[item.id] = mesh
            centers[item.id] = center
        }
        items = loaded.map(\.0)

        // El techo se oculta al abrir para ver el interior desde arriba.
        hidden = Set(items.filter { $0.title == SurfaceLabel.ceiling.displayName }.map(\.id))

        // Cámara en diagonal por encima de la escena, apuntando a su centro.
        let center = (minP + maxP) / 2
        let radius = max(simd_length(maxP - minP) / 2, 0.5)
        let camera = SCNNode()
        camera.camera = SCNCamera()
        camera.camera?.zNear = 0.01
        camera.camera?.zFar = Double(radius * 20)
        camera.position = SCNVector3(center.x + radius * 1.2, center.y + radius * 1.4, center.z + radius * 1.2)
        camera.look(at: SCNVector3(center.x, center.y, center.z))
        scene.rootNode.addChildNode(camera)
        target = SCNVector3(center.x, center.y, center.z)
        cameraNode = camera

        applyAppearance()
        isLoading = false
    }

    // MARK: - Vista

    func select(_ id: String?) {
        selectedID = (id == selectedID) ? nil : id
        applyAppearance()
    }

    func toggleHidden(_ id: String) {
        if hidden.contains(id) { hidden.remove(id) } else { hidden.insert(id) }
        if hidden.contains(id), selectedID == id { selectedID = nil }
        applyAppearance()
    }

    func isolate(_ id: String) {
        hidden = Set(items.map(\.id)).subtracting([id])
        selectedID = id
        applyAppearance()
    }

    func showAll() {
        hidden = []
        selectedID = nil
        applyAppearance()
    }

    // MARK: - Edición

    func worldCenter(of id: String) -> SCNVector3? { nodes[id]?.position }

    func beginEdit(_ id: String) {
        pendingEdit = (id, edits[id] ?? ObjectEdit())
    }

    func translate(_ id: String, by delta: SIMD3<Float>) {
        var edit = edits[id] ?? ObjectEdit()
        edit.dx += delta.x; edit.dy += delta.y; edit.dz += delta.z
        edits[id] = edit
        applyTransform(id)
    }

    func rotate(_ id: String, by angle: Float) {
        var edit = edits[id] ?? ObjectEdit()
        edit.yaw += angle
        edits[id] = edit
        applyTransform(id)
    }

    /// Sube o baja el objeto (por ejemplo, para apoyarlo en otra superficie).
    func lift(_ id: String, by meters: Float) {
        beginEdit(id)
        translate(id, by: SIMD3(0, meters, 0))
        commitEdit()
    }

    func commitEdit() {
        if let pending = pendingEdit, edits[pending.id] ?? ObjectEdit() != pending.edit {
            undoStack.append(pending)
            save()
        }
        pendingEdit = nil
    }

    func delete(_ id: String) {
        let previous = edits[id] ?? ObjectEdit()
        undoStack.append((id, previous))
        var edit = previous
        edit.borrado = true
        edits[id] = edit
        selectedID = nil
        applyAppearance()
        save()
    }

    func restore(_ id: String) {
        let previous = edits[id] ?? ObjectEdit()
        undoStack.append((id, previous))
        var edit = previous
        edit.borrado = false
        edits[id] = edit
        applyAppearance()
        save()
    }

    func undo() {
        guard let last = undoStack.popLast() else { return }
        edits[last.id] = last.edit
        applyAppearance()
        save()
    }

    /// Vuelve a la escena tal como se escaneó.
    func resetEdits() {
        edits = [:]
        undoStack = []
        applyAppearance()
        save()
    }

    /// Exporta la escena editada (objetos movidos/girados, sin los borrados ni los ocultos) a OBJ y USDZ.
    func exportEdited() -> [URL] {
        guard let folder else { return [] }
        let parts = items.filter { !hidden.contains($0.id) && edits[$0.id]?.borrado != true }.compactMap { item in
            meshes[item.id].map { ModelExporter.Part(name: item.title, mesh: transformed($0, id: item.id)) }
        }
        guard !parts.isEmpty else { return [] }
        var files: [URL] = []
        let material = FileManager.default.fileExists(atPath: folder.appendingPathComponent("modelo.mtl").path) ? "modelo.mtl" : nil
        let obj = folder.appendingPathComponent("escena_editada.obj")
        if (try? ModelExporter.writeOBJ(parts: parts, to: obj, materialFile: material)) != nil { files.append(obj) }
        let texture = UIImage(contentsOfFile: folder.appendingPathComponent("textura.jpg").path)
        let usdz = folder.appendingPathComponent("escena_editada.usdz")
        if (try? ModelExporter.writeUSDZ(parts: parts, texture: texture, to: usdz)) != nil { files.append(usdz) }
        return files
    }

    private func transformed(_ mesh: ColoredMesh, id: String) -> ColoredMesh {
        guard let edit = edits[id], edit != ObjectEdit(), let center = centers[id] else { return mesh }
        let c = cos(edit.yaw), s = sin(edit.yaw)
        func rotate(_ v: SIMD3<Float>) -> SIMD3<Float> {
            SIMD3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z)
        }
        var out = mesh
        out.positions = mesh.positions.map { rotate($0 - center) + center + edit.offset }
        out.normals = mesh.normals.map(rotate)
        return out
    }

    private func save() {
        guard let url = editsURL else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(EditFile(objetos: edits)).write(to: url, options: .atomic)
    }

    // MARK: - Aspecto

    private func applyTransform(_ id: String) {
        guard let node = nodes[id], let center = centers[id] else { return }
        let edit = edits[id] ?? ObjectEdit()
        let position = center + edit.offset
        node.position = SCNVector3(position.x, position.y, position.z)
        node.eulerAngles = SCNVector3(0, edit.yaw, 0)
    }

    private func applyAppearance() {
        for (id, node) in nodes {
            applyTransform(id)
            node.isHidden = hidden.contains(id) || edits[id]?.borrado == true
            node.opacity = (selectedID == nil || selectedID == id) ? 1 : 0.25
            node.geometry?.firstMaterial?.lightingModel = shaded ? .lambert : .constant
        }
    }
}

struct SceneViewerView: View {
    let folder: URL
    @State private var model = SceneViewerModel()
    @State private var showList = false
    @State private var exportedFiles: [URL] = []
    @State private var showExport = false
    @State private var confirmReset = false

    var body: some View {
        ZStack(alignment: .bottom) {
            SceneKitContainer(model: model)
                .ignoresSafeArea()

            if model.isLoading {
                ProgressView("Cargando modelo 3D…")
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
                    .frame(maxHeight: .infinity)
            } else if let error = model.errorMessage {
                ContentUnavailableView("Sin modelo", systemImage: "cube.transparent", description: Text(error))
            }

            if let item = model.selectedItem {
                selectionCard(item)
            } else if !model.isLoading {
                Text("Toca un objeto para seleccionarlo · arrastra para girar la vista · pellizca para acercar")
                    .font(.caption)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.bottom, 8)
            }
        }
        .navigationTitle("Visor y editor 3D")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button { model.undo() } label: { Image(systemName: "arrow.uturn.backward") }
                    .disabled(!model.canUndo)
                Button { showList = true } label: { Image(systemName: "list.bullet") }
                    .disabled(model.items.count < 2)
                Menu {
                    Toggle("Sombreado (ver relieve)", isOn: $model.shaded)
                    Button("Mostrar todo", systemImage: "eye") { model.showAll() }
                    Button("Exportar escena editada", systemImage: "square.and.arrow.up") {
                        exportedFiles = model.exportEdited()
                        showExport = !exportedFiles.isEmpty
                    }
                    Button("Restaurar escena original", systemImage: "arrow.counterclockwise", role: .destructive) {
                        confirmReset = true
                    }
                } label: { Image(systemName: "ellipsis.circle") }
            }
        }
        .sheet(isPresented: $showList) { objectList }
        .sheet(isPresented: $showExport) {
            NavigationStack {
                ExportedFilesView(title: "Escena editada", files: exportedFiles)
            }
            .presentationDetents([.medium])
        }
        .confirmationDialog("¿Deshacer todos los cambios (movimientos, giros y borrados)?",
                            isPresented: $confirmReset, titleVisibility: .visible) {
            Button("Restaurar", role: .destructive) { model.resetEdits() }
        }
        .task { await model.load(folder: folder) }
    }

    private func selectionCard(_ item: SceneViewerModel.Item) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(item.title).font(.headline)
                    Text(item.subtitle).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                Spacer()
                Button { model.select(nil) } label: { Image(systemName: "xmark.circle.fill").font(.title2) }
                    .foregroundStyle(.secondary)
            }

            if model.isMoving {
                Text("Arrastra con un dedo para moverlo · gira con dos dedos")
                    .font(.caption)
                    .foregroundStyle(.cyan)
                HStack(spacing: 10) {
                    Button("Bajar", systemImage: "arrow.down") { model.lift(item.id, by: -0.02) }
                    Button("Subir", systemImage: "arrow.up") { model.lift(item.id, by: 0.02) }
                    Spacer()
                    Button("Listo") { model.isMoving = false }
                        .buttonStyle(.borderedProminent)
                        .tint(.cyan)
                }
                .buttonStyle(.bordered)
                .font(.subheadline)
            } else {
                HStack(spacing: 8) {
                    if !item.isStructure {
                        Button("Mover", systemImage: "move.3d") { model.isMoving = true }
                    }
                    Button("Aislar", systemImage: "scope") { model.isolate(item.id) }
                    Button("Borrar", systemImage: "trash", role: .destructive) { model.delete(item.id) }
                    ShareLink(item: item.file) { Image(systemName: "square.and.arrow.up") }
                }
                .buttonStyle(.bordered)
                .font(.subheadline)
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .padding()
    }

    private var objectList: some View {
        NavigationStack {
            List {
                Section("En la escena") {
                    ForEach(model.activeItems) { item in
                        HStack {
                            Button {
                                model.select(item.id)
                                showList = false
                            } label: {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(item.title).foregroundStyle(.primary)
                                    Text(item.subtitle).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                                }
                            }
                            Spacer()
                            Button { model.toggleHidden(item.id) } label: {
                                Image(systemName: model.hidden.contains(item.id) ? "eye.slash" : "eye")
                            }
                            .buttonStyle(.borderless)
                        }
                        .swipeActions {
                            Button("Borrar", systemImage: "trash", role: .destructive) { model.delete(item.id) }
                        }
                    }
                }
                if !model.deletedItems.isEmpty {
                    Section("Borrados") {
                        ForEach(model.deletedItems) { item in
                            HStack {
                                Text(item.title).strikethrough().foregroundStyle(.secondary)
                                Spacer()
                                Button("Recuperar", systemImage: "arrow.uturn.backward") { model.restore(item.id) }
                                    .buttonStyle(.borderless)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Objetos")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Listo") { showList = false } }
            }
        }
        .presentationDetents([.medium, .large])
    }
}

/// `SCNView` con cámara orbital, selección por toque y gestos de edición (mover y girar el objeto).
private struct SceneKitContainer: UIViewRepresentable {
    let model: SceneViewerModel

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = model.scene
        view.backgroundColor = .black
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.antialiasingMode = .multisampling4X
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.defaultCameraController.inertiaEnabled = true

        let coordinator = context.coordinator
        view.addGestureRecognizer(UITapGestureRecognizer(target: coordinator, action: #selector(Coordinator.tapped(_:))))
        let pan = UIPanGestureRecognizer(target: coordinator, action: #selector(Coordinator.panned(_:)))
        pan.maximumNumberOfTouches = 1
        pan.isEnabled = false
        view.addGestureRecognizer(pan)
        let rotation = UIRotationGestureRecognizer(target: coordinator, action: #selector(Coordinator.rotated(_:)))
        rotation.isEnabled = false
        view.addGestureRecognizer(rotation)
        coordinator.pan = pan
        coordinator.rotation = rotation
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        if let camera = model.cameraNode, view.pointOfView !== camera {
            view.pointOfView = camera
            view.defaultCameraController.target = model.target
        }
        // Mientras se edita un objeto, los dedos lo mueven a él y no a la cámara.
        let editing = model.isMoving && model.selectedID != nil
        view.allowsCameraControl = !editing
        context.coordinator.pan?.isEnabled = editing
        context.coordinator.rotation?.isEnabled = editing
    }

    @MainActor
    final class Coordinator: NSObject {
        let model: SceneViewerModel
        weak var pan: UIPanGestureRecognizer?
        weak var rotation: UIRotationGestureRecognizer?
        private var lastPoint: CGPoint = .zero

        init(model: SceneViewerModel) { self.model = model }

        @objc func tapped(_ gesture: UITapGestureRecognizer) {
            guard !model.isMoving, let view = gesture.view as? SCNView else { return }
            let hit = view.hitTest(gesture.location(in: view), options: nil).first
            model.select(hit?.node.name)
        }

        /// Arrastre: desplaza el objeto en horizontal, a la misma profundidad de pantalla en que está.
        @objc func panned(_ gesture: UIPanGestureRecognizer) {
            guard let view = gesture.view as? SCNView, let id = model.selectedID else { return }
            let point = gesture.location(in: view)
            switch gesture.state {
            case .began:
                model.beginEdit(id)
                lastPoint = point
            case .changed:
                guard let center = model.worldCenter(of: id) else { return }
                let depth = view.projectPoint(center).z
                let from = view.unprojectPoint(SCNVector3(Float(lastPoint.x), Float(lastPoint.y), depth))
                let to = view.unprojectPoint(SCNVector3(Float(point.x), Float(point.y), depth))
                model.translate(id, by: SIMD3(to.x - from.x, 0, to.z - from.z))
                lastPoint = point
            case .ended, .cancelled, .failed:
                model.commitEdit()
            default:
                break
            }
        }

        @objc func rotated(_ gesture: UIRotationGestureRecognizer) {
            guard let id = model.selectedID else { return }
            switch gesture.state {
            case .began:
                model.beginEdit(id)
            case .changed:
                model.rotate(id, by: -Float(gesture.rotation))
                gesture.rotation = 0
            case .ended, .cancelled, .failed:
                model.commitEdit()
            default:
                break
            }
        }
    }
}
