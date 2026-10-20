import SceneKit
import SwiftUI
import Observation

/// Visor 3D de un escaneo de entorno, con su color real (SceneKit sí muestra el color por vértice).
/// Cada objeto detectado es un nodo independiente: se puede tocar, aislar, ocultar y exportar.
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

    var items: [Item] = []
    var hidden: Set<String> = []
    var selectedID: String?
    var isLoading = true
    var errorMessage: String?
    var shaded = false { didSet { applyAppearance() } }
    var cameraNode: SCNNode?
    var target = SCNVector3Zero

    let scene = SCNScene()
    @ObservationIgnored private var nodes: [String: SCNNode] = [:]
    @ObservationIgnored private var meshes: [String: ColoredMesh] = [:]
    @ObservationIgnored private var folder: URL?

    var selectedItem: Item? { items.first { $0.id == selectedID } }

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

        // Textura fotográfica compartida por todos los objetos (si el escaneo la tiene).
        let textureURL = folder.appendingPathComponent("textura.jpg")
        let texture = UIImage(contentsOfFile: textureURL.path)

        guard !loaded.isEmpty else {
            errorMessage = "No se encontró una malla con color en este escaneo."
            isLoading = false
            return
        }

        var minP = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maxP = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for (item, mesh) in loaded {
            let node = SCNNode(geometry: ModelExporter.makeGeometry(mesh, texture: texture))
            node.name = item.id
            scene.rootNode.addChildNode(node)
            nodes[item.id] = node
            meshes[item.id] = mesh
            for p in mesh.positions {
                minP = simd_min(minP, p)
                maxP = simd_max(maxP, p)
            }
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

    /// Exporta a OBJ solo los objetos visibles (la escena "editada").
    func exportVisible() -> URL? {
        guard let folder else { return nil }
        let parts = items.filter { !hidden.contains($0.id) }.compactMap { item in
            meshes[item.id].map { ModelExporter.Part(name: item.title, mesh: $0) }
        }
        guard !parts.isEmpty else { return nil }
        let url = folder.appendingPathComponent("escena_editada.obj")
        let material = FileManager.default.fileExists(atPath: folder.appendingPathComponent("modelo.mtl").path) ? "modelo.mtl" : nil
        return (try? ModelExporter.writeOBJ(parts: parts, to: url, materialFile: material)) != nil ? url : nil
    }

    private func applyAppearance() {
        for (id, node) in nodes {
            node.isHidden = hidden.contains(id)
            node.opacity = (selectedID == nil || selectedID == id) ? 1 : 0.25
            node.geometry?.firstMaterial?.lightingModel = shaded ? .lambert : .constant
        }
    }
}

struct SceneViewerView: View {
    let folder: URL
    @State private var model = SceneViewerModel()
    @State private var showList = false
    @State private var exportedURL: URL?

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
                Text("Toca un objeto para seleccionarlo · arrastra para girar · pellizca para acercar")
                    .font(.caption)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.bottom, 8)
            }
        }
        .navigationTitle("Visor 3D")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button { showList = true } label: { Image(systemName: "list.bullet") }
                    .disabled(model.items.count < 2)
                Menu {
                    Toggle("Sombreado (ver relieve)", isOn: $model.shaded)
                    Button("Mostrar todo", systemImage: "eye") { model.showAll() }
                    Button("Exportar lo visible (OBJ)", systemImage: "square.and.arrow.up") {
                        exportedURL = model.exportVisible()
                    }
                } label: { Image(systemName: "ellipsis.circle") }
            }
        }
        .sheet(isPresented: $showList) { objectList }
        .sheet(item: $exportedURL) { url in
            NavigationStack {
                ExportedFilesView(title: "Escena editada", files: [url])
            }
            .presentationDetents([.medium])
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
            HStack(spacing: 10) {
                Button("Aislar", systemImage: "scope") { model.isolate(item.id) }
                Button("Ocultar", systemImage: "eye.slash") { model.toggleHidden(item.id) }
                ShareLink(item: item.file) { Label("Exportar", systemImage: "square.and.arrow.up") }
            }
            .buttonStyle(.bordered)
            .font(.subheadline)
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .padding()
    }

    private var objectList: some View {
        NavigationStack {
            List {
                ForEach(model.items) { item in
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

extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
}

/// `SCNView` con cámara orbital y selección por toque.
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
        let tap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.tapped(_:)))
        view.addGestureRecognizer(tap)
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        if let camera = model.cameraNode, view.pointOfView !== camera {
            view.pointOfView = camera
            view.defaultCameraController.target = model.target
        }
    }

    @MainActor
    final class Coordinator: NSObject {
        let model: SceneViewerModel
        init(model: SceneViewerModel) { self.model = model }

        @objc func tapped(_ gesture: UITapGestureRecognizer) {
            guard let view = gesture.view as? SCNView else { return }
            let hit = view.hitTest(gesture.location(in: view), options: nil).first
            model.select(hit?.node.name)
        }
    }
}
